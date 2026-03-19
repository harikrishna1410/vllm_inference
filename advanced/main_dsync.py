import concurrent.futures
import os
import time
import uuid
from typing import List
import json
import sys

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.orchestrator import ClusterClient
from utils import get_logger, parse_args, submit_prompt, wait_for_vllm

logger = get_logger("main_el", log_dir=f"{os.getcwd()}/script_logs")


def create_prompt(nprompts) -> List[str]:
    # Send prompts
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


def main():
    t_start = time.time()
    logger.info("main started")

    args_dict = parse_args()
    print("Started main_dsync.py with args: ", args_dict, flush=True)

    # Setup the vllm servers 1 per node
    cpus = list(range(args_dict["num_cpus_per_node"]))
    cpus.pop(0)
    if args_dict["num_cpus_per_node"] == 104:
        cpus.pop(51)
    sys_config = SystemConfig(
        name="aurora", 
        ncpus=len(cpus), 
        ngpus=args_dict["num_gpus_per_node"], 
        cpus=cpus, 
        gpus=list(range(args_dict["num_gpus_per_node"]))
    )
    launcer_config = LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name=["async_processpool", "async_mpi"],
        nlevels=1,
        nchildren=len(get_nodes()),
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        children_scheduler_policy="simple_split_children_policy",
        checkpoint_dir=f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}",
        report_interval=10.0,
    )

    # No initial tasks
    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcer_config
    )

    t0 = time.time()
    logger.info("starting EnsembleLauncher")
    el.start()
    time.sleep(5.0)
    logger.info("EnsembleLauncher ready (%.1fs)", time.time() - t0)
    print(f"EnsembleLauncher is ready! (time: {time.time() - t0} seconds)", flush=True)

    # Load prompts
    prompts = []
    prompt_path = "/tmp/hf_home/prompts.jsonl"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data["prompt"])
        num_prompts = len(prompts)
    except FileNotFoundError:
        logger.error(f"ERROR: prompts file not found: {prompt_path}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"ERROR: failed reading prompts from {prompt_path}: {type(e).__name__}: {e}")
        sys.exit(2)
    print(f"Loaded {len(prompts)} prompts for inference on each vllm server", flush=True)
    
    # Preprocessing
    model_dir = args_dict["cache_dir"]
    local_cache = os.path.join("/tmp", f"{str(uuid.uuid4())}")
    copy_dir = os.path.join(
        local_cache,
        "hub",
        f"models--{args_dict['model'].replace('/', '--')}",
    )
    cache_dir = local_cache
    vllm_start_futures = []
    with ClusterClient(checkpoint_dir=launcer_config.checkpoint_dir) as client:
        #if not os.path.exists(copy_dir):
        #    t0 = time.time()
        #    logger.info("mkdir on %d nodes", len(get_nodes()))
        #    copy_futures = []
        #    for node in get_nodes():
        #        future = client.submit(f"mkdir -p {copy_dir}", nnodes=1, ppn=1)
        #        copy_futures.append(future)

        #    concurrent.futures.wait(copy_futures)
        #    logger.info("mkdir done (%.1fs)", time.time() - t0)

        #    t0 = time.time()
        #    logger.info(
        #        "dsync model to local cache on %d nodes: %s -> %s",
        #        len(get_nodes()),
        #        model_dir,
        #        copy_dir,
        #    )
        #    copy_futures = []
        #    for node in get_nodes():
        #        task = Task(
        #            task_id=str(uuid.uuid4()),
        #            nnodes=1,
        #            ppn=90,
        #            executable=f"dsync {model_dir + '/'} {copy_dir + '/'}",
        #            executor_name="async_mpi",
        #        )
        #        future = client.submit(task=task)
        #        copy_futures.append(future)

        #    concurrent.futures.wait(copy_futures)
        #    logger.info("dsync done (%.1fs)", time.time() - t0)

        #    if all([fut.exception() is None for fut in copy_futures]):
        #        cache_dir = local_cache
        #    else:
        #        raise RuntimeError("Copying models failed")

        # Start ONE vllm server per node
        t0 = time.time()
        logger.info(
            "starting vllm servers on %d nodes (port=%s, ngpus_per_model=%d)",
            len(get_nodes()),
            args_dict["port"],
            args_dict["ngpus_per_model"],
        )
        print(f"Starting vllm servers on {len(get_nodes())}", 
              f"nodes (port={args_dict['port']}, ngpus_per_model={args_dict['ngpus_per_model']})", 
              f"time: {time.time() - t_start} seconds", flush=True)
        for vllm_idx in range(len(get_nodes())):
            vllm_start_futures.append(
                client.submit(
                    f"{os.getcwd()}/start_vllm_server_dsync.sh {vllm_idx} {args_dict['port']} {args_dict['ngpus_per_model']} {args_dict['model']} {cache_dir} {args_dict['tmp_dir']}",
                    ppn=args_dict["ngpus_per_model"],
                    ngpus_per_process=1,
                )
            )

        ##Wait for them to finish
        vllm_wait_futures = []
        for i in range(len(get_nodes())):
            vllm_wait_futures.append(client.submit(wait_for_vllm, args_dict))

        concurrent.futures.wait(vllm_wait_futures)
        exceptions = [fut.exception() for fut in vllm_wait_futures]

        # Submit prompts
        if all([e is None for e in exceptions]):
            logger.info("all vllm servers ready (%.1fs since launch)", time.time() - t0)

            #prompts = create_prompt(args_dict["num_prompts"]) * len(get_nodes())
            prompts = prompts * len(get_nodes())
            logger.info("submitting %d prompts", len(prompts))
            print(f"Submitting {len(prompts)} total prompts", flush=True)
            t_prompts = time.time()

            prompt_futures = client.map(
                submit_prompt,
                zip(prompts, [args_dict] * len(prompts) * len(get_nodes())),
            )
            results = []
            for fut in concurrent.futures.as_completed(prompt_futures):
                result = fut.result()
                e = fut.exception()
                logger.info("prompt done: result=%s exception=%s", result, e)
                results.append(result)

            logger.info("all prompts done (%.1fs)", time.time() - t_prompts)
            print(f"All prompts done! (time: {time.time() - t_prompts} seconds)", flush=True)

        # stop the vllm servers
        t0 = time.time()
        logger.info("stopping vllm servers")
        print(f"Stopping vllm servers", flush=True)
        stop_futures = []
        for vllm_idx in range(len(get_nodes())):
            task = Task(
                task_id=str(uuid.uuid4()),
                nnodes=1,
                ppn=1,
                executable=f"{os.getcwd()}/stop_vllm_server.sh",
                executor_name="async_mpi",
            )
            stop_futures.append(client.submit(task))

        concurrent.futures.wait(stop_futures, timeout=30)
        logger.info("vllm servers stopped (%.1fs)", time.time() - t0)
        print(f"vllm servers stopped! (time: {time.time() - t0} seconds)", flush=True)

    # stop the cluster
    t0 = time.time()
    logger.info("stopping EnsembleLauncher")
    print(f"Stopping EnsembleLauncher", flush=True)
    el.stop()
    logger.info("EnsembleLauncher stopped (%.1fs)", time.time() - t0)
    print(f"EnsembleLauncher stopped! (time: {time.time() - t0} seconds)", flush=True)
    logger.info("main done (total %.1fs)", time.time() - t_start)
    print(f"main done! (time: {time.time() - t_start} seconds)", flush=True)


if __name__ == "__main__":
    main()
