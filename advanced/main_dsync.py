import concurrent.futures
import os
import time
import uuid
from typing import List
import json
import sys
from datetime import datetime

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig, PolicyConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.orchestrator import ClusterClient
from utils import get_logger, parse_args, submit_prompt, wait_for_vllm

logger = get_logger("main_el", log_dir=f"{os.getcwd()}/script_logs")


def print_with_timestamp(message):
    """Helper function to print messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def create_prompt(nprompts) -> List[str]:
    # Send prompts
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]

def main():
    t_start = time.time()
    logger.info("main started")

    args_dict = parse_args()
    print_with_timestamp(f"Started main_dsync.py with args: {args_dict}")

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
    #launcer_config = LauncherConfig(
    #    child_executor_name="async_mpi",
    #    task_executor_name=["async_processpool", "async_mpi"],
    #    nlevels=1,
    #    nchildren=len(get_nodes()),
    #    cluster=True,
    #    worker_logs=True,
    #    master_logs=True,
    #    return_stdout=True,
    #    children_scheduler_policy="simple_split_children_policy",
    #    checkpoint_dir=f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}",
    #    report_interval=10.0,
    #    results_flush_interval=0.5,
    #)
    launcer_config = LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name=["async_processpool", "async_mpi"],
        policy_config=PolicyConfig(nlevels=1, nchildren=len(get_nodes())),
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        children_scheduler_policy="simple_split_children_policy",
        checkpoint_dir=f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}",
        report_interval=10.0,
        results_flush_interval=0.5,
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
    print_with_timestamp(f"EnsembleLauncher is ready! (time: {time.time() - t0} seconds)")

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
    print_with_timestamp(f"Loaded {len(prompts)} prompts for inference on each vllm server")
    
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
    with ClusterClient(checkpoint_dir=launcer_config.checkpoint_dir, n_workers=4) as client:
        # Start ONE vllm server per node
        t0 = time.time()
        logger.info(
            "starting vllm servers on %d nodes (port=%s, ngpus_per_model=%d)",
            len(get_nodes()),
            args_dict["port"],
            args_dict["ngpus_per_model"],
        )
        print_with_timestamp(f"Starting vllm servers on {len(get_nodes())} nodes (port={args_dict['port']}, ngpus_per_model={args_dict['ngpus_per_model']}) time: {time.time() - t_start} seconds")
        for vllm_idx in range(len(get_nodes())):
            vllm_start_futures.append(
                client.submit(
                    f"{os.getcwd()}/start_vllm_server_dsync.sh {vllm_idx} {args_dict['port']} {args_dict['ngpus_per_model']} {args_dict['model']} {cache_dir} {args_dict['tmp_dir']}",
                    ppn=1, # NB limited to 1 CPU core per task
                    ngpus_per_process=1, # NB equal to TP size
                )
            )

        ##Wait for them to finish
        print_with_timestamp(f"Waiting for vllm servers to be ready ... (time: {time.time() - t_start} seconds)")
        vllm_wait_futures = []
        for i in range(len(get_nodes())):
            vllm_wait_futures.append(
                client.submit(wait_for_vllm, args_dict, check_interval=1)
            )

        concurrent.futures.wait(vllm_wait_futures)
        exceptions = [fut.exception() for fut in vllm_wait_futures]
        print_with_timestamp(f"Vllm servers ready! (time: {time.time() - t_start} seconds)")

        # Submit prompts
        if all([e is None for e in exceptions]):
            logger.info("all vllm servers ready (%.1fs since launch)", time.time() - t0)
            print_with_timestamp(f"All vllm servers ready! (time: {time.time() - t0} seconds)")

            #prompts = create_prompt(args_dict["num_prompts"]) * len(get_nodes())
            prompts = prompts * len(get_nodes())
            logger.info("submitting %d prompts", len(prompts))
            print_with_timestamp(f"Submitting {len(prompts)} total prompts time: {time.time() - t_start} seconds")
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
                if len(results) == 1:
                    print_with_timestamp(f"Received first prompt (time: {time.time() - t_start} seconds)")

            logger.info("all prompts done (%.1fs)", time.time() - t_prompts)
            print_with_timestamp(f"All prompts done! (time: {time.time() - t_start} seconds, inference time: {time.time() - t_prompts} seconds)")

        # stop the vllm servers
        t0 = time.time()
        logger.info("stopping vllm servers")
        print_with_timestamp(f"Stopping vllm servers time: {time.time() - t_start} seconds")
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
        print_with_timestamp(f"vllm servers stopped! time: {time.time() - t0} seconds")

    # stop the cluster
    t0 = time.time()
    logger.info("stopping EnsembleLauncher")
    print_with_timestamp(f"Stopping EnsembleLauncher time: {time.time() - t_start} seconds")
    el.stop()
    logger.info("EnsembleLauncher stopped (%.1fs)", time.time() - t0)
    print_with_timestamp(f"EnsembleLauncher stopped! time: {time.time() - t_start} seconds")
    logger.info("main done (total %.1fs)", time.time() - t_start)
    print_with_timestamp(f"main done! time: {time.time() - t_start} seconds")


if __name__ == "__main__":
    main()
