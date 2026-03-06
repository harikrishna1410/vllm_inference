import concurrent.futures
import os
import random
import time
import uuid
from typing import List

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.orchestrator import ClusterClient
from utils import get_logger, parse_args, submit_prompt, wait_for_vllm

logger = get_logger("main_multi_node", log_dir=f"{os.getcwd()}/script_logs")

UTILS_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils.py")


def create_prompt(nprompts) -> List[str]:
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


def main():
    t_start = time.time()
    logger.info("main started")

    args_dict = parse_args()
    nodes = get_nodes()
    nnodes = len(nodes)
    logger.info("running on %d nodes: %s", nnodes, nodes)

    # Setup EnsembleLauncher
    cpus = list(range(104))
    cpus.pop(52)
    cpus.pop(0)
    sys_config = SystemConfig(
        name="aurora", ncpus=102, ngpus=12, cpus=cpus, gpus=list(range(12))
    )
    launcher_config = LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name=["async_processpool", "async_mpi"],
        nlevels=1,
        nchildren=nnodes,
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        children_scheduler_policy="simple_split_children_policy",
        checkpoint_dir=f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}",
        report_interval=10.0,
    )

    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )

    t0 = time.time()
    logger.info("starting EnsembleLauncher")
    el.start()
    time.sleep(5.0)
    logger.info("EnsembleLauncher ready (%.1fs)", time.time() - t0)

    model_dir = os.path.join(
        args_dict["cache_dir"],
        "hub",
        f"models--{args_dict['model'].replace('/', '--')}",
    )
    local_cache = os.path.join("/tmp", f"{str(uuid.uuid4())}")
    copy_dir = os.path.join(
        local_cache,
        "hub",
        f"models--{args_dict['model'].replace('/', '--')}",
    )
    cache_dir = local_cache

    ngpus = args_dict["ngpus_per_model"]
    master_addr = nodes[0]
    master_port = random.randint(20000, 30000)

    vllm_start_futures = []
    with ClusterClient(checkpoint_dir=launcher_config.checkpoint_dir) as client:
        if not os.path.exists(copy_dir):
            t0 = time.time()
            logger.info("mkdir on %d nodes", nnodes)
            copy_futures = []
            for node in nodes:
                future = client.submit(f"mkdir -p {copy_dir}", nnodes=1, ppn=1)
                copy_futures.append(future)

            concurrent.futures.wait(copy_futures)
            logger.info("mkdir done (%.1fs)", time.time() - t0)

            t0 = time.time()
            logger.info(
                "dsync model to local cache on %d nodes: %s -> %s",
                nnodes,
                model_dir,
                copy_dir,
            )
            copy_futures = []
            for node in nodes:
                task = Task(
                    task_id=str(uuid.uuid4()),
                    nnodes=1,
                    ppn=90,
                    executable=f"dsync {model_dir + '/'} {copy_dir + '/'}",
                    executor_name="async_mpi",
                )
                future = client.submit(task=task)
                copy_futures.append(future)

            concurrent.futures.wait(copy_futures)
            logger.info("dsync done (%.1fs)", time.time() - t0)

            if all([fut.exception() is None for fut in copy_futures]):
                cache_dir = local_cache
            else:
                raise RuntimeError("Copying models failed")

        # Start NGPUS engine processes per node, one per GPU
        t0 = time.time()
        logger.info(
            "starting vllm engines on %d nodes (%d procs/node, port=%s, master=%s:%d)",
            nnodes,
            ngpus,
            args_dict["port"],
            master_addr,
            master_port,
        )
        for idx in range(nnodes):
            env = {
                "MASTER_ADDR": f"{master_addr}",
                "MASTER_PORT": f"{master_port}",
                "WORLD_SIZE": f"{nnodes * ngpus} ",
                "RANK_OFFSET": f"{idx * ngpus} ",
                "NNODES": f"{nnodes}",
            }
            cmd = (
                f"{os.getcwd()}/start_vllm_engine.sh "
                f"{idx} {args_dict['port']} {ngpus} "
                f"{args_dict['model']} {cache_dir} {args_dict['tmp_dir']}"
            )
            task = Task(
                task_id=str(uuid.uuid4()),
                nnodes=1,
                ppn=ngpus,
                ngpus_per_process=1,
                executable=cmd,
                executor_name="async_mpi",
                env=env,
            )
            vllm_start_futures.append(client.submit(task=task))
            logger.info("vllm engines submitted for node index %d", idx)

        # Wait for vllm engine to be ready via MPI (mirrors the no-EL mpirun wait)
        wait_task = Task(
            task_id=str(uuid.uuid4()),
            nnodes=1,
            ppn=ngpus,
            executable=(
                f"python {UTILS_PY} --mode wait "
                f"--port {args_dict['port']} "
                f"--model {args_dict['model']} "
                f"--key {args_dict['key']}"
            ),
            executor_name="async_mpi",
        )
        wait_future = client.submit(task=wait_task)
        concurrent.futures.wait([wait_future])
        if wait_future.exception() is not None:
            raise RuntimeError(f"vllm wait failed: {wait_future.exception()}")
        logger.info("vllm engine ready (%.1fs since launch)", time.time() - t0)

        # prompts = create_prompt(args_dict["num_prompts"])
        # logger.info("submitting %d prompts", len(prompts))
        # t_prompts = time.time()

        # prompt_futures = client.map(
        #     submit_prompt, zip(prompts, [args_dict] * args_dict["num_prompts"])
        # )
        # results = []
        # for fut in concurrent.futures.as_completed(prompt_futures):
        #     result = fut.result()
        #     e = fut.exception()
        #     logger.info("prompt done: result=%s exception=%s", result, e)
        #     results.append(result)

        # logger.info("all prompts done (%.1fs)", time.time() - t_prompts)

        # Stop the vllm engines
        t0 = time.time()
        logger.info("stopping vllm engines")
        stop_futures = []
        for idx in range(nnodes):
            task = Task(
                task_id=str(uuid.uuid4()),
                nnodes=1,
                ppn=1,
                executable=f"{os.getcwd()}/stop_vllm_server.sh",
                executor_name="async_mpi",
            )
            stop_futures.append(client.submit(task))

        concurrent.futures.wait(stop_futures)
        logger.info("vllm engines stopped (%.1fs)", time.time() - t0)

    # Stop the cluster
    t0 = time.time()
    logger.info("stopping EnsembleLauncher")
    el.stop()
    logger.info("EnsembleLauncher stopped (%.1fs)", time.time() - t0)

    logger.info("main done (total %.1fs)", time.time() - t_start)


if __name__ == "__main__":
    main()
