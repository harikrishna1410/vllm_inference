import asyncio
import os
import time
import uuid
from typing import List

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import (
    LauncherConfig,
    MPIConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.inference import VLLMInference, copy_model
from ensemble_launcher.orchestrator import ClusterClient
from utils import get_logger, parse_args

logger = get_logger("main_offline", log_dir=f"{os.getcwd()}/script_logs")


def create_prompt(nprompts) -> List[str]:
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


async def async_main():
    t_start = time.time()
    logger.info("main_offline started")

    args_dict = parse_args()
    nodes = get_nodes()

    local_cache = os.path.join("/tmp", "model_cache")

    tic = time.perf_counter()
    copy_model.sync_to_root(
        model=args_dict["model"],
        cache_dir=args_dict["cache_dir"],
        np=102,
        node_local_cache=local_cache,
        logger=logger,
    )
    logger.info("Done sync to root")
    copy_model.scatter_from_root(
        model=args_dict["model"],
        nnodes=len(nodes),
        node_local_cache=local_cache,
        chunk_size=100 * 1024 * 1024,
        ppn=8,
        logger=logger,
        cpu_binding="--cpu-bind=list:1-12,13-24,25-36,37-48,53-64,65-76,77-88,89-100",
    )
    logger.info(f"Copying model took {time.perf_counter() - tic}s")

    cpus = list(range(104))
    cpus.pop(52)
    cpus.pop(0)
    sys_config = SystemConfig(
        name="aurora", ncpus=102, ngpus=12, cpus=cpus, gpus=list(range(12))
    )
    ckpt_dir = f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}"
    launcher_config = LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name=["async_processpool", "async_mpi"],
        comm_name="async_zmq",
        policy_config=PolicyConfig(nlevels=1, nchildren=len(nodes)),
        mpi_config=MPIConfig(flavor="mpich", cpu_bind_method="none"),
        cluster=True,
        worker_logs=True,
        master_logs=True,
        return_stdout=True,
        checkpoint_dir=ckpt_dir,
        report_interval=10.0,
        task_flush_interval=0.5,
        result_flush_interval=0.5,
    )

    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launcher_config
    )

    t0 = time.time()
    logger.info("starting EnsembleLauncher")
    el.start()
    await asyncio.sleep(10.0)
    logger.info("EnsembleLauncher ready (%.1fs)", time.time() - t0)

    actors = []
    actor_tasks = []
    handles = []

    for i in range(len(nodes)):
        actor_id = f"vllm-actor-{i}"

        actor = VLLMInference(
            name=actor_id,
            transport="zmq",
            model=args_dict["model"],
            cache_dir=local_cache,
            tensor_parallel_size=args_dict["ngpus_per_model"],
        )
        actors.append(actor)
        actor_tasks.append(
            actor.create_task(
                actor_id,
                nnodes=1,
                ppn=args_dict["ngpus_per_model"] * 2,
                ngpus_per_process=1 / 2,
                # cpu_affinity=[1, 9, 17, 25, 33, 41, 53, 61],
            )
        )

    # Submit actors and run inference
    with ClusterClient(checkpoint_dir=ckpt_dir, checkpoint_timeout=300) as client:
        t0 = time.time()
        actor_futures = []
        for task in actor_tasks:
            future = client.submit(task)
            actor_futures.append(future)

        logger.info("submitted %d actor tasks", len(actor_tasks))

        # Send prompts as batches to each actor
        prompts = create_prompt(args_dict["num_prompts"]) * len(nodes)
        chunks = [[] for _ in range(len(nodes))]
        for i, prompt in enumerate(prompts):
            chunks[i % len(nodes)].append(prompt)

        t_prompts = time.time()

        for i, actor in enumerate(actors):
            if not chunks[i]:
                continue
            logger.info(f"Waiting to create handle for actor-{i}")
            handle = actor.create_handle(timeout=600)
            if handle is not None:
                await handle.open()
                await handle.send(("generate", chunks[i]))
                logger.info(f"Submitted prompts to actor-{i}")
                handles.append(handle)
            else:
                if actor_futures[i].done() and actor_futures[i].exception() is not None:
                    logger.warning(
                        f"actor {i} failed with exception. {actor_futures[i].exception()}"
                    )
                el.stop()
                return

        # Collect results
        all_results = []
        done = set()
        while len(all_results) < len(prompts) and len(done) < len(actor_futures):
            for i, handle in enumerate(handles):
                if i in done:
                    continue
                if not chunks[i]:
                    continue
                try:
                    results = await asyncio.wait_for(handle.recv(), timeout=10.0)
                    logger.info(f"actor {i} returned results {results}")
                    all_results.extend(results)
                    done.add(i)
                except Exception as e:
                    logger.info(f"Waiting for results from actor-{i} timed out: {e}")

                if actor_futures[i].done() and actor_futures[i].exception() is not None:
                    logger.info(
                        f"actor {i} died with: {actor_futures[i].exception()}"
                    )
                    done.add(i)

        if len(all_results) == len(prompts):
            logger.info("all prompts done (%.1fs)", time.time() - t_prompts)

        for result in all_results:
            logger.info("result: %s", result[:200] if result else result)

        # Stop actors
        t0 = time.time()
        logger.info("stopping actors")
        for handle in handles:
            await handle.stop()

        for future in actor_futures:
            try:
                future.result(timeout=60.0)
            except Exception as e:
                logger.warning("actor future exception: %s", e)

        logger.info("actors stopped (%.1fs)", time.time() - t0)

    # Close ZMQ connections
    for handle in handles:
        await handle.close()

    t0 = time.time()
    logger.info("stopping EnsembleLauncher")
    el.stop()
    logger.info("EnsembleLauncher stopped (%.1fs)", time.time() - t0)

    logger.info("main_offline done (total %.1fs)", time.time() - t_start)


if __name__ == "__main__":
    asyncio.run(async_main())
