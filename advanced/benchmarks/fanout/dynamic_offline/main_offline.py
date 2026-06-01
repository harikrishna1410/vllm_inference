import asyncio
import multiprocessing as mp
import os
import secrets
import time
import uuid
from typing import List

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.comm import AsyncZMQTransport, transport_registry
from ensemble_launcher.config import (
    LauncherConfig,
    MPIConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.ensemble.actor import PrivateActorHandle
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.inference import PrivateVLLMInference, copy_model
from ensemble_launcher.orchestrator import ClusterClient
from utils import get_logger, parse_args

logger = get_logger("main_offline", log_dir=f"{os.getcwd()}/script_logs")


def create_prompt(nprompts) -> List[str]:
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


async def Warmup(args_dict, vllm_cache):
    transport: AsyncZMQTransport = transport_registry.get("zmq")["transport"]()
    server, client = transport.create_child_pipe("server", "secret", "actor", "secret")
    actor = PrivateVLLMInference(
        name="warmpup-actor",
        model=args_dict["model"],
        cache_dir=args_dict["cache_dir"],
        tensor_parallel_size=args_dict["ngpus_per_model"],
        client_conn=client,
        model_info_cache=vllm_cache,
    )
    os.environ["ZE_AFFINITY_MASK"] = ",".join(
        map(str, list(range(args_dict["ngpus_per_model"])))
    )
    mp.set_start_method("spawn")
    p = mp.Process(target=actor)
    p.start()
    try:
        handle = PrivateVLLMInference.create_handle(server)
        await handle.open()
        await handle.send(("generate", ("hello",), ()), "actor:secret")
        result = await handle.recv()
        await handle.stop()
        await handle.close()
    finally:
        p.join(30.0)
        if p.is_alive():
            p.kill()
        os.environ.pop("ZE_AFFINITY_MASK")
    return result


async def async_main():
    t_start = time.time()
    logger.info("main_offline started")

    args_dict = parse_args()
    nodes = get_nodes()

    local_cache = os.path.join("/tmp", "model_cache")

    pre_build_vllm_cache = args_dict["pre_build_vllm_cache"] == 1
    vllm_cache = os.path.join(os.getcwd(), "vllm_cache")
    node_local_vllm_cache = "/tmp/vllm_cache"
    if pre_build_vllm_cache:
        # Build vllm cache
        try:
            result = await asyncio.wait_for(Warmup(args_dict, vllm_cache), timeout=600)
            logger.info(f"Warmup returned result: {result}")
        except Exception as e:
            logger.warning(f"Warmup failed with error {e}")

    ## Copy model and vllm_cache to /tmp
    tic = time.perf_counter()
    copy_model.sync_to_root(
        model=args_dict["model"],
        cache_dir=args_dict["cache_dir"],
        np=102,
        node_local_cache=local_cache,
        logger=logger,
        cache_modelinfo=pre_build_vllm_cache,
        vllm_cache=vllm_cache,
        node_local_vllm_cache=node_local_vllm_cache,
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
        cache_modelinfo=pre_build_vllm_cache,
        node_local_vllm_cache=node_local_vllm_cache,
    )
    logger.info(f"Copying model took {time.perf_counter() - tic}s")

    # Create EL
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

    # Create transport
    transport: AsyncZMQTransport = transport_registry.get("zmq")["transport"]()

    actors = []
    actor_tasks = []

    server_id = "actor_handle"
    server_secret = secrets.token_hex(16)

    n_actors = 12 * len(nodes) // args_dict["ngpus_per_model"]
    for i in range(n_actors):
        actor_id = f"vllm-actor-{i}"

        server, client = transport.create_child_pipe(
            server_id, server_secret, actor_id, server_secret
        )

        actor = PrivateVLLMInference(
            name=actor_id,
            model=args_dict["model"],
            cache_dir=local_cache,
            tensor_parallel_size=args_dict["ngpus_per_model"],
            client_conn=client,
            use_cached_modelinfo=pre_build_vllm_cache,
            model_info_cache=node_local_vllm_cache if pre_build_vllm_cache else None,
        )
        actors.append(actor)
        actor_tasks.append(
            actor.create_task(
                actor_id,
                nnodes=1,
                ppn=args_dict["ngpus_per_model"] * 2,
                ngpus_per_process=1 / 2,
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
        prompts = create_prompt(args_dict["num_prompts"]) * n_actors
        chunks = [[] for _ in range(n_actors)]
        for i, prompt in enumerate(prompts):
            chunks[i % n_actors].append(prompt)

        # create a single handle for all the actors
        handle: PrivateActorHandle = PrivateVLLMInference.create_handle(server)
        await handle.open()

        ## Wait for ready tasks
        logger.info("Waiting for actors to be ready...")
        await handle.wait_for_ready(expected=n_actors)

        send_tasks = []
        loop = asyncio.get_running_loop()
        for i, actor in enumerate(actors):
            if not chunks[i]:
                continue
            send_tasks.append(
                loop.create_task(
                    handle.send(
                        ("generate", (chunks[i],), None),
                        target_id=f"vllm-actor-{i}:{server_secret}",
                    )
                )
            )
            logger.info(f"Created send prompts task for actor-{i}")

        logger.info("Waiting to recv results...")

        start = time.perf_counter()
        done = set()
        all_results = []
        while len(done) < n_actors:
            try:
                result = await asyncio.wait_for(handle.recv(), timeout=5.0)
                all_results.append(result[1])
                done.add(result[0])
            except Exception:
                for i, f in enumerate(actor_futures):
                    actor_id = f"vllm-actor-{i}:{server_secret}"
                    if actor_id in done:
                        continue
                    if f.done():
                        done.add(actor_id)
                        if f.exception() is not None:
                            logger.error(
                                f"Actor {actor_id} failed with exception {f.exception()}"
                            )

        logger.info("all prompts done (%.1fs)", time.perf_counter() - start)

        # Stop actors
        t0 = time.time()
        logger.info("stopping actors")
        await handle.broadcast(("stop", (), None), expected=n_actors)

        for future in actor_futures:
            try:
                future.result(timeout=60.0)
            except Exception as e:
                logger.warning("actor future exception: %s", e)

        logger.info("actors stopped (%.1fs)", time.time() - t0)

    await handle.close()

    t0 = time.time()
    logger.info("stopping EnsembleLauncher")
    el.stop()
    logger.info("EnsembleLauncher stopped (%.1fs)", time.time() - t0)

    logger.info("main_offline done (total %.1fs)", time.time() - t_start)


if __name__ == "__main__":
    asyncio.run(async_main())
