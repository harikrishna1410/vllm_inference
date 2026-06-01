import asyncio
import concurrent.futures
import multiprocessing as mp
import os
import queue
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


async def _async_worker_process(
    worker_id,
    actor_indices,
    args_dict,
    local_cache,
    pre_build_vllm_cache,
    node_local_vllm_cache,
    ckpt_dir,
    result_queue,
    start_event,
):
    """Async logic executed within each separate process."""
    worker_logger = get_logger(
        f"worker_{worker_id}", log_dir=f"{os.getcwd()}/script_logs"
    )
    worker_logger.info(
        f"Worker {worker_id} started handling {len(actor_indices)} actors."
    )

    transport: AsyncZMQTransport = transport_registry.get("zmq")["transport"]()
    server_id = f"actor_handle_worker_{worker_id}"
    server_secret = secrets.token_hex(16)

    actors = []
    actor_tasks = []

    for i in actor_indices:
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

    with ClusterClient(
        checkpoint_dir=ckpt_dir, checkpoint_timeout=300
    ) as client_cluster:
        actor_futures = [client_cluster.submit(task) for task in actor_tasks]
        worker_logger.info(f"Worker {worker_id} submitted {len(actor_tasks)} tasks.")

        handle: PrivateActorHandle = PrivateVLLMInference.create_handle(server)
        await handle.open()

        worker_logger.info(f"Worker {worker_id} waiting for actors to be ready...")
        await handle.wait_for_ready(expected=len(actor_indices))

        # --- SYNC: Signal ready and wait for global start ---
        result_queue.put(("READY", worker_id))
        worker_logger.info(
            f"Worker {worker_id} ready. Waiting for global start signal..."
        )

        while not start_event.is_set():
            await asyncio.sleep(0.01)

        worker_logger.info(
            f"Worker {worker_id} received start signal. Dispatching prompts..."
        )
        # ---------------------------------------------------

        prompts = create_prompt(args_dict["num_prompts"])

        done = set()
        for i, actor_idx in enumerate(actor_indices):
            target_id = f"vllm-actor-{actor_idx}:{server_secret}"
            try:
                await handle.send(
                    ("generate", (prompts,), None),
                    target_id=target_id,
                )
                worker_logger.info(f"Sent prompt to {target_id}")
            except Exception as e:
                worker_logger.error(
                    f"Sending prompt to {target_id} failed with error: {e}"
                )
                done.add(target_id)
                result_queue.put(str(e))

        worker_logger.info(f"Worker {worker_id} waiting to recv results...")

        while len(done) < len(actor_indices):
            try:
                result = await asyncio.wait_for(handle.recv(), timeout=5.0)
                result_queue.put(result[1])
                done.add(result[0])
                worker_logger.info(f"Received result from {result[0]}")
            except Exception:
                for i, f in enumerate(actor_futures):
                    target_id = f"vllm-actor-{actor_indices[i]}:{server_secret}"
                    if target_id in done:
                        continue
                    if f.done():
                        done.add(target_id)
                        result_queue.put(f"ERROR: {target_id}")
                        if f.exception() is not None:
                            worker_logger.error(
                                f"Actor {target_id} failed: {f.exception()}"
                            )

        worker_logger.info(f"Worker {worker_id} stopping actors...")
        await handle.broadcast(("stop", (), None), expected=len(actor_indices))

        for future in actor_futures:
            try:
                future.result(timeout=60.0)
            except Exception as e:
                worker_logger.warning("Actor future exception: %s", e)

    await handle.close()


def run_worker_process(
    worker_id,
    actor_indices,
    args_dict,
    local_cache,
    pre_build_vllm_cache,
    node_local_vllm_cache,
    ckpt_dir,
    result_queue,
    start_event,
):
    """Synchronous entry point for the multiprocessing pool to execute async logic."""
    asyncio.run(
        _async_worker_process(
            worker_id,
            actor_indices,
            args_dict,
            local_cache,
            pre_build_vllm_cache,
            node_local_vllm_cache,
            ckpt_dir,
            result_queue,
            start_event,
        )
    )


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
        try:
            result = await asyncio.wait_for(Warmup(args_dict, vllm_cache), timeout=600)
            logger.info(f"Warmup returned result: {result}")
        except Exception as e:
            logger.warning(f"Warmup failed with error {e}")

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
        children_scheduler_policy="fixed_leafs_children_policy",
        policy_config=PolicyConfig(
            nlevels=1 if len(nodes) <= 256 else 2, leaf_nodes=len(nodes)
        ),
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

    n_actors = 12 * len(nodes) // args_dict["ngpus_per_model"]
    ACTORS_PER_PROCESS = 32 * 12

    actor_chunks = [
        list(range(i, min(i + ACTORS_PER_PROCESS, n_actors)))
        for i in range(0, n_actors, ACTORS_PER_PROCESS)
    ]

    # Initialize shared primitives via Manager
    manager = mp.Manager()
    result_queue = manager.Queue()
    start_event = manager.Event()

    logger.info(
        f"Distributing {n_actors} actors across {len(actor_chunks)} isolated processes."
    )
    loop = asyncio.get_running_loop()

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=len(actor_chunks))
    process_futures = []
    for worker_id, indices in enumerate(actor_chunks):
        future = loop.run_in_executor(
            executor,
            run_worker_process,
            worker_id,
            indices,
            args_dict,
            local_cache,
            pre_build_vllm_cache,
            node_local_vllm_cache,
            ckpt_dir,
            result_queue,
            start_event,
        )
        process_futures.append(future)

    # --- Phase 1: Wait for Initialization Barrier ---
    logger.info(f"Waiting for {len(actor_chunks)} worker clusters to initialize...")
    workers_ready = 0
    while workers_ready < len(actor_chunks):
        try:
            msg = result_queue.get_nowait()
            if isinstance(msg, tuple) and msg[0] == "READY":
                workers_ready += 1
                logger.info(
                    f"Worker {msg[1]} is fully ready ({workers_ready}/{len(actor_chunks)})."
                )
        except queue.Empty:
            await asyncio.sleep(0.01)
            # Safety check: if workers crashed during init, abort loop
            if all(f.done() for f in process_futures) and result_queue.empty():
                logger.error("Workers crashed during initialization phase.")
                break

    # --- Phase 2: Start Inference Timer & Release Barrier ---
    logger.info(
        "All actors ready. Triggering global start signal and starting timer..."
    )
    t_inference_start = time.perf_counter()
    start_event.set()

    # --- Phase 3: Result Collection ---
    results_received = 0
    while results_received < n_actors:
        try:
            res = result_queue.get_nowait()
            # Ignore duplicate/stray READY signals if any happen to leak through
            if isinstance(res, tuple) and res[0] == "READY":
                continue

            results_received += 1
            if (
                results_received % (n_actors // 10 or 1) == 0
                or results_received == n_actors
            ):
                logger.info(
                    f"Progress: {results_received}/{n_actors} results received."
                )

        except queue.Empty:
            await asyncio.sleep(0.01)
            if all(f.done() for f in process_futures) and result_queue.empty():
                logger.error(
                    "All workers finished or crashed, but expected results were not met."
                )
                break

    inference_duration = time.perf_counter() - t_inference_start
    logger.info("== INFERENCE COMPLETED ==")
    logger.info(f"Total Actors: {n_actors}")
    logger.info(f"Time to generate and drain queue: {inference_duration:.4f} seconds.")
    logger.info(f"Throughput: {n_actors / inference_duration:.2f} results/sec.")

    await asyncio.gather(*process_futures, return_exceptions=True)
    executor.shutdown()

    t0 = time.time()
    logger.info("stopping EnsembleLauncher")
    el.stop()
    logger.info("EnsembleLauncher stopped (%.1fs)", time.time() - t0)

    logger.info("main_offline done (total %.1fs)", time.time() - t_start)


if __name__ == "__main__":
    asyncio.run(async_main())
