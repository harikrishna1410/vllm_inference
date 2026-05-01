import asyncio
import concurrent.futures
import os
import secrets
import time
import uuid
from typing import List

import cloudpickle

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.comm.pipe.async_transport import AsyncZMQTransport
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.ensemble.ensemble import Actor
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.orchestrator import ClusterClient
from utils import get_logger, parse_args

logger = get_logger("main_offline", log_dir=f"{os.getcwd()}/script_logs")


class VLLMInference:
    def __init__(self, model: str, cache_dir: str, tensor_parallel_size: int = 1):
        self.model = model
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None

    def _ensure_initialized(self):
        if self._llm is None:
            from vllm import LLM

            os.environ["HF_HOME"] = self.cache_dir
            os.environ["HF_HUB_CACHE"] = os.path.join(self.cache_dir, "hub")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

            self._llm = LLM(
                model=self.model,
                tensor_parallel_size=self.tensor_parallel_size,
                download_dir=os.path.join(self.cache_dir, "hub"),
                trust_remote_code=True,
            )

    def __call__(self, prompts, temperature=0.0, max_tokens=1024):
        self._ensure_initialized()
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        outputs = self._llm.generate(prompts, sampling_params)
        results = [output.outputs[0].text for output in outputs]

        return results[0] if single else results


def create_prompt(nprompts) -> List[str]:
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


async def async_main():
    t_start = time.time()
    logger.info("main_offline started")

    args_dict = parse_args()
    nodes = get_nodes()

    cpus = list(range(104))
    cpus.pop(52)
    cpus.pop(0)
    sys_config = SystemConfig(
        name="aurora", ncpus=102, ngpus=12, cpus=cpus, gpus=list(range(12))
    )
    ckpt_dir = f"{os.getcwd()}/ckpt_{str(uuid.uuid4())}"
    launcher_config = LauncherConfig(
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        policy_config=PolicyConfig(nlevels=1, nchildren=len(nodes)),
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
    await asyncio.sleep(5.0)
    logger.info("EnsembleLauncher ready (%.1fs)", time.time() - t0)

    # Copy model to node-local storage via dsync
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

    with ClusterClient(checkpoint_dir=ckpt_dir) as client:
        if not os.path.exists(copy_dir):
            t0 = time.time()
            logger.info("mkdir on %d nodes", len(nodes))
            copy_futures = []
            for node in nodes:
                future = client.submit(f"mkdir -p {copy_dir}", nnodes=1, ppn=1)
                copy_futures.append(future)

            concurrent.futures.wait(copy_futures)
            logger.info("mkdir done (%.1fs)", time.time() - t0)

            t0 = time.time()
            logger.info(
                "dsync model to local cache on %d nodes: %s -> %s",
                len(nodes),
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

            if not all(fut.exception() is None for fut in copy_futures):
                raise RuntimeError("Copying models failed")

    # Create ZMQ transport and actor pipes
    transport = AsyncZMQTransport()
    actors = []
    servers = []
    actor_identities = []

    for i in range(len(nodes)):
        controller_id = f"controller-{i}"
        controller_secret = secrets.token_hex(16)
        actor_id = f"vllm-actor-{i}"
        actor_secret = secrets.token_hex(16)

        server, client_conn = transport.create_child_pipe(
            parent_id=controller_id,
            parent_secret=controller_secret,
            child_id=actor_id,
            child_secret=actor_secret,
        )
        await server.open()

        inference_fn = VLLMInference(
            model=args_dict["model"],
            cache_dir=cache_dir,
            tensor_parallel_size=args_dict["ngpus_per_model"],
        )

        actor = Actor(
            task_id=f"vllm-actor-task-{i}",
            nnodes=1,
            ppn=1,
            ngpus_per_process=args_dict["ngpus_per_model"],
            executable=inference_fn,
            connection=client_conn,
        )
        actors.append(actor)
        servers.append(server)
        actor_identities.append(
            (f"{actor_id}:{actor_secret}", f"{controller_id}:{controller_secret}")
        )

    # Submit actors and run inference
    with ClusterClient(checkpoint_dir=ckpt_dir) as client:
        t0 = time.time()
        logger.info("submitting %d actor tasks", len(actors))
        actor_futures = []
        for actor in actors:
            future = client.submit(actor)
            actor_futures.append(future)

        logger.info("waiting for actors to initialize (model loading)")
        await asyncio.sleep(10.0)

        # Send prompts as batches to each actor
        prompts = create_prompt(args_dict["num_prompts"])
        chunks = [[] for _ in range(len(nodes))]
        for i, prompt in enumerate(prompts):
            chunks[i % len(nodes)].append(prompt)

        t_prompts = time.time()
        logger.info(
            "sending %d prompts across %d actors", len(prompts), len(nodes)
        )

        for i, (server, (target_id, _)) in enumerate(
            zip(servers, actor_identities)
        ):
            if not chunks[i]:
                continue
            payload = cloudpickle.dumps((chunks[i],))
            await server.send(payload, target_id)

        # Collect results
        all_results = []
        for i, server in enumerate(servers):
            if not chunks[i]:
                continue
            frames = await asyncio.wait_for(server.recv(), timeout=3600.0)
            results = cloudpickle.loads(frames[1])
            logger.info("actor %d returned %d results", i, len(results))
            all_results.extend(results)

        logger.info("all prompts done (%.1fs)", time.time() - t_prompts)

        for result in all_results:
            logger.info("result: %s", result[:200] if result else result)

        # Stop actors
        t0 = time.time()
        logger.info("stopping actors")
        for server, (target_id, _) in zip(servers, actor_identities):
            stop_payload = cloudpickle.dumps("stop")
            await server.send(stop_payload, target_id)

        for future in actor_futures:
            try:
                future.result(timeout=60.0)
            except Exception as e:
                logger.warning("actor future exception: %s", e)

        logger.info("actors stopped (%.1fs)", time.time() - t0)

    # Close ZMQ connections
    for server in servers:
        await server.close()

    t0 = time.time()
    logger.info("stopping EnsembleLauncher")
    el.stop()
    logger.info("EnsembleLauncher stopped (%.1fs)", time.time() - t0)

    logger.info("main_offline done (total %.1fs)", time.time() - t_start)


if __name__ == "__main__":
    asyncio.run(async_main())
