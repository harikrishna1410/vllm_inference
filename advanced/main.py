import concurrent.futures
import os
import time
import uuid
from typing import List

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.orchestrator import ClusterClient

from utils import parse_args, submit_prompt, wait_for_vllm


def create_prompt(nprompts) -> List[str]:
    # Send prompts
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


def main():
    args_dict = parse_args()

    # Setup the vllm servers 1 per node
    cpus = list(range(104))
    cpus.pop(52)
    cpus.pop(0)
    sys_config = SystemConfig(
        name="aurora", ncpus=102, ngpus=12, cpus=cpus, gpus=list(range(12))
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

    el.start()
    time.sleep(5.0)

    ##preprocessing
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
    vllm_start_futures = []
    with ClusterClient(checkpoint_dir=launcer_config.checkpoint_dir) as client:
        if not os.path.exists(copy_dir):
            # submit the copy task
            copy_futures = []
            for node in get_nodes():
                future = client.submit(f"mkdir -p {copy_dir}", nnodes=1, ppn=1)
                copy_futures.append(future)

            done, not_done = concurrent.futures.wait(copy_futures)

            copy_futures = []
            for node in get_nodes():
                task = Task(
                    task_id=str(uuid.uuid4()),
                    nnodes=1,
                    ppn=90,
                    executable=f"dsync {model_dir + '/'} {copy_dir + '/'}",
                    executor_name="async_mpi",
                )
                future = client.submit(task=task)
                copy_futures.append(future)

            done, not_done = concurrent.futures.wait(copy_futures)

            if all([fut.exception() is None for fut in copy_futures]):
                cache_dir = local_cache
            else:
                raise RuntimeError("Copying models failed")

        ##Start one vllm server per node
        for vllm_idx in range(len(get_nodes())):
            vllm_start_futures.append(
                client.submit(
                    f"{os.getcwd()}/start_vllm_server.sh {vllm_idx} {args_dict['port']} {args_dict['ngpus_per_model']} {args_dict['model']} {cache_dir} {args_dict['tmp_dir']}",
                    ngpus_per_process=args_dict["ngpus_per_model"],
                )
            )
        ##Wait for them to finish
        vllm_wait_futures = []
        for i in range(len(get_nodes())):
            vllm_wait_futures.append(client.submit(wait_for_vllm, args_dict))

        done, not_done = concurrent.futures.wait(vllm_wait_futures)
        exceptions = []
        for fut in vllm_wait_futures:
            exceptions.append(fut.exception())

        if all([e is None for e in exceptions]):
            print("Succefully launched the vllm server. Submitting prompts")
            prompts = create_prompt(args_dict["num_prompts"])
            prompt_futures = client.map(
                submit_prompt, zip(prompts, [args_dict] * args_dict["num_prompts"])
            )
            results = []
            for fut in concurrent.futures.as_completed(prompt_futures):
                result = fut.result()
                e = fut.exception()
                print(f"Got result: {result} and Exception: {e}")
                results.append(result)
            print("All prompt tasks done")

        # stop the vllm servers
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

        done, not_done = concurrent.futures.wait(stop_futures)
    # stop the cluster
    el.stop()


if __name__ == "__main__":
    main()
