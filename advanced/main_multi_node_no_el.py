import os
import random
import subprocess
import time
import uuid
from typing import List

from utils import get_logger, parse_args, submit_prompt, wait_for_vllm

logger = get_logger("main_multi_node_no_el", log_dir=f"{os.getcwd()}/script_logs")

UTILS_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils.py")


def create_prompt(nprompts) -> List[str]:
    prompt = "Hi, can you introduce yourself?"
    return [prompt for i in range(nprompts)]


def get_nodes():
    nodefile = os.environ.get("PBS_NODEFILE")
    if not nodefile:
        raise RuntimeError("PBS_NODEFILE not set")
    with open(nodefile) as f:
        return [line.strip() for line in f if line.strip()]


def mpi(nnodes, cmd):
    return ["mpirun", "-np", str(nnodes), "--ppn", "1"] + cmd


def main():
    t_start = time.time()
    logger.info("main started")

    args_dict = parse_args()
    nodes = get_nodes()
    nnodes = len(nodes)
    logger.info("running on %d nodes: %s", nnodes, nodes)

    model_dir = os.path.join(
        args_dict["cache_dir"],
        "hub",
        f"models--{args_dict['model'].replace('/', '--')}",
    )
    local_cache = os.path.join("/tmp", str(uuid.uuid4()))
    copy_dir = os.path.join(
        local_cache,
        "hub",
        f"models--{args_dict['model'].replace('/', '--')}",
    )
    cache_dir = local_cache

    # Copy model to local cache on all nodes
    if not os.path.exists(copy_dir):
        t0 = time.time()
        logger.info("mkdir on %d nodes: %s", nnodes, copy_dir)
        subprocess.run(mpi(nnodes, ["mkdir", "-p", copy_dir]), check=True)

        logger.info("dsync started on %d nodes: %s -> %s", nnodes, model_dir, copy_dir)
        copy_procs = []
        for node in nodes:
            proc = subprocess.Popen(
                [
                    "mpirun",
                    "--host",
                    node,
                    "-np",
                    "90",
                    "--ppn",
                    "90",
                    "dsync",
                    model_dir + "/",
                    copy_dir + "/",
                ]
            )
            copy_procs.append((node, proc))

        failed = []
        for node, proc in copy_procs:
            rc = proc.wait()
            if rc != 0:
                failed.append(node)
                logger.error("dsync failed on node %s (rc=%d)", node, rc)
            else:
                logger.info("dsync done on node %s", node)

        if failed:
            raise RuntimeError(f"dsync failed on nodes: {failed}")
        logger.info("model copy done on all nodes (%.1fs)", time.time() - t0)

    # Start NGPUS engine processes per node, one per GPU
    ngpus = args_dict["ngpus_per_model"]
    master_addr = nodes[0]
    master_port = random.randint(20000, 30000)
    t0 = time.time()
    logger.info(
        "starting vllm engines on %d nodes (%d procs/node, port=%s, master=%s:%d)",
        nnodes,
        ngpus,
        args_dict["port"],
        master_addr,
        master_port,
    )

    vllm_procs = []
    for idx, node in enumerate(nodes):
        env_prefix = (
            f"MASTER_ADDR={master_addr} "
            f"MASTER_PORT={master_port} "
            f"WORLD_SIZE={nnodes * ngpus} "
            f"RANK_OFFSET={idx * ngpus} "
            f"NNODES={nnodes}"
        )
        cmd = (
            f"{env_prefix} "
            f"{os.getcwd()}/start_vllm_engine.sh "
            f"{idx} {args_dict['port']} {ngpus} "
            f"{args_dict['model']} {cache_dir} {args_dict['tmp_dir']}"
        )
        proc = subprocess.Popen(
            [
                "mpirun",
                "--host",
                node,
                "-np",
                str(ngpus),
                "--ppn",
                str(ngpus),
                "bash",
                "-c",
                cmd,
            ]
        )
        logger.info("vllm engines launched on node %s (pid=%d)", node, proc.pid)
        vllm_procs.append((node, proc))

    try:
        # Wait for vllm to be ready on all nodes
        subprocess.run(
            [
                "mpirun",
                "-np",
                f"{ngpus}",
                "--ppn",
                f"{ngpus}",
                "python",
                UTILS_PY,
                "--mode",
                "wait",
                "--port",
                args_dict["port"],
                "--model",
                args_dict["model"],
                "--key",
                args_dict["key"],
            ],
            check=True,
        )
        logger.info("all vllm engines ready (%.1fs since launch)", time.time() - t0)

        prompts = create_prompt(args_dict["num_prompts"])
        logger.info("submitting %d prompts", len(prompts))
        t_prompts = time.time()

        for i, prompt in enumerate(prompts):
            t_p = time.time()
            try:
                result = submit_prompt(prompt, args_dict)
                logger.info("prompt %d done (%.2fs): %s", i, time.time() - t_p, result)
            except Exception as e:
                logger.error("prompt %d failed (%.2fs): %s", i, time.time() - t_p, e)

        logger.info("all prompts done (%.1fs)", time.time() - t_prompts)
    finally:
        t0 = time.time()
        logger.info("stopping vllm engines")
        subprocess.run(mpi(nnodes, [f"{os.getcwd()}/stop_vllm_server.sh"]), check=False)
        for node, proc in vllm_procs:
            proc.wait()
            logger.info("vllm engines stopped on node %s", node)
        logger.info("all vllm engines stopped (%.1fs)", time.time() - t0)

    logger.info("main done (total %.1fs)", time.time() - t_start)


if __name__ == "__main__":
    main()
