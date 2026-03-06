import os
import socket
import subprocess
import time
import uuid
from typing import List

from utils import get_logger, parse_args, submit_prompt, wait_for_vllm

logger = get_logger("main_no_el", log_dir=f"{os.getcwd()}/script_logs")

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


def launch_vllm_ssh(nodes, cmd_str):
    """Launch vllm server on each node via SSH. Returns list of (node, Popen) tuples."""
    procs = []
    for node in nodes:
        proc = subprocess.Popen(["ssh", node, "bash", "-c", cmd_str])
        procs.append((node, proc))
    return procs


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

    # Start one vllm server per node in the background
    t0 = time.time()
    launch = args_dict["launch"]
    logger.info(
        "starting vllm servers on %d nodes via %s (port=%s, ngpus=%d)",
        nnodes,
        launch,
        args_dict["port"],
        args_dict["ngpus_per_model"],
    )

    vllm_cmd = (
        f"{os.getcwd()}/start_vllm_server.sh 0 {args_dict['port']} "
        f"{args_dict['ngpus_per_model']} {args_dict['model']} "
        f"{cache_dir} {args_dict['tmp_dir']}"
    )

    if launch == "ssh":
        vllm_procs = []
        for node in nodes:
            if node == socket.gethostname():
                proc = subprocess.Popen(
                    ["bash", "-c", f"cd {os.getcwd()} && {vllm_cmd}"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                proc = subprocess.Popen(
                    ["ssh", node, "bash", "-c", f"cd {os.getcwd()} && {vllm_cmd}"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            vllm_procs.append((node, proc))
            logger.info("vllm server launched on %s via SSH (pid=%d)", node, proc.pid)
    else:
        vllm_proc = subprocess.Popen(
            [
                "mpirun",
                "-np",
                f"{nnodes}",
                "--ppn",
                "1",
                "--cpu-bind",
                "list:"
                + ":".join(map(str, range(1, args_dict["ngpus_per_model"] + 1))),
                "bash",
                "-c",
                vllm_cmd,
            ],
        )
        logger.info("vllm server processes launched via mpirun (pid=%d)", vllm_proc.pid)

    try:
        # Wait for vllm to be ready on all nodes
        wait_cmd = [
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
        ]
        if launch == "ssh":
            wait_procs = []
            for node in nodes:
                if node == socket.gethostname():
                    proc = subprocess.Popen(
                        wait_cmd,
                        stdin=subprocess.DEVNULL,
                    )
                else:
                    proc = subprocess.Popen(
                        ["ssh", node] + wait_cmd,
                        stdin=subprocess.DEVNULL,
                    )
                wait_procs.append((node, proc))
            for node, proc in wait_procs:
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"wait_for_vllm failed on node {node} (rc={rc})")
        else:
            subprocess.run(
                mpi(nnodes, wait_cmd),
                check=True,
                stdin=subprocess.DEVNULL,
            )
        logger.info("all vllm servers ready (%.1fs since launch)", time.time() - t0)

        prompts = create_prompt(args_dict["num_prompts"])
        logger.info("submitting %d prompts to each of %d servers", len(prompts), nnodes)
        t_prompts = time.time()

        for node in nodes:
            logger.info("submitting %d prompts to server on %s", len(prompts), node)
            t_node = time.time()
            for i, prompt in enumerate(prompts):
                t_p = time.time()
                try:
                    result = submit_prompt(prompt, args_dict, host=node)
                    logger.info(
                        "prompt %d done (%.2fs): %s", i, time.time() - t_p, result
                    )
                except Exception as e:
                    logger.error(
                        "prompt %d failed (%.2fs): %s", i, time.time() - t_p, e
                    )
            logger.info("server %s done (%.1fs)", node, time.time() - t_node)

        logger.info("all prompts done (%.1fs)", time.time() - t_prompts)
    finally:
        t0 = time.time()
        logger.info("stopping vllm servers")
        stop_cmd = f"{os.getcwd()}/stop_vllm_server.sh"
        if launch == "ssh":
            procs = []
            for node in nodes:
                p = subprocess.Popen(["ssh", node, "bash", "-c", stop_cmd])
                procs.append(p)
        else:
            subprocess.run(mpi(nnodes, stop_cmd), check=False)

        if launch == "ssh":
            for node, proc in vllm_procs:
                proc.wait()
                logger.info("vllm server on %s exited", node)
        else:
            vllm_proc.wait()
        logger.info("vllm servers stopped (%.1fs)", time.time() - t0)

    logger.info("main done (total %.1fs)", time.time() - t_start)


def main_single_node():
    t_start = time.time()
    logger.info("main_single_node started")

    args_dict = parse_args()

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

    # Copy model to local cache
    if not os.path.exists(copy_dir):
        t0 = time.time()
        logger.info("model copy started: %s -> %s", model_dir, copy_dir)
        os.makedirs(copy_dir, exist_ok=True)
        subprocess.run(["dsync", model_dir + "/", copy_dir + "/"], check=True)
        logger.info("model copy done (%.1fs)", time.time() - t0)

    # Start vllm server in the background
    t0 = time.time()
    logger.info(
        "starting vllm server (port=%s, ngpus=%d)",
        args_dict["port"],
        args_dict["ngpus_per_model"],
    )
    vllm_proc = subprocess.Popen(
        [
            f"{os.getcwd()}/start_vllm_server.sh",
            "0",
            args_dict["port"],
            str(args_dict["ngpus_per_model"]),
            args_dict["model"],
            cache_dir,
            args_dict["tmp_dir"],
        ]
    )
    logger.info("vllm server process launched (pid=%d)", vllm_proc.pid)

    try:
        wait_for_vllm(args_dict)
        logger.info("vllm server ready (%.1fs since launch)", time.time() - t0)

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
        logger.info("stopping vllm server")
        subprocess.run([f"{os.getcwd()}/stop_vllm_server.sh"], check=False)
        vllm_proc.wait()
        logger.info("vllm server stopped (%.1fs)", time.time() - t0)

    logger.info("main_single_node done (total %.1fs)", time.time() - t_start)


if __name__ == "__main__":
    if len(get_nodes()) == 1:
        main_single_node()
    else:
        main()
