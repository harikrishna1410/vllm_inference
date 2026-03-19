import argparse
import logging
import os
import sys
from logging import Logger
from typing import List, TypedDict


def get_logger(name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger


class Args(TypedDict):
    model: str
    host: str
    port: str
    key: str
    num_prompts: int
    cache_dir: str
    tmp_dir: str
    ngpus_per_model: int
    mode: str
    launch: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="EL inference with vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8000",
        help="Port number for the vLLM server (default: 8000)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="EMPTY",
        help="API key for authentication (default: EMPTY)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts to send (default: 1)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/lus/flare/projects/datascience/hari/vllm_inference/.cache",
        help="Model cache dir",
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="/tmp",
        help="tmp dir",
    )
    parser.add_argument(
        "--ngpus-per-model",
        type=int,
        default=1,
        help="Number of GPUs per model, equal to the tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="wait",
        choices=["wait", "submit"],
        help="decide the mode to launch ",
    )
    parser.add_argument(
        "--launch",
        type=str,
        default="mpi",
        choices=["mpi", "ssh"],
        help="method to launch vllm servers on multi-node (default: mpi)",
    )
    parser.add_argument(
        "--num-gpus_per_node",
        type=int,
        default=12,
        help="Number of GPUs per node (default: 12 as on Aurora)",
    )
    parser.add_argument(
        "--num-cpus_per_node",
        type=int,
        default=104,
        help="Number of CPUs per node (default: 104 as on Aurora)",
    )

    args = parser.parse_args()
    args_dict = Args(**(vars(args)))
    return args_dict


def submit_prompt(prompt: str, args_dict: Args, host: str = None):
    import os
    import socket

    from openai import OpenAI

    if host is None:
        host = socket.gethostname()
    openai_api_base = f"http://{host}:{args_dict['port']}/v1"

    try:
        del os.environ["http_proxy"]
        del os.environ["https_proxy"]
        del os.environ["HTTP_PROXY"]
        del os.environ["HTTPS_PROXY"]
        os.environ["no_proxy"] = "localhost,127.0.0.1"
    except Exception as e:
        pass

    client = OpenAI(
        api_key=args_dict["key"],
        base_url=openai_api_base,
    )

    response = client.chat.completions.create(
        model=args_dict["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
        stream=False,
    )
    return response.choices[0].message.content


def submit_prompt_to_all(prompt: str, args_dict: Args, logger: Logger = None):
    import os
    import socket

    from openai import OpenAI

    host = socket.gethostname()
    local_rank = os.environ.get("PALS_LOCAL_RANKID", 0)
    openai_api_base = f"http://{host}:{int(args_dict['port']) + int(local_rank)}/v1"

    if logger is not None:
        logger.info(f"connecting to {openai_api_base}")

    try:
        del os.environ["http_proxy"]
        del os.environ["https_proxy"]
        del os.environ["HTTP_PROXY"]
        del os.environ["HTTPS_PROXY"]
        os.environ["no_proxy"] = "localhost,127.0.0.1"
    except Exception as e:
        pass

    with OpenAI(
        api_key=args_dict["key"],
        base_url=openai_api_base,
    ) as client:
        response = client.chat.completions.create(
            model=args_dict["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
            stream=False,
        )
    return response.choices[0].message.content


def wait_for_vllm(args_dict: Args, timeout_seconds=3600, check_interval=10):
    import os
    import socket
    import time

    start_time = time.time()
    logger = get_logger(
        __name__ + f"_{socket.gethostname()}", log_dir=f"{os.getcwd()}/logs"
    )
    while time.time() - start_time < timeout_seconds:
        try:
            # response = submit_prompt("Hi", args_dict)
            response = submit_prompt_to_all("Hi", args_dict, logger=logger)
            logger.info(f"wait_for_vllm: Got response {response}")
            return response
        except Exception as e:
            logger.info(f"wait_for_vllm: Getting response failed with error {e}")
            pass
        time.sleep(check_interval)
    raise RuntimeError(f"vLLM not ready yet after {timeout_seconds}")


if __name__ == "__main__":
    args_dict = parse_args()
    if args_dict["mode"] == "wait":
        wait_for_vllm(args_dict)
    elif args_dict["mode"] == "submit":
        submit_prompt("hi", args_dict)
    else:
        raise RuntimeError(f"Unknown mode {args_dict['mode']}")
