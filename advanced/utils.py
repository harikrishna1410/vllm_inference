import argparse
import asyncio
import logging
import os
import random
import uuid
from glob import glob
from logging import Logger
from typing import List, TypedDict

import cloudpickle
from ensemble_launcher.ensemble.actor import Actor
from ensemble_launcher.inference.utils import build_model_cache


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
    parser = argparse.ArgumentParser(description="EL inference with vLLM")
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
    parser.add_argument(
        "--pre-build-vllm-cache",
        type=int,
        default=0,
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
    except Exception:
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

    from openai import OpenAI

    host = "0.0.0.0"
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
    except Exception:
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


class VLLMInference(Actor):
    def __init__(
        self,
        name: str,
        transport: str,
        model: str,
        cache_dir: str,
        tensor_parallel_size: int = 1,
    ):
        super().__init__(name, transport)
        self.model = model
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None
        self.logger = None

    def on_start(self):
        if self.logger is None:
            self.logger = get_logger(
                f"vllm-{random.randint(0, 1000)}", f"{os.getcwd()}/script_logs"
            )
        if self._llm is None:
            from vllm import LLM

            snapshots = glob(
                f"{self.cache_dir}/hub/models--{self.model.replace('/', '--')}/snapshots/*"
            )
            self.logger.info(f"model: {snapshots[0]}")
            self._llm = LLM(
                model=snapshots[0],
                tensor_parallel_size=self.tensor_parallel_size,
                trust_remote_code=True,
            )
            self.logger.info("init done!")

    def action(self, prompts="hello", temperature=0.0, max_tokens=1024):
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


class LocalActorWrapper:
    def __init__(self, executable, connection, loop=None):
        self._callable = executable
        self._connection = connection
        self._loop = loop
        self._stop = None

    async def _invoke(self, *args):
        result = self._callable()
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _action_loop(self):
        self._stop = asyncio.Event()
        await self._connection.open()
        while not self._stop.is_set():
            frames = await self._connection.recv()
            args = cloudpickle.loads(frames[-1])
            if args == "stop":
                self._stop.set()
            else:
                if isinstance(args, tuple):
                    result = await self._invoke()
                await self._connection.send(cloudpickle.dumps(result))

        return result

    def __call__(self, *args, **kwds):
        self._callable._ensure_initialized()
        return asyncio.run(self._action_loop())


class AsyncVLLMInference:
    def __init__(self, model: str, cache_dir: str, tensor_parallel_size: int = 1):
        self.model = model
        self.cache_dir = cache_dir
        self.tensor_parallel_size = tensor_parallel_size
        self._engine = None
        self.logger = None

    async def _ensure_initialized(self):
        if self._engine is None:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            engine_args = AsyncEngineArgs(
                model=self.model,
                tensor_parallel_size=self.tensor_parallel_size,
                download_dir=os.path.join(self.cache_dir, "hub"),
                trust_remote_code=True,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def __call__(self, prompts, temperature=0.0, max_tokens=1024):
        self.logger = get_logger("vllm", f"{os.getcwd()}/script_logs")
        self.logger.info("Starting init...")
        try:
            await self._ensure_initialized()
        except Exception as e:
            self.logger.info(f"init failed with error: {e}")
            return str(e)
        self.logger.info("Init done")
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

        if isinstance(prompts, str):
            prompts = [prompts]
            single = True
        else:
            single = False

        async def _generate_one(prompt):
            request_id = str(uuid.uuid4())
            final_output = None
            async for output in self._engine.generate(
                prompt, sampling_params, request_id
            ):
                final_output = output
            return final_output.outputs[0].text

        self.logger.info("generating....")
        results = await asyncio.gather(*[_generate_one(p) for p in prompts])

        return results[0] if single else list(results)


def main(fn, args):
    fn(*args)


def exec_main(llm, args):
    main(llm, args)


def test_asyncio_llm(model):
    async def _inner():
        from vllm import LLM

        snapshots = glob(
            f"/tmp/model_cache/hub/models--{model.replace('/', '--')}/snapshots/*"
        )
        llm = LLM(model=snapshots[0], tensor_parallel_size=1)
        result = llm.generate(["hello"])
        print(result)

    asyncio.run(_inner())


def call_llm(model: str, prompts: List):

    os.environ["VLLM_CACHE_ROOT"] = f"/tmp/vllm_cache_{uuid.uuid4().hex[:6]}"
    os.makedirs(os.environ["VLLM_CACHE_ROOT"])
    os.environ["MASTER_PORT"] = str(random.randint(10000, 99999))
    build_model_cache()

    from vllm import LLM, SamplingParams

    snapshots = glob(
        f"/tmp/model_cache/hub/models--{model.replace('/', '--')}/snapshots/*"
    )

    llm = LLM(
        model=snapshots[0],
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    outputs = llm.generate(prompts, sampling_params)

    return outputs


if __name__ == "__main__":
    # args_dict = parse_args()
    # os.environ["ZE_AFFINITY_MASK"] = "0"
    # result = asyncio.run(
    #     AsyncVLLMInference(
    #         model=args_dict["model"],
    #         cache_dir="/tmp/81968554-d3fa-4b03-8ff9-1ae1a50d9aac",
    #     )("hello")
    # )

    # print(result)

    # from concurrent.futures import ProcessPoolExecutor

    # infer = VLLMInference(args_dict["model"], cache_dir="/tmp/model_cache")

    # # asyncio.run(main(infer, ("hello",)))

    # # infer("hello")

    # with ProcessPoolExecutor() as exec:
    #     future = exec.submit(exec_main, infer, ("hello",))

    # print(future.result())

    ##
    # test_asyncio_llm(args_dict["model"])

    # from vllm.platforms.xpu import XPUPlatform
    # import torch
    from ensemble_launcher import executors

    print(dir(executors))

    # print(zmq.__file__)
    # print(zmq.zmq_version())

    # snapshots = glob(
    #     f"/tmp/model_cache/hub/models--{args_dict['model'].replace('/', '--')}/snapshots/*"
    # )
    # llm = LLM(model=snapshots[0], tensor_parallel_size=1)
    # print("success")
