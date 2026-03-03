import argparse
import sys
import time
from datetime import datetime

from openai import OpenAI


def print_with_timestamp(message):
    """Helper function to print messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


parser = argparse.ArgumentParser(
    description="vLLM client for sending inference prompts."
)
parser.add_argument(
    "--host", type=str, required=True, help="Hostname of the vLLM server"
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

args = parser.parse_args()

host = args.host
model = args.model
port = args.port
key = args.key

openai_api_base = f"http://{host}:{port}/v1"

client = OpenAI(
    api_key=key,
    base_url=openai_api_base,
)


def wait_for_vllm(timeout_seconds=300, check_interval=10):
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.0,
                max_tokens=1024,
                stream=False,
            )
            return True
        except Exception as e:
            print_with_timestamp(
                f"vLLM not ready yet. waiting time left = {timeout_seconds - (time.time() - start_time):.0f}s"
            )
        time.sleep(check_interval)
    return False


print_with_timestamp(f"Waiting for vLLM server at {host}:{port} to be ready...")
ready = wait_for_vllm(3600)
if not ready:
    print_with_timestamp(
        f"ERROR: vLLM server at {host}:{port} not ready after waiting."
    )
    sys.exit(1)
