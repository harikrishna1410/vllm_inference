import sys
import argparse
import time
from openai import OpenAI
from datetime import datetime

def print_with_timestamp(message):
    """Helper function to print messages with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

parser = argparse.ArgumentParser(description='vLLM client for sending inference prompts.')
parser.add_argument('--host', type=str, required=True, help='Hostname of the vLLM server')
parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Model name to use')
parser.add_argument('--port', type=str, default='8000', help='Port number for the vLLM server (default: 8000)')
parser.add_argument('--key', type=str, default='EMPTY', help='API key for authentication (default: EMPTY)')
parser.add_argument('--num-prompts', type=int, default=1, help='Number of prompts to send (default: 1)')

args = parser.parse_args()

host = args.host
model = args.model
port = args.port
key = args.key
num_prompts = args.num_prompts

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
                stream=False
            )
            return True
        except Exception as e:
            print_with_timestamp(f"vLLM not ready yet. waiting time left = {timeout_seconds - (time.time() - start_time):.0f}s")
        time.sleep(check_interval)
    return False

print_with_timestamp(f"Waiting for vLLM server at {host}:{port} to be ready...")
ready = wait_for_vllm(3600)
if not ready:
    print_with_timestamp(f"ERROR: vLLM server at {host}:{port} not ready after waiting.")
    sys.exit(1)

# Send prompts
prompt = "Hi, can you introduce yourself?"
print_with_timestamp(f"Sending {num_prompts} prompt(s) to model {model}...")

start_time = time.time()
success_count = 0
error_count = 0

for i in range(num_prompts):
    try:
        print_with_timestamp(f"Sending prompt {i+1}/{num_prompts}...")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
            stream=False
        )
        success_count += 1
        if i == 0:  # Print first response
            print_with_timestamp("First response received:")
            print(f"\n{response.choices[0].message.content}\n")
    except Exception as e:
        error_count += 1
        print_with_timestamp(f"Error on prompt {i+1}: {e}")

end_time = time.time()
total_time = end_time - start_time

print_with_timestamp(f"\n{'='*80}")
print_with_timestamp(f"Summary:")
print_with_timestamp(f"  Total prompts: {num_prompts}")
print_with_timestamp(f"  Successful: {success_count}")
print_with_timestamp(f"  Failed: {error_count}")
print_with_timestamp(f"  Total time: {total_time:.2f}s")
print_with_timestamp(f"  Average time per prompt: {total_time/num_prompts:.2f}s")
print_with_timestamp(f"{'='*80}")
print_with_timestamp("Done!")
