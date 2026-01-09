# vLLM Weak Scaling Inference Test

This repository contains scripts for running weak scaling tests of vLLM inference on Aurora supercomputer using [Ensemble Launcher](https://github.com/argonne-lcf/ensemble_launcher)

## Overview

The weak scaling test runs multiple independent instances of the same model across different nodes, with each node handling its own vLLM server and client.

## Architecture

- **vLLM Server**: Runs on each node with configurable GPU count
- **Client**: Sends inference prompts to the local vLLM server
- **One-to-One Relation**: Each client talks to its dedicated server

## Project Structure

```
.
├── submit_infer.sh               # PBS job submission script
├── infer_no_ray.sh               # Main script to launch server + client
├── client.py                     # OpenAI client for sending prompts
├── create_ensemble_config.py    # Generate ensemble configuration
├── ensemble_config.json          # Generated ensemble configuration
├── launcher_config.json          # Ensemble launcher settings
├── system_config.json            # Aurora system configuration
```

## Prerequisites

1. **Environment**: Activate your vLLM environment
    ```bash
    python3 -m venv ~/.env
    source ~/.env/bin/activate
    ```

2. **Ensemble Launcher**: Clone and install Ensemble Launcher
    ```bash
    git clone https://github.com/argonne-lcf/ensemble_launcher.git
    pip install -e ./ensemble_launcher
    ```

3. **Hugging Face Token**: Set your token for model access
   ```bash
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   ```

4. **Model Cache**: Ensure models are cached in `.cache/` directory or specify a different location

## Quick Start

### 1. Generate Ensemble Configuration

```bash
# Basic usage: 1 node, 8 GPUs per instance, 10 prompts
python create_ensemble_config.py --nnodes 1 --ngpus 8 --num-prompts 10

# Scale to multiple nodes
python create_ensemble_config.py --nnodes 4 --ngpus 8 --num-prompts 100

# Use different model
python create_ensemble_config.py --nnodes 2 --model meta-llama/Llama-3.1-70B-Instruct --ngpus 4
```

**Arguments:**
- `--nnodes`: Number of nodes (ensemble size)
- `--ngpus`: GPUs per model instance
- `--model`: Model name from HuggingFace
- `--num-prompts`: Number of prompts each client sends

### 2. Submit PBS Job

Edit [submit_infer.sh](submit_infer.sh) to set PBS parameters:

```bash
#PBS -l select=2          # Number of nodes
#PBS -l walltime=0:30:00  # Wall time
#PBS -q debug             # Queue
#PBS -A datascience       # Account
```

Submit the job:
```bash
qsub submit_infer.sh
```

### 3. Monitor Logs

Logs are organized by GPU count and model:
```bash
ls logs_8gpus/meta-llama/Llama-3.1-8B-Instruct/

# View server logs
tail -f logs_8gpus/meta-llama/Llama-3.1-8B-Instruct/1.vllm.server.log

# View client logs
tail -f logs_8gpus/meta-llama/Llama-3.1-8B-Instruct/1.vllm.client.log
```

## Scripts

### create_ensemble_config.py

Generates the ensemble configuration for weak scaling tests. Creates three config files:
- `ensemble_config.json`: Task definitions with parameters
- `system_config.json`: Aurora system resources (CPUs, GPUs)
- `launcher_config.json`: Ensemble launcher settings

### infer_no_ray.sh

Main worker script that:
1. Sets up HuggingFace environment and cache directories
2. Configures Intel OneAPI and CCL settings
3. Downloads model if not cached
4. Launches vLLM server with appropriate GPU configuration
5. Runs client to send prompts
6. Cleans up server process

**Usage:**
```bash
./infer_no_ray.sh IDX PORT NGPUS [MODEL] [CACHE_DIR] [TEMPDIR] [NUM_PROMPTS]
```

**Example:**
```bash
./infer_no_ray.sh 1 8000 8 meta-llama/Llama-3.1-8B-Instruct .cache /tmp 10
```

### client.py

OpenAI-compatible client that:
- Waits for vLLM server to be ready (up to 1 hour timeout)
- Sends specified number of prompts
- Reports timing statistics and success/error counts

**Usage:**
```bash
python client.py --host hostname --model model_name --port 8000 --num-prompts 10
```

## Environment Variables

Key environment variables set by the scripts:

**HuggingFace:**
- `HF_HOME`, `HF_HUB_CACHE`: Cache directories
- `HF_TOKEN`: Authentication token
- `HF_HUB_OFFLINE=1`: Use cached models only

## Configuration Files

### ensemble_config.json

Defines the ensemble of tasks:
```json
{
  "ensembles": {
    "8gpus_meta-llama--Llama-3.1-8B-Instruct": {
      "nnodes": 1,
      "ppn": 1,
      "ngpus_per_process": 8,
      "relation": "one-to-one",
      "cmd_template": "./infer_no_ray.sh {idx} {port} {ngpus} {model} ...",
      "idx": [1, 2, ...],
      "port": [8000, 8001, ...],
      ...
    }
  }
}
```

### system_config.json

Aurora system configuration for EnsembleLauncher

### launcher_config.json

Ensemble launcher configuration using MPI for child processes and multiprocessing for tasks.

## Weak Scaling Test Workflow

1. **Generate Config**: Run `create_ensemble_config.py` with desired parameters
2. **Submit Job**: Use `qsub submit_infer.sh` to submit to PBS
3. **Ensemble Launch**: `el` command launches all tasks across nodes
4. **Per Node**: Each node runs `infer_no_ray.sh` which:
   - Starts vLLM server on unique port
   - Client waits for server ready
   - Client sends prompts
   - Logs timing and results
5. **Analysis**: Review logs for throughput and latency metrics

## Tips

**GPU Configuration:**
- Single GPU: Automatic configuration
- Multi-GPU: Uses `--distributed-executor-backend mp` with tensor parallelism

**Model Caching:**
- First run downloads models (requires proxy settings)
- Subsequent runs use cached models with `HF_HUB_OFFLINE=1`

**Debugging:**
- Check server logs for vLLM startup issues
- Check client logs for connection and inference errors
- Use `VLLM_LOGGING_LEVEL=DEBUG` for detailed output

**Resource Management:**
- Each instance uses dedicated GPUs via `ZE_AFFINITY_MASK`
- Unique ports prevent conflicts between instances
- Separate temp directories isolate each instance

## Example Output

Client log summary:
```
[2026-01-08 14:23:45] Summary:
[2026-01-08 14:23:45]   Total prompts: 100
[2026-01-08 14:23:45]   Successful: 100
[2026-01-08 14:23:45]   Failed: 0
[2026-01-08 14:23:45]   Total time: 245.32s
[2026-01-08 14:23:45]   Average time per prompt: 2.45s
```

## Troubleshooting

**vLLM server won't start:**
- Check GPU availability with `sycl-ls`
- Verify model is cached or proxy is set
- Check `ZE_AFFINITY_MASK` is correct

**Client connection timeout:**
- Verify server log shows "Application startup complete"
- Check port is not in use
- Ensure `no_proxy` is set correctly

**Out of memory:**
- Reduce `--ngpus` per instance
- Use smaller model
- Check GPU memory with `xpu-smi`

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Aurora Documentation](https://docs.alcf.anl.gov/aurora/)
- [Ensemble Launcher](https://github.com/argonne-lcf/ensemble_launcher)
