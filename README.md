# vLLM Weak Scaling Inference Test

This repository contains scripts for running weak scaling tests of vLLM inference on Aurora supercomputer using [Ensemble Launcher](https://github.com/argonne-lcf/ensemble_launcher).

## Overview

The weak scaling test runs multiple independent instances of the same model across different nodes, with each node handling its own vLLM server and client.

## Architecture

- **vLLM Server**: Runs on each node with configurable GPU count
- **Client**: Sends inference prompts to the local vLLM server
- **One-to-One Relation**: Each client talks to its dedicated server

## Project Structure

```
.
├── simple/                           # Simple mode: config-driven via el CLI
│   ├── submit_infer.sh               # PBS job submission script
│   ├── infer_no_ray.sh               # Worker script: launches server + client per node
│   ├── client.py                     # OpenAI client for sending prompts
│   └── create_ensemble_config.py     # Generates ensemble/system/launcher config JSONs
│
└── advanced/                         # Advanced mode: programmatic Python API
    ├── submit_infer.sh               # PBS job submission script
    ├── main.py                       # Orchestrator: model copy, server start, inference
    ├── start_vllm_server.sh          # Starts vLLM server on a node
    ├── stop_vllm_server.sh           # Stops vLLM server on a node
    ├── wait_for_vllm.py              # Polls until vLLM server is ready
    ├── download_model_hf.py          # Helper to download models from HuggingFace
    └── utils.py                      # Arg parsing, prompt submission, wait logic
```

## Prerequisites

1. **Environment**: Activate your vLLM environment
    ```bash
    source ~/.vllm_envs/inference/bin/activate
    ```

2. **Ensemble Launcher** (`pull_workers` branch): Clone and install
    ```bash
    git clone https://github.com/argonne-lcf/ensemble_launcher.git
    cd ensemble_launcher
    git checkout pull_workers
    pip install -e .
    cd ..
    ```

3. **Hugging Face Token**: Set your token for model access
   ```bash
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   ```

4. **Model Cache**: Ensure models are cached or use `download_model_hf.py` to fetch them

## Modes

### Simple Mode (`simple/`)

Config-driven approach using the `el` CLI. Each node runs `infer_no_ray.sh` which starts the vLLM server, waits for it to be ready, sends prompts via `client.py`, and cleans up.

#### 1. Generate Ensemble Configuration

```bash
cd simple/

# Basic usage: 1 node, 8 GPUs per instance, 10 prompts
python create_ensemble_config.py --nnodes 1 --ngpus 8 --num-prompts 10

# Scale to multiple nodes
python create_ensemble_config.py --nnodes 4 --ngpus 8 --num-prompts 100

# Use a different model
python create_ensemble_config.py --nnodes 2 --model meta-llama/Llama-3.1-70B-Instruct --ngpus 4
```

**Arguments:**
- `--nnodes`: Number of nodes (ensemble size)
- `--ngpus`: GPUs per model instance
- `--model`: Model name from HuggingFace
- `--num-prompts`: Number of prompts each client sends

#### 2. Submit PBS Job

Edit [simple/submit_infer.sh](simple/submit_infer.sh) to set PBS parameters, then submit:

```bash
qsub simple/submit_infer.sh
```

The job runs `el ensemble_config.json ...` which distributes `infer_no_ray.sh` across nodes.

---

### Advanced Mode (`advanced/`)

Programmatic approach using the Ensemble Launcher Python API directly. `main.py` acts as the orchestrator: it starts a live `EnsembleLauncher` cluster, copies the model to node-local `/tmp` using `dsync` for faster I/O, starts one vLLM server per node, waits for readiness, then fans out prompts across the cluster using `ClusterClient.map()`, and finally shuts everything down.

Key differences from simple mode:
- **Dynamic task submission** via `ClusterClient` — no pre-generated JSON configs
- **Model staging**: copies model from shared filesystem to node-local `/tmp` using `dsync` before server start, reducing load on the parallel filesystem
- **Parallel prompt dispatch**: uses `client.map()` to submit all prompts concurrently across nodes
- **Lifecycle management**: programmatic start/stop of servers within a single job

#### 1. Edit Job Parameters (optional)

The `main.py` reads arguments from the command line. Defaults are set in `utils.py`:
- `--model`: `meta-llama/Llama-3.1-8B-Instruct`
- `--port`: `8000`
- `--num-prompts`: `10`
- `--ngpus-per-model`: `8`
- `--cache-dir`: `/lus/flare/projects/datascience/hari/vllm_inference/.cache`
- `--tmp-dir`: `/tmp`

#### 2. Submit PBS Job

Edit [advanced/submit_infer.sh](advanced/submit_infer.sh) to set PBS parameters (`select`, `walltime`, `queue`, `account`), then submit:

```bash
cd advanced/
qsub submit_infer.sh
```

The job runs `python3 main.py [args]` directly — no config generation step needed.

#### 3. How `main.py` Works

1. Starts `EnsembleLauncher` with `async_mpi` children and `async_processpool`/`async_mpi` task executors
2. For each node: creates the local cache directory and runs `dsync` to copy the model from shared storage to `/tmp`
3. Starts one vLLM server per node via `start_vllm_server.sh`
4. Waits for all servers to respond (up to 1 hour)
5. Dispatches all prompts in parallel across the cluster via `client.map(submit_prompt, ...)`
6. Stops all vLLM servers via `stop_vllm_server.sh`
7. Stops the `EnsembleLauncher` cluster

---

## Monitor Logs

Simple mode logs are organized by GPU count and model:
```bash
ls simple/logs_8gpus/meta-llama/Llama-3.1-8B-Instruct/

# View server logs
tail -f simple/logs_8gpus/meta-llama/Llama-3.1-8B-Instruct/1.vllm.server.log

# View client logs
tail -f simple/logs_8gpus/meta-llama/Llama-3.1-8B-Instruct/1.vllm.client.log
```

Advanced mode logs are written per hostname under `<hostname>/logs_<N>gpus/<model>/`.

Ensemble Launcher logs are written to `logs/` in the current directory.

## Environment Variables

Key environment variables set by the scripts:

**HuggingFace:**
- `HF_HOME`, `HF_HUB_CACHE`: Cache directories
- `HF_TOKEN`: Authentication token
- `HF_HUB_OFFLINE=1`: Use cached models only (set after model is staged)

**Intel/vLLM (Aurora-specific):**
- `ZE_FLAT_DEVICE_HIERARCHY=FLAT`
- `CCL_PROCESS_LAUNCHER=None`
- `OCL_ICD_FILENAMES=libintelocl.so`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
