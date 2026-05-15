#!/bin/bash
# Usage: ./infer_no_ray.sh IDX PORT NGPUS [VLLM_MODEL] [CACHE_DIR] [TEMPDIR] [NNODES]
#
# Required Arguments:
#   IDX          - Job index/identifier
#   PORT         - vLLM server port
#   NGPUS        - Number of GPUs to use
#
# Optional Arguments:
#   VLLM_MODEL   - Model name (default: meta-llama/Llama-3.1-8B-Instruct)
#   CACHE_DIR    - HuggingFace cache directory (default: $(pwd)/.cache)
#   TEMPDIR      - Temporary directory (default: /tmp)
#   NNODES  - Number of prompts to send (default: 1)
#
# Examples:
#   ./infer_no_ray.sh 1 8000 8                     # IDX=1, PORT=8000, 8 GPUs
#   ./infer_no_ray.sh 2 8001 4 meta-llama/Llama-3.1-70B-Instruct  # Custom model
#   ./infer_no_ray.sh 1 8000 8 meta-llama/Llama-3.1-8B-Instruct .cache ~/.conda_envs/vllm /tmp 10  # 10 prompts

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "SCRIPT_DIR: '$SCRIPT_DIR'"

# Check required arguments
if [ $# -lt 3 ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 IDX PORT NGPUS [VLLM_MODEL] [CACHE_DIR] [TEMPDIR] [NNODES]"
    echo ""
    echo "Required arguments:"
    echo "  IDX          - Job index/identifier"
    echo "  PORT         - vLLM server port"
    echo "  NGPUS        - Number of GPUs to use"
    echo ""
    echo "Example: $0 1 8000 8"
    exit 1
fi

# Command line arguments
IDX=${1}
PORT=${2}
NGPUS=${3}
VLLM_MODEL=${4:-"meta-llama/Llama-3.1-8B-Instruct"}
CACHE_DIR=${5:-"$(pwd)/.cache"}
TEMPDIR=${6:-"/tmp"}
NNODES=${7:-1}

echo "IDX:$IDX"
echo "PORT:$PORT"
echo "NGPUS:$NGPUS"
echo "VLLM_MODEL:$VLLM_MODEL"
echo "CACHE_DIR:$CACHE_DIR"
echo "TEMPDIR:$TEMPDIR"
echo "NNODES:$NNODES"
echo "ZE_AFFINITY_MASK: $ZE_AFFINITY_MASK"

if [ ! -d "${CACHE_DIR}" ]; then
    echo "$(date) TSB creating cache directory: ${CACHE_DIR}"
    mkdir -p ${CACHE_DIR}
else
    echo "$(date) TSB cache directory already exists: ${CACHE_DIR}"
fi

# HuggingFace environment variables
export HF_HOME=${CACHE_DIR}
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN=${HUGGINGFACE_HUB_TOKEN}
export HF_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Temporary directory setup
MODEL_IDX_HASH=$(echo "${VLLM_MODEL}/${IDX}" | md5sum | cut -d' ' -f1 | cut -c1-8)
export TMPDIR="${TEMPDIR}/tmp-${MODEL_IDX_HASH}"

if [ ! -d "${TMPDIR}" ]; then
    echo "$(date) TSB creating TMPDIR: ${TMPDIR}"
    mkdir -p ${TMPDIR}
else
    echo "$(date) TSB TMPDIR already exists: ${TMPDIR}"
fi

# Proxy settings (needed for model download)
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# Intel OneAPI and CCL settings
unset CCL_PROCESS_LAUNCHER
export CCL_PROCESS_LAUNCHER=None
unset ONEAPI_DEVICE_SELECTOR
export OCL_ICD_FILENAMES="libintelocl.so"
export FI_MR_CACHE_MONITOR=userfaultfd

# vLLM specific settings
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOG_LEVEL=DEBUG
export VLLM_TRACE_FUNCTION_CALLS=1
export VLLM_SKIP_MODEL_REGISTRY=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Python and tokenizer settings
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo HF_HOME=${HF_HOME}
echo HF_HUB_CACHE=${HF_HUB_CACHE}
echo HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
echo HF_MODULES_CACHE=${HF_MODULES_CACHE}
echo TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
echo HF_TOKEN=${HF_TOKEN}
echo HF_HUB_TOKEN=${HF_HUB_TOKEN}

echo "Testing model access..."
python -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download('${VLLM_MODEL}', local_files_only=True, cache_dir='${HF_HUB_CACHE}')
    print('Model found locally')
except:
    print('Model not found locally, will download...')
    snapshot_download('${VLLM_MODEL}', cache_dir='${HF_HUB_CACHE}')
    print('Model downloaded successfully')
"

### set up logging directory
LOG_DIR="${SCRIPT_DIR}/$(hostname)/logs_${NGPUS}gpus/$VLLM_MODEL"

if [ ! -d $LOG_DIR ]; then
    echo "$(date) TSB creating directory for model logs: $LOG_DIR"
    mkdir -p $LOG_DIR
else
    echo "$(date) TSB model log directory already exists: $LOG_DIR"
fi

HOSTNAME=$(hostname)

echo "$(date) TSB script directory is: $SCRIPT_DIR"
echo "$(date) TSB hostname: $HOSTNAME"

export LOCAL_RANK=0
export RANK=$(( RANK_OFFSET + PALS_LOCAL_RANKID ))

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=${PALS_LOCAL_RANKID}

echo "$(date) TSB RANK:$RANK LOCAL_RANK:$LOCAL_RANK WORLD_SIZE:$WORLD_SIZE MASTER_ADDR:$MASTER_ADDR MASTER_PORT:$MASTER_PORT ZE_AFFINITY_MASK:$ZE_AFFINITY_MASK"
echo "$(date) TSB running vllm engine rank=${RANK}/${WORLD_SIZE} with ${NGPUS} GPUs"



vllm serve ${VLLM_MODEL} --distributed-executor-backend external_launcher --port $(( PORT + RANK )) --tensor-parallel-size ${NGPUS} --pipeline-parallel-size ${NNODES} --trust-remote-code 1> $LOG_DIR/${IDX}_rank${RANK}.vllm.engine.log 2>$LOG_DIR/${IDX}_rank${RANK}.vllm.engine.err
