#!/bin/bash
# Usage: ./start_vllm_server_dsync.sh IDX PORT NGPUS [VLLM_MODEL] [CACHE_DIR] [TEMPDIR]
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
#
# Examples:
#   ./start_vllm_server_dsync.sh 1 8000 8                     # IDX=1, PORT=8000, 8 GPUs
#   ./start_vllm_server_dsync.sh 2 8001 4 meta-llama/Llama-3.1-70B-Instruct  # Custom model
#   ./start_vllm_server_dsync.sh 1 8000 8 meta-llama/Llama-3.1-8B-Instruct .cache ~/.conda_envs/vllm /tmp 10  # 10 prompts

set -o pipefail
export PYTHONNOUSERSITE=1
HOSTNAME=$(hostname)

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "SCRIPT_DIR: '$SCRIPT_DIR'"

# Check required arguments
if [ $# -lt 3 ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 IDX PORT NGPUS [VLLM_MODEL] [CACHE_DIR] [TEMPDIR] [BACKEND]"
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
BACKEND=${7:-1}

echo "IDX:$IDX"
echo "PORT:$PORT"
echo "NGPUS:$NGPUS"
echo "VLLM_MODEL:$VLLM_MODEL"
echo "CACHE_DIR:$CACHE_DIR"
echo "TEMPDIR:$TEMPDIR"
echo "ZE_AFFINITY_MASK: $ZE_AFFINITY_MASK"

if [ ! -d "${CACHE_DIR}" ]; then
    echo "$(date) TSB creating cache directory: ${CACHE_DIR}"
    mkdir -p ${CACHE_DIR}
else
    echo "$(date) TSB cache directory already exists: ${CACHE_DIR}"
fi

# HuggingFace environment variables
export HF_HOME="/tmp/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_TOKEN=${HUGGINGFACE_HUB_TOKEN}
export HF_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}
export HF_HUB_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1

# Temporary directory setup
export RAY_TMPDIR="/tmp"
MODEL_IDX_HASH=$(echo "${VLLM_MODEL}/${IDX}" | md5sum | cut -d' ' -f1 | cut -c1-8)
export TMPDIR="${TEMPDIR}/tmp-${MODEL_IDX_HASH}"
if [ ! -d "${TMPDIR}" ]; then
    echo "$(date) TSB creating TMPDIR: ${TMPDIR}"
    mkdir -p ${TMPDIR}
else
    echo "$(date) TSB TMPDIR already exists: ${TMPDIR}"
fi

# vLLM config
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export FI_MR_CACHE_MONITOR=userfaultfd
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=DEBUG
export OCL_ICD_SO="/opt/aurora/25.190.0/oneapi/2025.2/lib/libintelocl.so"
ray stop -f
export no_proxy="localhost,127.0.0.1" #Set no_proxy for the client to interact with the locally hosted model
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
export OCL_ICD_FILENAMES="/opt/aurora/25.190.0/oneapi/2025.2/lib/libintelocl.so"
export VLLM_DISABLE_SINKS=1

# Proxy settings (needed for model download)
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# Intel OneAPI and CCL settings
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset CCL_PROCESS_LAUNCHER
export CCL_PROCESS_LAUNCHER=None
unset ONEAPI_DEVICE_SELECTOR
export FI_MR_CACHE_MONITOR=userfaultfd

# vLLM specific settings
export VLLM_TRACE_FUNCTION_CALLS=1
export VLLM_SKIP_MODEL_REGISTRY=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Python and tokenizer settings
export TOKENIZERS_PARALLELISM=false

echo HF_HOME=${HF_HOME}
echo HF_HUB_CACHE=${HF_HUB_CACHE}
echo HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
echo HF_MODULES_CACHE=${HF_MODULES_CACHE}
echo TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}
echo HF_TOKEN=${HF_TOKEN}
echo HF_HUB_TOKEN=${HF_HUB_TOKEN}

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

if [ -z "$ZE_AFFINITY_MASK" ]; then
        echo "$(date) TSB ZE_AFFINITY_MASK not set. Building one based on NGPUS" > $LOG_DIR/${IDX}.vllm.server.log
        export ZE_AFFINITY_MASK=$(seq -s, 0 $(($NGPUS - 1)))
fi

if [ $NGPUS -eq 1 ]; then
    echo "$(date) TSB running vllm with 1 GPU"
    OCL_ICD_FILENAMES="/opt/aurora/25.190.0/oneapi/2025.2/lib/libintelocl.so" \
      VLLM_DISABLE_SINKS=1 \
      vllm serve ${VLLM_MODEL} \
      --port $PORT \
      --tensor-parallel-size 1 \
      --trust-remote-code \
      1> $LOG_DIR/${IDX}.vllm.server.log 2>$LOG_DIR/${IDX}.vllm.server.err
else
    echo "$(date) TSB running vllm with ${NGPUS} GPUs"
    OCL_ICD_FILENAMES="/opt/aurora/25.190.0/oneapi/2025.2/lib/libintelocl.so" \
      VLLM_DISABLE_SINKS=1 \
      vllm serve ${VLLM_MODEL} \
      --distributed-executor-backend mp \
      --port $PORT \
      --tensor-parallel-size ${NGPUS} \
      --trust-remote-code \
      1> $LOG_DIR/${IDX}.vllm.server.log 2>$LOG_DIR/${IDX}.vllm.server.err
fi
