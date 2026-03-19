#!/bin/bash -l
##PBS -N el-vllm
##PBS -l select=2
##PBS -l walltime=01:00:00
##PBS -q debug-scaling
##PBS -A datascience
##PBS -l filesystems=home:flare
##PBS -j oe
#cd $PBS_O_WORKDIR

# Load frameworks module
module load frameworks
module list

BASE_DIR=$PWD
DSYNC=/home/balin/Useful/pbs_utils/stage_dsync

# Move env to /tmp on the nodes and source
VENV_FLARE_PATH=/flare/datascience/balin/vllm/el_Mar26/_venv
VENV_TMP_PATH=/tmp/hf_home/hub/_venv
$DSYNC -r 8 \
  $VENV_FLARE_PATH \
  $VENV_TMP_PATH
source $VENV_TMP_PATH/bin/activate

# Fix LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=\
#$VENV_TMP_PATH/lib/python3.12/site-packages/intel_extension_for_pytorch/lib:\
#$VENV_TMP_PATH/lib:\
#$VENV_TMP_PATH/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}:\
#/usr/lib64

# Move model weights to /tmp on the nodes
MODEL_FLARE_PATH=/flare/datascience/balin/vllm/dragon_Mar26/.cache
MODEL_TMP_PATH=/tmp/hf_home/hub/.cache/
$DSYNC -r 8 \
  $MODEL_FLARE_PATH \
  $MODEL_TMP_PATH
MODEL_DIR=$MODEL_TMP_PATH/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659

# Pre-build vLLM model-info caches
export VLLM_CACHE_ROOT=$BASE_DIR/.vllm_cache
if [ ! -d "${VLLM_CACHE_ROOT}" ]; then
    mkdir -p ${VLLM_CACHE_ROOT}
fi
echo "Building vLLM model-info caches in ${VLLM_CACHE_ROOT} ..."
python ./vllm_build_model_cache.py
echo "Cache build complete."

# Move model-info cache to /tmp on the nodes
MODELINFO_FLARE_PATH=$VLLM_CACHE_ROOT
MODELINFO_TMP_PATH=/tmp/hf_home/hub/.vllm_cache
$DSYNC -r 8 \
  $MODELINFO_FLARE_PATH \
  $MODELINFO_TMP_PATH
export VLLM_CACHE_ROOT=$MODELINFO_TMP_PATH

# Move prompts to /tmp on the nodes
PROMPTS_FLARE_PATH=$BASE_DIR/../data/prompts.jsonl
PROMPTS_TMP_PATH=/tmp/hf_home/prompts.jsonl
$DSYNC -r 8 \
  $PROMPTS_FLARE_PATH \
  $PROMPTS_TMP_PATH

# Other env variables
export OPENBLAS_NUM_THREADS=1
export GPUS_PER_NODE=1
export TP_SIZE=1

NODES=$(cat ${PBS_NODEFILE} | wc -l)
echo -e "\n\n$(date) Launching vLLM on ${NODES} nodes..."
python3 main_dsync.py --model $MODEL_DIR --ngpus-per-model $TP_SIZE --num-gpus_per_node $GPUS_PER_NODE
echo "$(date) All done!"
