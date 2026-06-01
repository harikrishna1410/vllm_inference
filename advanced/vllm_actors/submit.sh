#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR

rm -r logs* script_logs/* ckpt_* ./.actor_ckpt

module load mpifileutils

source ~/.vllm_envs/inference/bin/activate


# # python3 main_offline_vllm_actors.py --cache-dir "/flare/datascience/balin/vllm/.cache" --model "meta-llama/Llama-3.1-70B-Instruct" --ngpus-per-model 8 --num-prompts 32
python3 main_offline_multinode_vllm_actors.py --num-prompts 32 --ngpus-per-model 2