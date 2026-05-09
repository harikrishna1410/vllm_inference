#!/bin/bash -l
#PBS -l select=512
#PBS -l walltime=01:00:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR

rm -r logs* script_logs/* ckpt_* ./.actor_ckpt

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

module add mpifileutils

source ~/.vllm_envs/inference/bin/activate

# python3 main.py

python3 main_offline.py