#!/bin/bash -l
#PBS -l select=2
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR

rm -r logs*

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

source ~/.vllm_envs/inference/bin/activate

el "ensemble_config.json" --system-config-file "system_config.json" --launcher-config-file "launcher_config.json" --async-orchestrator
