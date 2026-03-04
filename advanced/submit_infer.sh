#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR

rm -r logs*

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

module add mpifileutils

source ~/.vllm_envs/inference/bin/activate

python3 main_multi_node_no_el.py

# python3 main_no_el.py

# python3 main.py
