#!/bin/bash -l
#PBS -l select=<node>
#PBS -l walltime=1:00:00
#PBS -q <queue>
#PBS -A datascience
#PBS -l filesystems=home:flare

cd $PBS_O_WORKDIR

rm -r logs*

export ZE_FLAT_DEVICE_HIERARCHY=FLAT

module add mpifileutils

source ~/.vllm_envs/inference/bin/activate

# python3 main_multi_node_no_el.py

# python3 main_multi_node.py

python3 main_no_el.py --launch "ssh"

python3 main.py
