#!/usr/bin/env python3
"""
Weak scaling test for vLLM inference.
Runs the same model configuration across multiple nodes.
Each node gets its own model instance with dedicated client and server.
"""
import json
import os
import logging
from ensemble_launcher.config import SystemConfig, LauncherConfig


def generate_config(nnodes, model, ngpus, num_prompts=10):
    """
    Generate ensemble config for weak scaling test.
    
    Args:
        nnodes: Number of nodes in the ensemble
        model: Model name to test
        ngpus: Number of GPUs to use per model instance
        num_prompts: Number of prompts each client will send
    """
    
    # Create one task per node (weak scaling)
    # Each node runs the same model with same GPU count
    config = {
        "ensembles": {
            f"{ngpus}gpus_{model.replace('/', '--')}": {
                "nnodes": 1,
                "ppn": 1,  # One process per node
                "ngpus_per_process": int(ngpus),
                "relation": "one-to-one",
                "cmd_template": "./infer_no_ray.sh {idx} {port} {ngpus} {model} {cache_dir} {tmpdir} {num_prompts}",
                "idx": list(range(1, nnodes + 1)),
                "port": [8000 + i for i in range(nnodes)],
                "ngpus": [int(ngpus)] * nnodes,
                "model": [model] * nnodes,
                "cache_dir": [os.path.join(os.getcwd(), ".cache")] * nnodes,
                "tmpdir": ["/tmp"] * nnodes,
                "num_prompts": [num_prompts] * nnodes,
            }
        }
    }
    
    fname = "./ensemble_config.json"
    with open(fname, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"\nGenerated ensemble config for weak scaling test:")
    print(f"  Model: {model}")
    print(f"  GPUs per instance: {ngpus}")
    print(f"  Number of nodes (ensemble size): {nnodes}")
    print(f"  Prompts per client: {num_prompts}")
    print(f"  Total GPU usage: {ngpus * nnodes}")
    print(f"  Config saved to: {fname}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Launch weak scaling test for vLLM inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run weak scaling test with 8 GPUs per instance
  python launch_ensembles.py --ngpus 8
  
  # Test with different model and more prompts
  python launch_ensembles.py --model meta-llama/Llama-3.1-70B-Instruct --ngpus 4 --num-prompts 50
  
  # Single GPU per instance, 20 prompts
  python launch_ensembles.py --ngpus 1 --num-prompts 20
        """
    )
    parser.add_argument(
        "--nnodes", 
        type=int, 
        default=1, 
        help="Number of GPUs to use per model instance (default: 8)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct", 
        help="Model name to use for testing"
    )
    parser.add_argument(
        "--ngpus", 
        type=int, 
        default=8, 
        help="Number of GPUs to use per model instance (default: 8)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of prompts each client will send for inference (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Generate ensemble configuration
    generate_config(args.nnodes, args.model, args.ngpus, args.num_prompts)
    
    # Setup system configuration for Aurora
    cpus = list(range(104))
    cpus.pop(52)  # Remove NUMA boundary CPU
    cpus.pop(0)   # Remove first CPU
    system_config = SystemConfig(
        name="aurora", 
        gpus=[f"{i}" for i in range(12)], 
        cpus=cpus, 
        ncpus=102
    )
    with open("system_config.json", "w") as f:
        json.dump(system_config.model_dump(), f, indent=4)
    
    # Setup launcher configuration
    launcher_config = LauncherConfig(
        child_executor_name="mpi", 
        task_executor_name="multiprocessing", 
        comm_name="zmq", 
        worker_logs=True, 
        master_logs=True, 
        log_level=logging.INFO
    )
    with open("launcher_config.json", "w") as f:
        json.dump(launcher_config.model_dump(), f, indent=4)

