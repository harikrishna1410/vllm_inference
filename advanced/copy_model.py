import math
import os
import subprocess
import sys

import typer
from mpi4py import MPI
from typer import Typer

msync = Typer()


@msync.command()
def dsync_to_root(
    cache_dir: str, model: str, node_local_cache: str = "/tmp/model_cache", np: int = 16
):
    """
    Step 1: Pulls data from shared storage to the Root node using an internal mpirun call.
    DO NOT run this command with mpirun.
    """
    model_cache = os.path.join(cache_dir, "hub", f"models--{model.replace('/', '--')}")
    local_model_cache = os.path.join(
        node_local_cache, "hub", f"models--{model.replace('/', '--')}"
    )

    # Prevent MPI Inception: Abort if launched via mpirun
    if MPI.COMM_WORLD.size > 1:
        print(
            "ERROR: Do not run dsync-to-root with mpirun. Run it with standard python3.",
            file=sys.stderr,
        )
        print("It internally spawns its own mpirun subprocess.", file=sys.stderr)
        raise typer.Exit(code=1)

    print(
        f"Fetching from {model_cache} to {local_model_cache} using {np} local processes..."
    )

    os.makedirs(local_model_cache, exist_ok=True)

    cmd = [
        "mpirun",
        "-np",
        str(np),
        "-ppn",
        str(np),
        "dsync",
        f"{model_cache}/",
        f"{local_model_cache}/",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)

    print("stdout:\n", p.stdout)
    if p.stderr:
        print("stderr:\n", p.stderr, file=sys.stderr)

    if p.returncode != 0:
        print(f"ERROR: dsync failed with return code {p.returncode}", file=sys.stderr)
        raise typer.Exit(code=p.returncode)


@msync.command()
def scatter_from_root(
    node_local_cache: str, model: str, CHUNK_SIZE: int = 100 * 1024 * 1024
):
    """
    Step 2: Broadcasts the local cache from Root to all other MPI ranks.
    MUST be run with mpirun (e.g., mpirun -n 1000 python3 msync.py scatter-from-root /tmp/model_cache)
    """
    comm = MPI.COMM_WORLD
    my_rank = comm.rank

    model_cache = os.path.join(
        node_local_cache, "hub", f"models--{model.replace('/', '--')}"
    )
    # 1. Gather file list on root (Recursive)
    if my_rank == 0:
        files = []
        # os.walk efficiently yields directories and files separately
        for dirpath, _, filenames in os.walk(model_cache):
            for filename in filenames:
                # We only append actual files, no extra syscalls needed
                files.append(os.path.join(dirpath, filename))

        if not files:
            print(
                f"WARNING: No files found in {model_cache}. Nothing to sync.",
                file=sys.stderr,
            )
    else:
        files = None

    files = comm.bcast(files, root=0)

    # Exit cleanly across all ranks if no files were found
    if not files:
        return

    # Allocate buffer once
    recv_buffer = bytearray(CHUNK_SIZE)

    for file in files:
        # Ensure target directory tree exists on worker nodes to prevent I/O crashes
        if my_rank > 0:
            os.makedirs(os.path.dirname(file), exist_ok=True)

        # 2. Get and broadcast file size
        if my_rank == 0:
            file_size = os.path.getsize(file)
        else:
            file_size = None

        file_size = comm.bcast(file_size, root=0)

        # Handle empty files gracefully without entering the chunking logic
        if file_size == 0:
            if my_rank > 0:
                open(file, "wb").close()
            continue

        nchunks = math.ceil(file_size / CHUNK_SIZE)

        # 3. Open files
        if my_rank == 0:
            f = open(file, "rb")
        else:
            f = open(file, "wb")

        # 4. Stream chunks safely
        try:
            for cid in range(nchunks):
                # Cleanly calculate remaining bytes to avoid dropping exact-multiple chunks
                current_chunk_size = min(CHUNK_SIZE, file_size - cid * CHUNK_SIZE)

                if my_rank == 0:
                    chunk = f.read(current_chunk_size)
                    recv_buffer[:current_chunk_size] = chunk

                # Broadcast ONLY the valid bytes for this iteration
                comm.Bcast([recv_buffer, current_chunk_size, MPI.BYTE], root=0)

                if my_rank > 0:
                    f.write(recv_buffer[:current_chunk_size])
        finally:
            f.close()

    if my_rank == 0:
        print(
            f"Successfully scattered {len(files)} files to {comm.size - 1} worker nodes."
        )


if __name__ == "__main__":
    msync()
