import os
from typing import List, Optional


def get_mpi_pe(num_workers: int) -> int:
    """Get number of PE (Parallel Environments) based on number of cores."""
    cpus = os.cpu_count()
    return cpus//num_workers//2


def create_mpi_cmd(num_workers: int, output_file: Optional[str] = None,
                   tag_output: bool = False) -> List[str]:
    """Return mpi prefix for running on <num_workers> cards.

    :param num_workers: Used to calculate num processes and core bindings.
    :param output_file: Path to a file that will contain logs from execution.
                        Directory that this file is in has to exist.
    """
    mpi_cmd_parts = []
    mpi_cmd_parts.append("mpirun")
    mpi_cmd_parts.append("--allow-run-as-root")
    mpi_pe = get_mpi_pe(num_workers)
    mpi_cmd_parts.extend(["--bind-to", "core", "--map-by", f"socket:PE={mpi_pe}"])
    mpi_cmd_parts.extend(["--np", f"{num_workers}"])

    if tag_output or output_file is not None:
        mpi_cmd_parts.append(f"--tag-output --merge-stderr-to-stdout")

    if output_file is not None:
        if not os.path.isdir(os.path.dirname(output_file)):
            raise OSError(f"Requested output file {output_file}, but the directory doesn't exist.")
        mpi_cmd_parts.append(f"--output-filename {output_file}")


    return mpi_cmd_parts
