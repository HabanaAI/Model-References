###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################
import os

def MPI_is_distributed():
  return all([var in os.environ for var in ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]])

def MPI_world_rank():
    return os.environ.get("OMPI_COMM_WORLD_RANK", 0)

def MPI_barrier():
  if MPI_is_distributed():
      from mpi4py import MPI
      MPI.COMM_WORLD.Barrier()
