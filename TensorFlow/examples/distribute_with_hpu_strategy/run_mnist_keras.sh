#!/bin/bash
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
echo "Distributed MNIST Training using Keras"
echo "An example of multi-worker training using HPUStrategy, where worker processes are spawned using Open MPI"

SCRIPT_PATH="`dirname \"$0\"`"          # A relative path from the current working directory to the directory containing this script.
PYTHON=${PYTHON:-python3}               # Python interpreter (command).
NUM_WORKERS=${NUM_WORKERS:-2}           # Number of worker processes participating in a training cluster.

echo NUM_WORKERS=$NUM_WORKERS

# Spawn several processes running the same Python training script.
# This example takes advantage of 'mpirun' of OpenMPI package.
# Note: OpenMPI is not mandatory for distributed training with HPUStrategy.
set -x
PYTHONPATH="$PYTHONPATH:$SCRIPT_PATH" \
    mpirun -np $NUM_WORKERS --tag-output --allow-run-as-root --bind-to none $PYTHON -m mnist_keras "$@"
set +x
