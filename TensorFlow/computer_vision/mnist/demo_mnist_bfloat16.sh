#!/bin/bash

WORKDIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
export PYTHONPATH=$(dirname $WORKDIR):$PYTHONPATH
export LOG_LEVEL_ALL=6
source $WORKDIR/../../common/common.sh
setup_libjemalloc

pushd ${WORKDIR}
python3 demo_mnist.py --batch_size=64 --iterations=400 --data_type=bf16
popd
