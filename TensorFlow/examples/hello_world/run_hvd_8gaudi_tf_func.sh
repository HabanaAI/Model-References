#!/bin/bash
# USAGE ./run_hvd_8gaudi.sh
SCRIPT_DIR=`dirname $(readlink -e ${BASH_SOURCE[0]})`
mpirun --allow-run-as-root -np 8 $PYTHON ${SCRIPT_DIR}/example_tf_func_hvd.py
