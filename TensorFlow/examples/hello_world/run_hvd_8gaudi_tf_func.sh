#!/bin/bash
# USAGE ./run_hvd_8gaudi.sh
SCRIPT_DIR=`readlink -e ${BASH_SOURCE} | xargs -I {} dirname {}`
mpirun --allow-run-as-root -np 8 python3 ${SCRIPT_DIR}/example_tf_func_hvd.py
