#!/bin/bash
# USAGE ./run_hvd_8gaudi.sh
export HCL_CONFIG_PATH=${HCL_CONFIG_PATH:-`pwd`/hcl_config.json}
echo "Using config file $HCL_CONFIG_PATH"
mpirun --allow-run-as-root -np 8 python3 example_hvd.py
