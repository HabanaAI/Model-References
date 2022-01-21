#!/bin/bash
# USAGE ./run_single_gaudi.sh
SCRIPT_DIR=`dirname $(readlink -e ${BASH_SOURCE[0]})`
python3 ${SCRIPT_DIR}/example.py
