#!/bin/bash

#

if [ -n "${DATASET_PATH_UNCOMPRESSED+x}" ] && [ -d "$DATASET_PATH_UNCOMPRESSED" ]; then
    if [ -n "${DATASET_PATH_OUTPUT+x}" ] && [ -d "$DATASET_PATH_OUTPUT" ]; then
        python prepare_synthetic_data.py
    fi
else
  export DATASET_PATH_UNCOMPRESSED=/tmp/input/
  export DATASET_PATH_OUTPUT=/tmp/output
  python prepare_synthetic_data.py
fi

cd $DATASET_PATH_OUTPUT
tar -cvf SD_synthetic_data_10001.tar *
