#!/bin/bash
###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
# Runs benchmark and reports time to convergence
set -e

# Source a config file passed after `-c` or `--config` argument
if [[ "$1" == "-c" || "$1" == "--config" ]]; then
    shift
    CONFIG_FILE=$1
    if [ -f $CONFIG_FILE ]; then
      source $CONFIG_FILE
    else
      echo "Config file ${CONFIG_FILE} does not exist"
    fi
fi

NUM_WORKERS_PER_HLS=${NUM_WORKERS_PER_HLS:-8}

MODEL_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASET_DIR=${DATASET_DIR:-"/data"}
MAX_EPOCHS=${MAX_EPOCHS:-10000}
START_EVAL_AT=${START_EVAL_AT:-1000}
EVALUATE_EVERY=${EVALUATE_EVERY:-20}
QUALITY_THRESHOLD=${QUALITY_THRESHOLD:-"0.908"}
BATCH_SIZE=${BATCH_SIZE:-7}
LR=${LR:-"2.0"}
LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-1000}
LR_DECAY_EPOCHS=${LR_DECAY_EPOCHS:-""}
LR_DECAY_FACTOR=${LR_DECAY_FACTOR:-"1.0"}
LOG_DIR=${LOG_DIR:-"/tmp"}

# Set bfloat16 and float32 lists used by Automatic Mixed Precision package - torch.amp
LOWER_LIST=${LOWER_LIST:-"$MODEL_DIR/ops_bf16.txt"}
FP32_LIST=${FP32_LIST:-"$MODEL_DIR/ops_fp32.txt"}

# Determine the number of available cores for each process
MPI_MAP_BY_PE=`lscpu | grep "^CPU(s):"| awk -v NUM=${NUM_WORKERS_PER_HLS} '{print int($2/NUM/2)}'`

if [ -d ${DATASET_DIR} ]; then
  # Start timing
  start=$(date +%s)
  start_fmt=$(date +%Y-%m-%d\ %r)
  echo "STARTING TIMING RUN AT $start_fmt"

  # Clear caches
  PROC_FS=${PROC_FS:-"/proc"}
  sync && echo 3 > $PROC_FS/sys/vm/drop_caches

  SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
  set -x
  # Run UNet3D training
  mpirun \
  --allow-run-as-root \
  --merge-stderr-to-stdout \
  --np $NUM_WORKERS_PER_HLS \
  --bind-to core \
  --rank-by core \
  --map-by slot:PE=$MPI_MAP_BY_PE \
  --report-bindings \
  -x LOWER_LIST=$LOWER_LIST \
  -x FP32_LIST=$FP32_LIST \
  python $SCRIPT_DIR/main.py \
    --data_dir $DATASET_DIR \
    --epochs $MAX_EPOCHS \
    --evaluate_every $EVALUATE_EVERY \
    --start_eval_at $START_EVAL_AT \
    --quality_threshold $QUALITY_THRESHOLD \
    --batch_size $BATCH_SIZE \
    --optimizer sgd \
    --learning_rate $LR \
    --lr_warmup_epochs $LR_WARMUP_EPOCHS \
    --lr_decay_epochs $LR_DECAY_EPOCHS \
    --lr_decay_factor $LR_DECAY_FACTOR \
    --log_dir $LOG_DIR \
    --device hpu \
    --amp
  set +x
  # end timing
  end=$(date +%s)
  end_fmt=$(date +%Y-%m-%d\ %r)
  echo "ENDING TIMING RUN AT $end_fmt"

  # report result
  result=$(( $end - $start ))
  result_name="image_segmentation"

  echo "RESULT,$result_name,$result,$USER,$start_fmt"
else
  echo "Directory ${DATASET_DIR} does not exist"
fi