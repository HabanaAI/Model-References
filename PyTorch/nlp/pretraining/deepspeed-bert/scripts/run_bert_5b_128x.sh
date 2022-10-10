#!/bin/bash

###############################################################################################
# Example: Pretraining phase 1 of BERT with 5B parameters on 4 HLS1 boxes with 8 devices.
###############################################################################################

# Params: run_pretraining
DATA_DIR=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus
MODEL_CONFIG=./scripts/bert_5b_config.json
DS_CONFIG=./scripts/deepspeed_config_bert_5b.json
HOSTSFILE=./scripts/hostsfile
RESULTS_DIR=./results/bert_5b
MAX_SEQ_LENGTH=128
NUM_STEPS_PER_CP=500
MAX_STEPS=2000000
LR=1e-4
LOG_FREQ=1

# Params: DeepSpeed
NUM_NODES=16
NGPU_PER_NODE=8

DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

# Multi-node environment
CMD="python -u ./run_pretraining.py \
     --use_hpu \
     --optimizer=lans \
     --scheduler_degree=1.0 \
     --resume_from_checkpoint \
     --do_train \
     --bert_model=bert-base-uncased \
     --config_file=$MODEL_CONFIG \
     --json-summary=$RESULTS_DIR/dllogger.json \
     --output_dir=$RESULTS_DIR/checkpoints \
     --seed=12439 \
     --input_dir=$DATA_DIR \
     --max_seq_length $MAX_SEQ_LENGTH \
     --max_predictions_per_seq=20 \
     --max_steps=$MAX_STEPS \
     --num_steps_per_checkpoint=$NUM_STEPS_PER_CP \
     --learning_rate=$LR \
     --disable_progress_bar \
     --log_freq=$LOG_FREQ \
     --deepspeed \
     --deepspeed_config=$DS_CONFIG"

#Configure multinode
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]
then
    MULTINODE_CMD="--hostfile=$HOSTSFILE"
fi

mkdir -p $RESULTS_DIR
deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          --master_addr 10.10.10.10 \
          $MULTINODE_CMD \
          $CMD
