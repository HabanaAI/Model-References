#!/bin/bash

#####################################################################################
# Example: Pretraining phase 1 of BERT with 1.5B parameters on multicard i.e 8 cards
#####################################################################################

# Params: run_pretraining
DATA_DIR=/data/pytorch/bert_pretraining/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus
MODEL_CONFIG=./scripts/bert_1.5b_config.json
DS_CONFIG=./scripts/deepspeed_config_bert_1.5b.json
RESULTS_DIR=./results/bert_1.5b
MAX_SEQ_LENGTH=128
NUM_STEPS_PER_CP=1000000
MAX_STEPS=155000
LR=0.0015
WARMUP=0.05
CONST=0.25

# Params: DeepSpeed
NUM_NODES=1
NGPU_PER_NODE=8

CMD="python -u ./run_pretraining.py \
     --use_hpu \
     --warmup_proportion=$WARMUP \
     --constant_proportion=$CONST \
     --do_train \
     --bert_model=bert-base-uncased \
     --config_file=$MODEL_CONFIG \
     --json-summary=$RESULTS_DIR/dllogger.json \
     --output_dir=$RESULTS_DIR/checkpoints \
     --seed=12439 \
     --optimizer=nvlamb \
     --use_lr_scheduler \
     --input_dir=$DATA_DIR \
     --max_seq_length $MAX_SEQ_LENGTH \
     --max_predictions_per_seq=20 \
     --max_steps=$MAX_STEPS \
     --num_steps_per_checkpoint=$NUM_STEPS_PER_CP \
     --learning_rate=$LR \
     --deepspeed \
     --deepspeed_config=$DS_CONFIG"

mkdir -p $RESULTS_DIR

deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          $CMD