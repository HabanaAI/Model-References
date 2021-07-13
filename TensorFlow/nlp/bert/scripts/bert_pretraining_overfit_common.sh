###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

BERT_BASE_DIR=$PRETRAINED_MODEL

DATASET_PATH=$HOME/tmp/bert-pretraining-overfit-dataset
RESULTS_PATH=${OUTPUT_DIR:-$HOME/tmp/bert-pretraining-overfit-output}
INIT_CHECKPOINT_PATH=/software/data/bert_checkpoints/$MODEL_TYPE/model.ckpt-0

seq_length=$P1_MAX_SEQ_LENGTH

if [ $seq_length == "128" ]; then
    max_pred_per_seq=20
elif [ $seq_length == "512" ]; then
    max_pred_per_seq=80
else
    print_warning "Unsupported max_sequence_length. Setting max_predictions_per_seq to floor(0.15*max_sequence_length). Please see -s parameter for details"
    max_pred_per_seq=$(echo "( 0.15 * $P1_MAX_SEQ_LENGTH ) / 1" | bc) # division by 1 truncates floating point
fi

function create_pretraining_data()
{
  if [ ! -d $DATASET_PATH ]; then
      mkdir -p $DATASET_PATH
      python3 $SCRIPT_DIR/create_pretraining_data.py \
        --input_file=$SCRIPT_DIR/data/sample_text.txt \
        --output_file=$DATASET_PATH/tf_examples.tfrecord \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --do_lower_case=True \
        --max_seq_length=$seq_length \
        --max_predictions_per_seq=$max_pred_per_seq \
        --masked_lm_prob=0.15 \
        --random_seed=12345 \
        --dupe_factor=5
  fi
}

run_per_ip create_pretraining_data

horovod_str=""
if [ $USE_HOROVOD == "true" ]; then
    horovod_str="--horovod"
fi

function prepare_results_path()
{
  mkdir -p $RESULTS_PATH
  rm -r $RESULTS_PATH/*
}

run_per_ip prepare_results_path

base_lr=0.006
num_acc_steps=1
learning_rate=$(echo "$base_lr * ( $P1_BATCH * $NUM_WORKERS_TOTAL * $num_acc_steps ) / 65536" | bc -l)

time $MPIRUN_CMD python3 $SCRIPT_DIR/run_pretraining.py \
  --input_files_dir=$DATASET_PATH \
  --eval_files_dir=$DATASET_PATH \
  --output_dir=$RESULTS_PATH \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT_CHECKPOINT_PATH \
  --train_batch_size=$P1_BATCH \
  --eval_batch_size=8 \
  --max_seq_length=$seq_length \
  --max_predictions_per_seq=$max_pred_per_seq \
  --num_train_steps=$P1_STEPS \
  --num_accumulation_steps=$num_acc_steps \
  --num_warmup_steps=$P1_WARMUP \
  --dllog_path=$RESULTS_PATH/bert_dllog.json \
  --learning_rate=$learning_rate \
  $horovod_str \
  --amp=False \
  --use_xla=False
