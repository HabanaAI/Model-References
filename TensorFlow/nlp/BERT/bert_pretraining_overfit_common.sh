
BERT_BASE_DIR=$PRETRAINED_MODEL

DATASET_PATH=$HOME/tmp/bert-pretraining-overfit-dataset
RESULTS_PATH=$HOME/tmp/bert-pretraining-overfit-output

function create_pretraining_data()
{
  if [ ! -d $DATASET_PATH ]; then
      mkdir -p $DATASET_PATH
      python3 $SCRIPT_DIR/create_pretraining_data.py \
        --input_file=$SCRIPT_DIR/sample_text.txt \
        --output_file=$DATASET_PATH/tf_examples.tfrecord \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --do_lower_case=True \
        --max_seq_length=128 \
        --max_predictions_per_seq=20 \
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

time $MPIRUN_CMD python3 $SCRIPT_DIR/pretraining/run_pretraining.py \
  --input_files_dir=$DATASET_PATH \
  --eval_files_dir=$DATASET_PATH \
  --output_dir=$RESULTS_PATH \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=$P1_BATCH \
  --eval_batch_size=8 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=$P1_STEPS \
  --num_accumulation_steps=1 \
  --num_warmup_steps=$P1_WARMUP \
  --dllog_path=$RESULTS_PATH/bert_dllog.json \
  --learning_rate=2e-5 \
  $horovod_str \
  --amp=False \
  --use_xla=False
