#! /bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
#
# Changes:
# - Removed NVidia container build version message
###############################################################################

train_batch_size_phase2=${1:-14}
eval_batch_size=${2:-48}
learning_rate_phase2=${3:-"5e-5"}
beta1_phase2=${4:-0.9}
beta2_phase2=${5:-0.999}
precision=${6:-"fp16"}
use_xla=${7:-"true"}
num_hpus=${8:-8}
warmup_steps_phase2=${9:-"0"}
train_steps_phase2=${10:-8103}
save_checkpoints_steps=${11:-1000}
num_accumulation_steps_phase2=${12:-4}
bert_model=${13:-"large"}
samples_between_eval=${14:-150000}
stop_threshold=${15:-"0.720"}
samples_start_eval=${16:-3000000}
max_eval_steps=${17:-100}
is_dist_eval_enabled=${18:-"false"}
eval_only=${19:-"false"}
use_lightweight_checkpoint=${20:-"false"}

#Edit to save logs & checkpoints in a different directory
RESULTS_DIR=${RESULTS_DIR:-/results}

export BERT_CONFIG=${PRETRAINED_MODEL}/bert_config.json

PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--amp"
elif [ "$precision" = "fp32" ] ; then
   PREC="--noamp"
elif [ "$precision" = "tf32" ] ; then
   PREC="--noamp"
elif [ "$precision" = "manual_fp16" ] ; then
   PREC="--noamp --manual_fp16"
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$use_xla" = "true" ] ; then
   PREC="$PREC --use_xla"
   echo "XLA activated"
else
   PREC="$PREC --nouse_xla"
fi

horovod_str=""
if [ $USE_HOROVOD == "true" ]; then
   horovod_str="--horovod"
fi

#PHASE 1 Config
PHASE1_CKPT=${INITIAL_CHECKPOINT:-${HOME}/MLPerf_BERT_checkpoint}
PHASE1_CKPT=${PHASE1_CKPT}/model.ckpt-28252

#PHASE 2

seq_len=$P2_MAX_SEQ_LENGTH
max_pred_per_seq=76 #####
gbs_phase2=$(expr $train_batch_size_phase2 \* $num_accumulation_steps_phase2)


RESULTS_DIR_PHASE2=${RESULTS_DIR}/phase_2
run_per_ip mkdir -m 777 -p $RESULTS_DIR_PHASE2

INPUT_FILES=${INPUT_FILES:-${HOME}/tensorflow_datasets/MLPerf_BERT_Wiki/}
EVAL_FILES=${EVAL_FILES:-${HOME}/tensorflow_datasets/mlperf_bert_eval_dataset/}

enable_device_warmup=True

# Test run configuration
if [[ ! -z "${DUMP_CONFIG}" ]]
then
   enable_device_warmup=False
   train_steps_phase2=1
   num_accumulation_steps_phase2=4
   save_checkpoints_steps=0
   test_params="--deterministic_run=True --dump_config=${DUMP_CONFIG}"
fi



function check_dirs()
{
   # Check if all necessary files are available before training
   for DIR_or_file in $DATA_DIR $RESULTS_DIR $BERT_CONFIG ${PHASE1_CKPT}.meta; do
      if [ ! -d "$DIR_or_file" ] && [ ! -f "$DIR_or_file" ]; then
         echo "Error! $DIR_or_file directory missing. Please mount correctly"
         exit -1
      fi
   done
}

run_per_ip check_dirs || exit -1

if $eval_only
then
  echo -------------------------------------------------------------------------
  echo "Running evaluation"
  echo
  echo "python3 $SCRIPT_DIR/run_pretraining.py"
  echo "    input_files_dir=$INPUT_FILES"
  echo "    init_checkpoint=$PHASE1_CKPT"
  echo "    eval_files_dir=$EVAL_FILES"
  echo "    output_dir=$RESULTS_DIR_PHASE2"
  echo "    bert_config_file=$BERT_CONFIG"
  echo "    do_train=False"
  echo "    do_eval=True"
  echo "    train_batch_size=$train_batch_size_phase2"
  echo "    eval_batch_size=$eval_batch_size"
  echo "    max_eval_steps=$max_eval_steps"
  echo "    max_seq_length=$seq_len"
  echo "    max_predictions_per_seq=$max_pred_per_seq"
  echo "    num_train_steps=$train_steps_phase2"
  echo "    num_accumulation_steps=$num_accumulation_steps_phase2"
  echo "    num_warmup_steps=$warmup_steps_phase2"
  echo "    save_checkpoints_steps=$save_checkpoints_steps"
  echo "    learning_rate=$learning_rate_phase2"
  echo "    $horovod_str $PREC"
  echo "    allreduce_post_accumulation=True"
  echo "    enable_device_warmup=0"
  echo "    samples_between_eval=$samples_between_eval"
  echo "    stop_threshold=$stop_threshold"
  echo "    samples_start_eval=$samples_start_eval"
  echo "    is_dist_eval_enabled=$is_dist_eval_enabled"
  echo "    dllog_path=$RESULTS_DIR_PHASE2/bert_dllog.json"
  echo "    use_lightweight_checkpoint=$use_lightweight_checkpoint"
  echo -------------------------------------------------------------------------
  time $MPIRUN_CMD python3 $SCRIPT_DIR/run_pretraining.py \
      --input_files_dir=$INPUT_FILES \
      --init_checkpoint=$PHASE1_CKPT \
      --eval_files_dir=$EVAL_FILES\
      --output_dir=$RESULTS_DIR_PHASE2 \
      --bert_config_file=$BERT_CONFIG \
      --do_train=False \
      --do_eval=True \
      --train_batch_size=$train_batch_size_phase2 \
      --eval_batch_size=$eval_batch_size \
      --max_eval_steps=$max_eval_steps \
      --max_seq_length=$seq_len \
      --max_predictions_per_seq=$max_pred_per_seq \
      --num_train_steps=$train_steps_phase2 \
      --num_accumulation_steps=$num_accumulation_steps_phase2 \
      --num_warmup_steps=$warmup_steps_phase2 \
      --save_checkpoints_steps=$save_checkpoints_steps \
      --learning_rate=$learning_rate_phase2 \
      $horovod_str $PREC \
      --allreduce_post_accumulation=True \
      --enable_device_warmup=0 \
      --samples_between_eval=$samples_between_eval \
      --stop_threshold=$stop_threshold \
      --samples_start_eval=$samples_start_eval \
      --is_dist_eval_enabled=$is_dist_eval_enabled \
      --dllog_path=$RESULTS_DIR_PHASE2/bert_dllog.json \
      --use_lightweight_checkpoint=$use_lightweight_checkpoint
else
  echo -------------------------------------------------------------------------
  echo "Running the Pre-Training :: Phase 2"
  echo
  echo "python3 $SCRIPT_DIR/run_pretraining.py"
  echo "    input_files_dir=$INPUT_FILES"
  echo "    init_checkpoint=$PHASE1_CKPT"
  echo "    eval_files_dir=$EVAL_FILES"
  echo "    output_dir=$RESULTS_DIR_PHASE2"
  echo "    bert_config_file=$BERT_CONFIG"
  echo "    do_train=True"
  echo "    do_eval=False"
  echo "    is_dist_eval_enabled=$is_dist_eval_enabled"
  echo "    train_batch_size=$train_batch_size_phase2"
  echo "    eval_batch_size=$eval_batch_size"
  echo "    max_eval_steps=$max_eval_steps"
  echo "    max_seq_length=$seq_len"
  echo "    max_predictions_per_seq=$max_pred_per_seq"
  echo "    num_train_steps=$train_steps_phase2"
  echo "    num_accumulation_steps=$num_accumulation_steps_phase2"
  echo "    num_warmup_steps=$warmup_steps_phase2"
  echo "    save_checkpoints_steps=$save_checkpoints_steps"
  echo "    learning_rate=$learning_rate_phase2"
  echo "    $horovod_str $PREC"
  echo "    allreduce_post_accumulation=True"
  echo "    enable_device_warmup=$enable_device_warmup"
  echo "    samples_between_eval=$samples_between_eval"
  echo "    stop_threshold=$stop_threshold"
  echo "    samples_start_eval=$samples_start_eval"
  echo "    enable_habana_backend=True"
  echo "    enable_packed_data_mode=True"
  echo "    avg_seq_per_pack=2"
  echo "    compute_lm_loss_per_seq=False"
  echo "    dllog_path=$RESULTS_DIR_PHASE2/bert_dllog.json"
  echo "    use_lightweight_checkpoint=$use_lightweight_checkpoint"
  echo "    $test_params"
  echo -------------------------------------------------------------------------

  time $MPIRUN_CMD python3 $SCRIPT_DIR/run_pretraining.py \
      --input_files_dir=$INPUT_FILES \
      --init_checkpoint=$PHASE1_CKPT \
      --eval_files_dir=$EVAL_FILES\
      --output_dir=$RESULTS_DIR_PHASE2 \
      --bert_config_file=$BERT_CONFIG \
      --do_train=True \
      --do_eval=False \
      --is_dist_eval_enabled=$is_dist_eval_enabled \
      --train_batch_size=$train_batch_size_phase2 \
      --eval_batch_size=$eval_batch_size \
      --max_eval_steps=$max_eval_steps \
      --max_seq_length=$seq_len \
      --max_predictions_per_seq=$max_pred_per_seq \
      --num_train_steps=$train_steps_phase2 \
      --num_accumulation_steps=$num_accumulation_steps_phase2 \
      --num_warmup_steps=$warmup_steps_phase2 \
      --save_checkpoints_steps=$save_checkpoints_steps \
      --learning_rate=$learning_rate_phase2 \
      $horovod_str $PREC \
      --allreduce_post_accumulation=True \
      --enable_device_warmup=$enable_device_warmup \
      --samples_between_eval=$samples_between_eval \
      --stop_threshold=$stop_threshold \
      --samples_start_eval=$samples_start_eval \
      --enable_habana_backend \
      --enable_packed_data_mode \
      --avg_seq_per_pack=2 \
      --compute_lm_loss_per_seq=False \
      --dllog_path=$RESULTS_DIR_PHASE2/bert_dllog.json \
      --use_lightweight_checkpoint=$use_lightweight_checkpoint \
      --beta_1=$beta1_phase2 \
      --beta_2=$beta2_phase2 \
      $test_params
fi
