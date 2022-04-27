#!/usr/bin/env bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

precision=${2:-"fp32"}
electra_model=${3:-"google/electra-base-discriminator"}
epochs=${4:-"2"}
batch_size=${5:-"16"}
infer_batch_size=${6:-"128"}
learning_rate=${7:-"4e-4"}
num_hpu=${8:-"0"} #Add number of HPUs
seed=${9:-"$RANDOM"}
SQUAD_VERSION=${10:-"1.1"}
squad_dir=${11:-"squad_dir_loc"}             # SQUAD Data location
OUT_DIR=${12:-"results/"}
init_checkpoint=${13:-"init_checkpoint_loc"} # Electra Pretraind model in https://storage.googleapis.com/electra-data/electra_base.zip
mode=${14:-"train_eval"}
env=${15:-"interactive"}
cache_dir=${16:-"$squad_dir"}
max_steps=${17:-"-1"}
train_iter_per_epoch=${18:-"-1"}


echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

if [ $1 == "hpu" ] ; then
    use_hpu="true"
else
    use_hpu="false"
fi

use_fp16=""
if [ "$precision" = "hmp" ] ; then
  echo "mixed-precision training and xla activated!"
  use_fp16="--hmp --xla"
fi

mpi_command=""
TEST_RESULT=$(awk 'BEGIN {print ('1' <= '${num_hpu}')}')
if [ "$TEST_RESULT" == 1 ] ; then
    mpi_command="mpirun -np $num_hpu --allow-run-as-root "
fi

if [ "$env" = "cluster" ] ; then
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" "
fi

v2=""
echo "Running SQuAD-v$SQUAD_VERSION"
if [ "$SQUAD_VERSION" = "2.0" ] ; then
  v2=" --version_2_with_negative "
fi

# CMD=" $mpi_command python run_tf_squad_new.py " #might have to change to:
CMD=" $mpi_command python run_tf_squad.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_batch_size=$infer_batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v$SQUAD_VERSION.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_batch_size=$infer_batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_batch_size=$infer_batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v$SQUAD_VERSION.py "
  CMD+="--do_eval "
fi

CMD+=" $v2 "
CMD+=" --data_dir $squad_dir "
CMD+=" --do_lower_case "
CMD+=" --electra_model=$electra_model "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --warmup_proportion 0.05 "
CMD+=" --weight_decay_rate 0.01 "
CMD+=" --layerwise_lr_decay 0.8 "
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --beam_size 5 "
CMD+=" --joint_head True "
CMD+=" --null_score_diff_threshold -5.6 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" $use_fp16"
CMD+=" --cache_dir=$cache_dir "
CMD+=" --max_steps=$max_steps "
CMD+=" --vocab_file=vocab/vocab.txt"
CMD+=" --use_hpu=$use_hpu "
CMD+=" --log_freq=1 "
CMD+=" --train_iter_per_epoch=$train_iter_per_epoch "

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE
