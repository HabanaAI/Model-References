#!/bin/bash

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

precision=${2:-"fp32"}
train_batch_size_p1=${3:-2}
learning_rate_p1=${4:-"6e-3"}
num_hpu=${5:-0} #Add number of HPUs
xla=${6:-"xla"}
warmup_steps_p1=${7:-"5"}
train_steps_p1=${8:-20}
save_checkpoint_steps=${9:-20}
resume_training=${10:-"false"}
optimizer=${11:-"lamb"}
accumulate_gradients=${12:-"true"}
gradient_accumulation_steps_p1=${13:-4}
seed=${14:-100}
job_name=${15:-"electra_lamb_pretraining"}
train_batch_size_p2=${16:-2}
learning_rate_p2=${17:-"4e-3"}
warmup_steps_p2=${18:-"5"}
train_steps_p2=${19:-20}
gradient_accumulation_steps_p2=${20:-4}
ELECTRA_MODEL=${21:-"base"}
DATASET_P1="/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_128/books_wiki_en_corpus/training/books_wiki*" #change this path for dataset seq_len_128
DATA_DIR_P1=${22:-"$DATASET_P1"}
DATASET_P2="/data/tensorflow/bert/books_wiki_en_corpus/tfrecord/seq_len_512/books_wiki_en_corpus/training/books_wiki*" #change this path for dataset seq_len_512
DATA_DIR_P2=${23:-"$DATASET_P2"}

CODEDIR=${24:-"."} #change this path for location of ELECTRA directory
init_checkpoint=${25:-"None"}
restore_checkpoint=${restore_checkpoint:-"true"}
RESULTS_DIR=$CODEDIR/results

if [ $1 == "hpu" ] ; then
    use_hpu="true"
else
    use_hpu="false"
fi

if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi

PREFIX=""
TEST_RESULT=$(awk 'BEGIN {print ('1' <= '${num_hpu}')}')
if [ "$TEST_RESULT" == 1 ] ; then
    PREFIX="mpirun -np $num_hpu --allow-run-as-root "
fi

if [ "$precision" = "hmp" ] ; then
   PREC="--hmp"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

if [ "$xla" = "xla" ] ; then
   PREC="$PREC --xla"
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_p1"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--restore_checkpoint=latest"
fi

if [ "$init_checkpoint" != "None" ] ; then
   CHECKPOINT="--restore_checkpoint=$init_checkpoint"
fi

CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --model_name=${ELECTRA_MODEL}"
CMD+=" --pretrain_tfrecords=$DATA_DIR_P1"
CMD+=" --model_size=${ELECTRA_MODEL}"
CMD+=" --train_batch_size=$train_batch_size_p1"
CMD+=" --max_seq_length=128 --disc_weight=50.0 --generator_hidden_size=0.3333333 "
CMD+=" --num_train_steps=$train_steps_p1"
CMD+=" --num_warmup_steps=$warmup_steps_p1"
CMD+=" --save_checkpoints_steps=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_p1"
CMD+=" --optimizer=${optimizer} --skip_adaptive --opt_beta_1=0.878 --opt_beta_2=0.974 --lr_decay_power=0.5"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" --log_dir ${RESULTS_DIR}"
CMD+=" --use_hpu=$use_hpu "
CMD+=" --log_freq=1 "

CMD="$PREFIX $PYTHON $CMD"
echo "Launch command: $CMD"

printf -v TAG "electra_pretraining_phase1_%s" "$precision"
DATESTAMP=`date +'%y%m%d%H%M%S'`
LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
printf "Logs written to %s\n" "$LOGFILE"

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished pretraining phase1"

#Start Phase2
ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_p2"
fi

RESTORE_CHECKPOINT=""
if [ "$restore_checkpoint" == "true" ] ; then
   RESTORE_CHECKPOINT="--restore_checkpoint=latest --phase2"
fi

CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --model_name=${ELECTRA_MODEL}"
CMD+=" --pretrain_tfrecords=$DATA_DIR_P2"
CMD+=" --model_size=${ELECTRA_MODEL}"
CMD+=" --train_batch_size=$train_batch_size_p2"
CMD+=" --max_seq_length=512 --disc_weight=50.0 --generator_hidden_size=0.3333333 ${RESTORE_CHECKPOINT}"
CMD+=" --num_train_steps=$train_steps_p2"
CMD+=" --num_warmup_steps=$warmup_steps_p2"
CMD+=" --save_checkpoints_steps=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_p2"
CMD+=" --optimizer=${optimizer} --skip_adaptive --opt_beta_1=0.878 --opt_beta_2=0.974 --lr_decay_power=0.5"
CMD+=" --seed=$seed"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" --log_dir ${RESULTS_DIR}"
CMD+=" --use_hpu=$use_hpu "
CMD+=" --log_freq=1 "

CMD="$PREFIX $PYTHON $CMD"
echo "Launch command: $CMD"


printf -v TAG "electra_pretraining_phase2_%s" "$precision"
DATESTAMP=`date +'%y%m%d%H%M%S'`
LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
printf "Logs written to %s\n" "$LOGFILE"

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished pretraining phase2"
