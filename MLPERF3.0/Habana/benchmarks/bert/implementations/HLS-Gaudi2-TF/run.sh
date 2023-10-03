#! /bin/bash

#set -x
###############################################################################
# Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
#
###############################################################################

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export BASE_PATH="$( cd "$(dirname "$(readlink -f ${SCRIPT_DIR}/defaults.cfg)" )" && pwd)"
export PYTHONPATH=${BASE_PATH}:${BASE_PATH}/../TensorFlow/common

PT_VERSION=`python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'`
TF_VERSION=`python3 -c "import tensorflow as tf; print(tf.__version__.replace('.', '_'))"`
PATCH_PATH=/usr/local/lib/python${PT_VERSION}/dist-packages/habana_frameworks/tensorflow/tf${TF_VERSION}/lib/habanalabs
export PYTHONPATH=${PATCH_PATH}:${PYTHONPATH}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-7}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-125}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
PRECISION=${PRECISION:-fp32}
WARMUP_STEPS=${WARMUP_STEPS:-0}
TRAIN_STEPS=${TRAIN_STEPS:-8103}
SAVE_CHECKPOINTS_STEPS=${SAVE_CHECKPOINTS_STEPS:-335}
NUM_ACCUMULATION_STEPS=${NUM_ACCUMULATION_STEPS:-4}
SAMPLES_BETWEEN_EVAL=${SAMPLES_BETWEEN_EVAL:-150080}
STOP_THRESHOLD=${STOP_THRESHOLD:-0.720}
SAMPLES_START_EVAL=${SAMPLES_START_EVAL:-3000000}
MAX_EVAL_STEPS=${MAX_EVAL_STEPS:-0}
IS_DIST_EVAL_ENABLED=${IS_DIST_EVAL_ENABLED:-false}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-512}
MAX_PRED_PER_SEQ=${MAX_PRED_PER_SEQ:-76}
FAST_PERF_ONLY=${FAST_PERF_ONLY:-0}
PACKED_DATA=${PACKED_DATA:-False}
TESTDATE=${TESTDATE}
TESTTIME=${TESTTIME}
LAMB_BETA_1=${LAMB_BETA_1:-0.9}
LAMB_BETA_2=${LAMB_BETA_2:-0.999}
EPSILON=${EPSILON:-1e-6}
LAMB_WEIGHT_DECAY_RATE=${LAMB_WEIGHT_DECAY_RATE:-0.01}
LAMB_LEARNING_RATE_DECAY_POLY_POWER=${LAMB_LEARNING_RATE_DECAY_POLY_POWER:-1.0}
NUM_WORKERS_PER_HLS=${NUM_WORKERS_PER_HLS:-4}
DO_TRAIN=${DO_TRAIN:-True}
DO_EVAL=${DO_EVAL:-True}
EXPERIMENTAL_SLACK=${EXPERIMENTAL_SLACK:-True}
NUM_DIST_EVAL_WORKERS=${NUM_DIST_EVAL_WORKERS:-0}
OPTIMIZER=${OPTIMIZER:-'lamb'}

export TF_BF16_CONVERSION=${BASE_PATH}/../TensorFlow/common/bf16_config/bert.json
export USE_LIGHTWEIGHT_CHECKPOINT=${USE_LIGHTWEIGHT_CHECKPOINT:-True}
export LIGHTWEIGHT_CHECKPOINT_IMPL=${LIGHTWEIGHT_CHECKPOINT_IMPL:-"basic"}
export USE_ASYNC_CHECKPOINTING=${USE_ASYNC_CHECKPOINTING:-False}
export BERT_CONFIG_FILE=${BERT_CONFIG_FILE:-${BERT_CONFIG_DIR}/bert_config.json}

if [[ $SIGNALING_FROM_GRAPH -eq 1 ]]; then
	export TF_DISABLE_SCOPED_ALLOCATOR=True
	export HOROVOD_FUSION_THRESHOLD=0
	export TF_USE_SIGNALING_FROM_ENCAP_OP=1
else
	export TF_USE_SIGNALING_FROM_ENCAP_OP=0
fi

# Currently sharded LAMB works only when ScopedAllocator is disabled and loop unrolling is False
if [ $OPTIMIZER == "sharded_lamb" ]; then
    export TF_DISABLE_SCOPED_ALLOCATOR=True
    AUX_PARAMS="${AUX_PARAMS} --loop_unrolling_for_train_op=False"
fi

# Under the hood, AMP (Arithmetic Mixed Precision) training is applied via TF_BF16_CONVERSION
# default precision is fp32.
precision="--noamp"

USE_HOROVOD=${USE_HOROVOD:-"False"}
if [ $USE_HOROVOD == "True" ]; then
   horovod="--horovod --allreduce_post_accumulation=True"
   IS_DIST_EVAL_ENABLED="True"
else
   horovod=""
fi
PROFILE=0
if [[ ${SYN_PROFILE} -eq 1 ]]; then
	PROFILE=1
fi
if [[ -n ${TF_PROFILE_STEPS} ]]; then
	TF_PROFILE_STEPS="--profile_steps=${TF_PROFILE_STEPS}"
	PROFILE=1
else
	TF_PROFILE_STEPS=''
fi
if [[ -n ${HW_PROFILE_RANGE} ]]; then
	export ENABLE_DEBUG_INFO=1
	PROFILE=1
fi

#PHASE 1 Config
export PHASE1_CKPT=${PHASE1_CKPT:-/root/datasets/bert_pretraining/MLPerf_BERT_checkpoint/model.ckpt-28252}
export INPUT_FILES_DIR=${INPUT_FILES_DIR:-/root/datasets/bert_pretraining/training}
export EVAL_FILES_DIR=${EVAL_FILES_DIR:-/root/datasets/bert_pretraining/evaluation}

#Generate Host Folder
if [ $USE_DRAM_OUTPUT == "True" ]; then
    host=$(hostname)
    if [ "$OMPI_COMM_WORLD_LOCAL_RANK" == "0" ]; then
        mkdir -p /mnt/dramfs
        mount -t tmpfs -o size=200g tmpfs /mnt/dramfs
    fi
    export OUTPUT_DIR=/mnt/dramfs/bert_gaudi${NUM_WORKERS_TOTAL}_${TESTDATE}_${TESTTIME}/${host}
    mkdir -p $OUTPUT_DIR
fi

# clear cache
if [[ $OMPI_COMM_WORLD_LOCAL_RANK -eq 0 ]]; then
  PROC_FS=${PROC_FS:-"/proc"}
  sync && echo 3 > $PROC_FS/sys/vm/drop_caches
fi

if [ $PACKED_DATA == "False" ]; then
   packing_arg=""
else
   packing_arg="--enable_packed_data_mode  --avg_seq_per_pack=2"
fi

AUX_PARAMS=$(echo ${AUX_PARAMS} | sed s/:/\ /g)

enable_device_warmup=True

TRAIN_COMMAND="python3 ${BASE_PATH}/../TensorFlow/nlp/bert/run_pretraining.py \
	--input_files_dir=$INPUT_FILES_DIR \
	--init_checkpoint=$PHASE1_CKPT \
	--eval_files_dir=$EVAL_FILES_DIR\
	--output_dir=$OUTPUT_DIR \
	--bert_config_file=$BERT_CONFIG_FILE \
	--do_train=$DO_TRAIN \
	--do_eval=$DO_EVAL \
	--experimental_slack=$EXPERIMENTAL_SLACK \
	--is_dist_eval_enabled=$IS_DIST_EVAL_ENABLED \
	--train_batch_size=$TRAIN_BATCH_SIZE \
	--eval_batch_size=$EVAL_BATCH_SIZE \
	--max_eval_steps=$MAX_EVAL_STEPS \
	--max_seq_length=$MAX_SEQ_LENGTH \
	--max_predictions_per_seq=$MAX_PRED_PER_SEQ \
	--num_train_steps=$TRAIN_STEPS \
	--num_accumulation_steps=$NUM_ACCUMULATION_STEPS \
	--num_warmup_steps=$WARMUP_STEPS \
	--save_checkpoints_steps=$SAVE_CHECKPOINTS_STEPS \
	--learning_rate=$LEARNING_RATE \
	$horovod \
	$precision \
	$packing_arg \
	--enable_device_warmup=$enable_device_warmup \
	--samples_between_eval=$SAMPLES_BETWEEN_EVAL \
	--stop_threshold=$STOP_THRESHOLD \
	--samples_start_eval=$SAMPLES_START_EVAL \
	--beta_1=$LAMB_BETA_1 \
	--beta_2=$LAMB_BETA_2 \
	--epsilon=$EPSILON \
	--weight_decay_rate=$LAMB_WEIGHT_DECAY_RATE \
	--power=$LAMB_LEARNING_RATE_DECAY_POLY_POWER \
	--enable_habana_backend \
	--dllog_path=$LOG_DIR/bert_dllog.json \
	--use_lightweight_checkpoint=$USE_LIGHTWEIGHT_CHECKPOINT \
	--lightweight_checkpoint_impl=$LIGHTWEIGHT_CHECKPOINT_IMPL \
	--use_async_checkpointing=$USE_ASYNC_CHECKPOINTING \
	--num_dist_eval_workers=$NUM_DIST_EVAL_WORKERS \
	--optimizer_type=$OPTIMIZER \
	${TF_PROFILE_STEPS} \
	${AUX_PARAMS}
"

LD_PRELOAD=${PRELOAD_PATH} ${TRAIN_COMMAND}

if [[ $OMPI_COMM_WORLD_LOCAL_RANK == "0" ]]; then
   rm -rf $OUTPUT_DIR/*/model.ckpt-*
   rm -rf $OUTPUT_DIR/*/checkpoint
   if [[ $USE_DRAM_OUTPUT == "True" ]]; then
	cp -r $LOG_DIR/result_* /root/scratch/bert/bert_gaudi${NUM_WORKERS_TOTAL}_${TESTDATE}_${TESTTIME}
	rm -rf /mnt/dramfs/bert_gaudi${NUM_WORKERS_TOTAL}_${TESTDATE}_${TESTTIME}
   fi
fi
