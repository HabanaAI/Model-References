#! /bin/bash

#set -x
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
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
MAX_PRED_PER_SEQ=${MAX_PRED_PER_SEQ:-76} #####
FAST_PERF_ONLY=${FAST_PERF_ONLY:-0}
PACKED_DATA=${PACKED_DATA:-False}
TESTDATE=${TESTDATE}
TESTTIME=${TESTTIME}
#START_WARMUP_STEPS=${START_WARMUP_STEPS}
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

export ENABLE_EXPERIMENTAL_FLAGS=${ENABLE_EXPERIMENTAL_FLAGS:-True}
export RUN_TPC_FUSER=${RUN_TPC_FUSER:-False}
export TF_BF16_CONVERSION=${BASE_PATH}/../TensorFlow/common/bf16_config/bert.json
export USE_LIGHTWEIGHT_CHECKPOINT=${USE_LIGHTWEIGHT_CHECKPOINT:-True}
export LIGHTWEIGHT_CHECKPOINT_IMPL=${LIGHTWEIGHT_CHECKPOINT_IMPL:-"basic"}
export USE_ASYNC_CHECKPOINTING=${USE_ASYNC_CHECKPOINTING:-False}
#export TF_MODULES_RELEASE_BUILD=${TF_MODULES_RELEASE_BUILD:-/usr/lib/habanalabs/}
#export GC_KERNEL_PATH=${GC_KERNEL_PATH:-/usr/lib/habanalabs/libtpc_kernels.so}
#export HABANA_LOGS=${HABANA_LOGS:-/var/log/habana_logs/}

#Edit to save logs & checkpoints in a different directory
#export OUTPUT_DIR=${OUTPUT_DIR:-/tmp/bert_pretrain/phase_2}
export BERT_CONFIG_FILE=${BERT_CONFIG_FILE:-${BERT_CONFIG_DIR}/bert_config.json}

if [[ $NUM_WORKERS_TOTAL -ge "256" ]]; then
    export HCL_SLICE_STRONG_ORDER="True"
    export HCL_SLICE_SIZE=900MB
elif [[ $NUM_WORKERS_TOTAL -ge "128" ]]; then
    export HCL_SLICE_STRONG_ORDER="True"
    export HCL_SLICE_SIZE=128MB
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
	sync
	echo 3 > /proc/sys/vm/drop_caches
fi

#env > env_${OMPI_COMM_WORLD_LOCAL_RANK}.log
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
	${TF_PROFILE_STEPS} \
	${AUX_PARAMS}
"
# --start_warmup_steps=$START_WARMUP_STEPS \

function set_profiling()
{
	if [[ ${SYN_PROFILE} -eq 1 ]]; then
		export HABANA_SYNAPSE_LOGGER=synapse
		rm -f .local.synapse_log.*
		rm -f rank*
	fi
	if [[ -n ${TF_PROFILE_STEPS} ]]; then
		export HABANA_SYNAPSE_LOGGER=tf
	fi
	if [[ -n ${HW_PROFILE_RANGE} ]]; then
		export HABANA_SYNAPSE_LOGGER=hw
		export HABANA_PROFILE_RANGE=${HW_PROFILE_RANGE}
		export HABANA_PROFILE=1
		export HABANA_PROFILE_CONFIG_PATH=${PROFILE_CONFIG}
		export TF_HOOK_MODE=all
		# export MODEL_GARDEN_ROOT=/root/MLPERF  # This is moved to ~/.bashrc
		#export LOG_LEVEL_HwTrace=1 LOG_LEVEL_PROF_hl0=1 LOG_LEVEL_ALL=5
		rm -f mergedHW*
		rm -f .local.synapse_log.*
	fi
}

if [[ $PROFILE_ALL -eq 1 ]]; then
	set_profiling
else
	if [[ $OMPI_COMM_WORLD_LOCAL_RANK -eq 0 ]]; then
		set_profiling
	fi
fi

if [[ $CPU_BIND_TYPE == "numa" ]] && [[ $HLS_TYPE == "HLS1-H" ]]; then
	LOCAL_SNC_VALUE=$(( OMPI_COMM_WORLD_LOCAL_RANK ))
	if [[ $LOCAL_SNC_VALUE -eq 0 ]]; then
		CPU_RANGE="0,2,3,7,8,9,14,15,16,17,21,22,23"
		export HLS1_MODULE_ID=3 # Device /dev/hl0
	elif [[  $LOCAL_SNC_VALUE -eq 2 ]]; then
		CPU_RANGE="28,29,30,31,35,36,37,42,43,44,45,49,50,51"
		export HLS1_MODULE_ID=2   # Device /dev/hl2
	elif [[  $LOCAL_SNC_VALUE -eq 1 ]]; then
		CPU_RANGE="4,5,6,10,11,12,13,18,19,20,24,25,26,27"
		export HLS1_MODULE_ID=0  # Device /dev/hl1
	else
		CPU_RANGE="33,34,38,39,40,41,46,47,48,52,53,54,55"
		export HLS1_MODULE_ID=1   # Device /dev/hl3
	fi
	LD_PRELOAD=${PRELOAD_PATH} numactl --physcpubind=${CPU_RANGE} ${TRAIN_COMMAND}
else
	LD_PRELOAD=${PRELOAD_PATH} ${TRAIN_COMMAND}
fi

if [[ $OMPI_COMM_WORLD_LOCAL_RANK == "0" ]]; then
   rm -rf $OUTPUT_DIR/*/model.ckpt-*
   rm -rf $OUTPUT_DIR/*/checkpoint
   if [[ $USE_DRAM_OUTPUT == "True" ]]; then
	cp -r $LOG_DIR/result_* /root/scratch/bert/bert_gaudi${NUM_WORKERS_TOTAL}_${TESTDATE}_${TESTTIME}
	rm -rf /mnt/dramfs/bert_gaudi${NUM_WORKERS_TOTAL}_${TESTDATE}_${TESTTIME}
   fi
fi
#numactl --physcpubind=${CPU_RANGE} ${TRAIN_COMMAND}
