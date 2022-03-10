#!/bin/bash

if [[ $DEBUG -eq 1 ]]; then
    set -x
    env
    #LOG_LEVEL:0 - TRACE, 1 - DEBUG, 2 - INFO, 3 - WARNING, 4 - ERROR, 5 - CRITICAL, 6 - OFF
    export LOG_LEVEL_ALL_HCL=2
else
    export LOG_LEVEL_ALL_HCL=6
fi

if [ -z $BASE_PATH ]; then
	BASE_PATH="$( cd "$(dirname "$(readlink -f ./defaults.cfg)" )" && pwd)"
	PYTHONPATH=${BASE_PATH}:$PYTHONPATH
fi

TRAIN_SCRIPT=${BASE_PATH}/TensorFlow/computer_vision/Resnets/resnet_keras/resnet_ctl_imagenet_main.py
PT_VERSION=`python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'`
TF_VERSION=`python3 -c "import tensorflow as tf; print(tf.__version__.replace('.', '_'))"`
PATCH_PATH=/usr/local/lib/python${PT_VERSION}/dist-packages/habana_frameworks/tensorflow/tf${TF_VERSION}/lib/habanalabs
#PRELOAD_PATH=/usr/lib/habanalabs/dynpatch_prf_remote_call.so
# This is required for HW profiling but does not hurt so we add it always
export PYTHONPATH=${PATCH_PATH}:${PYTHONPATH}

# Fixed varaibles, not inherited from launcher
export HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=${HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE:-false} # ART:ON by default
export HABANA_USE_STREAMS_FOR_HCL=true
export HBN_TF_REGISTER_DATASETOPS=1
export TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS=1
export EXPERIMENTAL_PRELOADING=1
export ENABLE_TENSORBOARD=false
export REPORT_ACCURACY_METRICS=true
export DIST_EVAL=true
export ENABLE_DEVICE_WARMUP=true
export TF_DISABLE_MKL=1
export SYNTHETIC_DATA=${SYNTHETIC_DATA}

#HCL variable addition
export SCALE_OUT_PORTS=10
export STRIDED_ALLREDUCE_THRESHOLD_SIZE=1024000000
export MULTI_STREAMS_ENABLE=0
export HOROVOD_HCCL_ALLREDUCE_SLICE_SIZE_KiB=0

if [[ $NUM_WORKERS -ge "${POD_SIZE}" ]]; then
   export POD_SIZE=${POD_SIZE}
else
   export POD_SIZE=$NUM_WORKERS
fi

if [[ $MODELING -eq 1 ]]; then
    ENABLE_CHECKPOINT=true
else
    ENABLE_CHECKPOINT=false
fi
if [[ $TF_ENABLE_BF16_CONVERSION -eq 1 ]]; then
    DATA_TYPE="bf16"
else
    DATA_TYPE="fp32"
fi
if [[  ${NO_EVAL} -eq 1 ]]; then
    SKIP_EVAL=true
else
    SKIP_EVAL=false
fi
if [[ ${USE_LARS_OPTIMIZER} -eq 1 ]]; then
	OPTIMIZER="LARS"
else
	OPTIMIZER="SGD"
fi
if [[ ${USE_HOROVOD} -eq 1 ]]; then
	DIST_EVAL=true
	USE_HOROVOD='--use_horovod'
else
	DIST_EVAL=false
	USE_HOROVOD=''
fi
if [[ ${SYNTHETIC_DATA} -eq 1 ]]; then
	SYNTHETIC_DATA=true
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

#export LOG_LEVEL_HCL=2

TRAIN_COMMAND="python3 ${TRAIN_SCRIPT}
    --model_dir=${WORK_DIR}
    --data_dir=${IMAGENET_DIR}
    --batch_size=${BATCH_SIZE}
    --distribution_strategy=off
    --num_gpus=0
    --data_format=channels_last
    --train_epochs=${TRAIN_EPOCHS}
    --train_steps=${TRAIN_STEPS}
    --experimental_preloading=${EXPERIMENTAL_PRELOADING}
    --log_steps=${DISPLAY_STEPS}
    --steps_per_loop=${STEPS_PER_LOOP}
    --enable_checkpoint_and_export=${ENABLE_CHECKPOINT}
    --enable_tensorboard=${ENABLE_TENSORBOARD}
    --epochs_between_evals=${EPOCHS_BETWEEN_EVALS}
    --base_learning_rate=${BASE_LEARNING_RATE}
    --warmup_epochs=${WARMUP_EPOCHS}
    --optimizer=${OPTIMIZER}
    --lr_schedule=polynomial
    --label_smoothing=${LABEL_SMOOTH}
    --weight_decay=${WEIGHT_DECAY}
    --single_l2_loss_op
    ${USE_HOROVOD}
    ${TF_PROFILE_STEPS}
    --profile=${PROFILE}
    --modeling=${MODELING}
    --data_loader_image_type=${DATA_TYPE}
    --dtype=${DATA_TYPE}
    --eval_offset_epochs=${EVAL_OFFSET_EPOCHS}
    --report_accuracy_metrics=${REPORT_ACCURACY_METRICS}
    --dist_eval=${DIST_EVAL}
    --target_accuracy=${STOP_THRESHOLD}
    --enable_device_warmup=${ENABLE_DEVICE_WARMUP}
    --lars_decay_epochs=${LARS_DECAY_EPOCHS}
    --momentum=${LR_MOMENTUM}
    --skip_eval=${SKIP_EVAL}
    --use_synthetic_data=${SYNTHETIC_DATA}
    --dataset_cache=${DATASET_CACHE}
"
echo ${TRAIN_COMMAND}

echo "[run] General Settings:"
echo "[run] RESNET_SIZE" $RESNET_SIZE
echo "[run] IMAGENET_DIR" $IMAGENET_DIR
echo "[run] BATCH_SIZE"  $BATCH_SIZE
echo "[run] NUM_WORKERS" $NUM_WORKERS
echo "[run] TRAIN_EPOCHS" $TRAIN_EPOCHS
echo "[run] TRAIN_STEPS" $TRAIN_STEPS
echo "[run] DISPLAY_STEPS" $DISPLAY_STEPS
echo "[run] USE_LARS_OPTIMIZER" $USE_LARS_OPTIMIZER
echo "[run] CPU_BIND_TYPE" $CPU_BIND_TYPE
echo "[run] HABANA_PROFILE" $HABANA_PROFILE
echo "[run] EPOCHS_BETWEEN_EVALS" $EPOCHS_BETWEEN_EVALS
echo "[run] TRAIN_AND_EVAL" $TRAIN_AND_EVAL
echo "[run] TF_ENABLE_BF16_CONVERSION" $TF_ENABLE_BF16_CONVERSION
echo "[run] DATASET_CACHE" $DATASET_CACHE
echo "[run] USE_HOROVOD" $USE_HOROVOD
echo "[run] TF_PROFILE_STEPS" $TF_PROFILE_STEPS
echo "[run] HW_PROFILE_RANGE" $HW_PROFILE_RANGE
echo "[run] PROFILE_CONFIG" $PROFILE_CONFIG
echo "[run] SYN_PROFILE" $SYN_PROFILE
echo
echo "[run] Learning Setting:"
echo "[run] WEIGHT_DECAY" $WEIGHT_DECAY
echo "[run] LABEL_SMOOTH" $LABEL_SMOOTH
echo "[run] BASE_LEARNING_RATE" $BASE_LEARNING_RATE
echo "[run] WARMUP_EPOCHS" $WARMUP_EPOCHS
echo "[run] USE_MLPERF" $USE_MLPERF
echo "[run] NO_EVAL" $NO_EVAL
echo "[run] STOP_THRESHOLD" $STOP_THRESHOLD
echo "[run] LR_MOMENTUM" $LR_MOMENTUM
echo "[run] EVAL_OFFSET_EPOCHS" $EVAL_OFFSET_EPOCHS
echo "[run] LARS_DECAY_EPOCHS" $LARS_DECAY_EPOCHS
echo "[run] SYNTHETIC_DATA" $SYNTHETIC_DATA

#if [ ! -f /usr/lib/habanalabs/dynpatch_prf_remote_call.so ]; then
#	ln -s ${PATCH_PATH}/dynpatch_prf_remote_call.so /usr/lib/habanalabs/dynpatch_prf_remote_call.so
#fi

LOCAL_SNC_VALUE=$(( OMPI_COMM_WORLD_LOCAL_RANK % 4))
if [[ $OMPI_COMM_WORLD_LOCAL_RANK -eq 0 ]]; then
	if [[ $DEBUG -eq 1 ]]; then
	      echo "[run] Dropping Caches"
	      echo "$(hostname) $(ulimit -a)"
	fi
	sync
	echo 3 > /proc/sys/vm/drop_caches

	# Set Synapse logger configuration only for one Gaudi
	if [[ ${SYN_PROFILE} -eq 1 ]]; then
      		export HBN_SYNAPSE_LOGGER_COMMANDS=no_eager_flush:stop_data_capture:use_pid_suffix
                export PRELOAD_PATH=${PATCH_PATH}/synapse_logger.so:${PRELOAD_PATH}
	fi
	# Set merged (host+Gaudi) HW trace configuration for one Gaudi only
	if [[ -n ${HW_PROFILE_RANGE} ]]; then
		export HABANA_PROFILE=1
		export HABANA_SYNAPSE_LOGGER=range_hw
		export TF_HOOK_MODE=all
		export TF_HW_PROFILE_RANGE=${HW_PROFILE_RANGE}
		export HABANA_PROFILE_CONFIG_PATH=${PROFILE_CONFIG}
		echo "HABANA_PROFILE_CONFIG_PATH=${HABANA_PROFILE_CONFIG_PATH}"
		export LOG_LEVEL_HwTrace=1 LOG_LEVEL_PROF_hl0=1 LOG_LEVEL_ALL=5
	fi

fi
if [[ $CPU_BIND_TYPE == "numa" ]]; then
	if [[ $LOCAL_SNC_VALUE -eq 0 ]]; then
        	CPU_RANGE="0,1,2,3,7,8,9,14,15,16,17,22,23"
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
