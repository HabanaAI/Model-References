export LOG_LEVEL_ALL=6

if [[ $USE_MLPERF -eq 1 ]]; then
    export TF_CPP_MIN_LOG_LEVEL=6
    export TF_CPP_MIN_VLOG_LEVEL=0
fi

DEMODIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
source $DEMODIR/../../common/common.sh

setup_libjemalloc

if [ -z $DATA_DIR ] || [ -z $BATCH_SIZE ] || [ -z $EPOCHS ]; then
    echo -e "Missing necessary variables"
    exit 1;
fi

if [ $CHECKPOINT_STEPS -eq 0 ] && [ $NO_EVAL -ne 1 ]; then
    echo -e "--checkpoint-steps 0 is allowed only together with --no-eval flag"
    echo -e "If you want to minimize number of checkpoints during training"
    echo -e "provide big number, for example --checkpoint-steps 2147483647"
    exit 1;
fi

if [ -z $MODEL_DIR ]; then
    MODEL_DIR="$HOME/tmp/resnet/"
    printf "*** Cleaning temp files in ${MODEL_DIR}...\n\n"
    rm -rf "$MODEL_DIR"
fi

if [ $STEPS -ne -1 ]; then
  TRAIN_PARAMS="${TRAIN_PARAMS} --max_train_steps=${STEPS}"
fi

if [ $EVAL_STEPS -ne -1 ]; then
  TRAIN_PARAMS="${TRAIN_PARAMS} --max_eval_steps=${EVAL_STEPS}"
  EVAL_PARAMS="${EVAL_PARAMS} --max_eval_steps=${EVAL_STEPS}"
fi

if [ $RECOVER_TRAINING -eq 1 ] && [ $STEPS -eq -1 ]; then
    printf "When training recovery is enabled need to set exact steps count"
    exit 1;
fi

# Disabling bf16 data loading for ResNeXt is a workaround that improves performance
# TODO SW-32809: Investigate why some variables were executed on bf16 in ResNeXt and check if
#                registering them for bf16 execution on HPU is feasible
if [ $TF_ENABLE_BF16_CONVERSION -eq 1 ] && [ $RUN_RESNEXT == "false" ]; then
  TRAIN_PARAMS="${TRAIN_PARAMS} --data_loader_image_type=bf16"
  EVAL_PARAMS="${EVAL_PARAMS} --data_loader_image_type=bf16"
fi

if [ $ENABLE_LARS_OPTIMIZER == "true" ] && [ ${USE_HOROVOD} == "false" ]; then
  TRAIN_PARAMS="${TRAIN_PARAMS}
    --weight_decay=0.0001
    --label_smoothing=0.1
    --epochs_for_lars=${EPOCHS:-40}
    --start_learning_rate=2.5
    --warmup_epochs=3
    --train_offset=0
    --is_mlperf_enabled=false"
fi

if [ $ENABLE_LARS_OPTIMIZER == "true" ] && [ ${USE_HOROVOD} == "true" ]; then
  if [[ $USE_MLPERF -eq 1 ]]; then
    __epochs_for_lars=$EPOCHS
    __train_offset=2
    EPOCHS=$((EPOCHS - __train_offset))

    TRAIN_PARAMS="${TRAIN_PARAMS}
      --weight_decay=0.0001
      --label_smoothing=0.1
      --epochs_for_lars=$__epochs_for_lars
      --start_learning_rate=9.5
      --warmup_epochs=2
      --train_offset=$__train_offset
      --is_mlperf_enabled=true"
  else
    TRAIN_PARAMS="${TRAIN_PARAMS}
      --weight_decay=${WEIGHT_DECAY:-0.0001}
      --label_smoothing=0.1
      --epochs_for_lars=${EPOCHS:-40}
      --start_learning_rate=${START_LEARNING_RATE:-9.5}
      --warmup_epochs=${WARMUP_EPOCHS:-2}
      --train_offset=0
      --is_mlperf_enabled=false"
  fi
fi

if [[ -n ${STOP_THRESHOLD} ]]; then
  TRAIN_PARAMS="${TRAIN_PARAMS} --stop_threshold=${STOP_THRESHOLD}"
fi

if [ $NO_EVAL -eq 1 ]; then
  TRAIN_PARAMS="${TRAIN_PARAMS} --disable_eval=true"
fi

if [ $RUN_RESNEXT == "true" ]; then
  TRAIN_PARAMS="${TRAIN_PARAMS}
    --model_type=resnext
    --weight_decay=6.103515625e-05
    --momentum=0.875
    --label_smoothing=0.1
    --warmup_epochs=8
    --use_cosine_lr"
fi

# Disabling MKL for tensoflow flavors that were compiled with it enabled.
# This is done to disable _FusedBatchNormEx optimization that is not supported on HPU. For details refer to SW-34564.
export TF_DISABLE_MKL=1

TRAINING_COMMAND="python3 imagenet_main.py
    --num_gpus=1
    --data_dir ${DATA_DIR}
    --distribution_strategy=off
    --data_format=channels_last
    --batch_size=${BATCH_SIZE}
    --resnet_size=${RESNET_SIZE}
    --save_checkpoint_steps=${CHECKPOINT_STEPS}
    --train_epochs=${EPOCHS}
    --model_dir=${MODEL_DIR}
    --display_steps=${DISPLAY_STEPS}
    --experimental_preloading=${EXPERIMENTAL_PRELOADING}
    --use_horovod=${USE_HOROVOD}
    --use_train_and_evaluate=${USE_TRAIN_AND_EVALUATE}
    --epochs_between_evals=${EPOCHS_BETWEEN_EVALS}
    --dummy_epoch=${DUMMY_EPOCH}
    ${TRAIN_PARAMS}"

printf "*** Starting training...\n\n"

LD_PRELOAD=${LD_PRELOAD}:${LD_PRELOAD_PYTH} $TRAINING_COMMAND

if [ $RECOVER_TRAINING -eq 1 ]; then
  while [ ! -f $MODEL_DIR/model.ckpt-$STEPS.meta ]
  do
    if [ ${USE_HOROVOD} == "true" ]; then
      echo "Sleep 120s"
      sleep 120s
    fi
    LD_PRELOAD=${LD_PRELOAD}:${LD_PRELOAD_PYTH} $TRAINING_COMMAND
  done
fi
