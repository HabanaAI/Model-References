#!/bin/bash

export LOG_LEVEL_ALL=6
DEMODIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
source $DEMODIR/../../../common/common.sh
setup_libjemalloc

if [ -z $DATA_DIR ] || [ -z $BATCH_SIZE ] || [ -z $EPOCHS ]; then
    echo -e "Missing necessary variables"
    exit 1;
fi

if [ -z $MODEL_DIR ]; then
    MODEL_DIR="$HOME/tmp/resnet/"
    printf "*** Cleaning temp files in ${MODEL_DIR}...\n\n"
    rm -rf "$MODEL_DIR"
fi

if [ $STEPS -ne -1 ]; then
  TRAIN_PARAMS="${TRAIN_PARAMS} --train_steps=${STEPS}"
fi

if [ $RECOVER_TRAINING -eq 1 ] && [ $STEPS -eq -1 ]; then
    printf "When training recovery is enabled need to set exact steps count"
    exit 1;
fi

if [ $TF_ENABLE_BF16_CONVERSION -eq 1 ]; then
  TRAIN_PARAMS="${TRAIN_PARAMS} --data_loader_image_type=bf16"
  EVAL_PARAMS="${EVAL_PARAMS} --data_loader_image_type=bf16"
fi

if [  ${NO_EVAL} -eq 1 ]; then
    TRAIN_PARAMS="${TRAIN_PARAMS} --skip_eval=true"
fi

if [ ${USE_HOROVOD} == "true" ]; then
    TRAIN_PARAMS="${TRAIN_PARAMS} --use_horovod"
fi

if [ ${USE_HPU_STRATEGY} == "true" ]; then
    TRAIN_PARAMS="${TRAIN_PARAMS} --distribution_strategy=hpu --use_tf_while_loop=False"
else
    TRAIN_PARAMS="${TRAIN_PARAMS} --distribution_strategy=off"
fi

# Disabling MKL for tensoflow flavors that were compiled with it enabled.
# This is done to disable _FusedBatchNormEx optimization that is not supported on HPU. For details refer to SW-34564.
export TF_DISABLE_MKL=1

if [[ ${USE_LARS_OPTIMIZER} -eq 1 || ${USE_LARS_OPTIMIZER} == "true" ]]; then
    TRAIN_PARAMS="${TRAIN_PARAMS} --base_learning_rate=9.5"
    TRAIN_PARAMS="${TRAIN_PARAMS} --warmup_epochs=3"
    TRAIN_PARAMS="${TRAIN_PARAMS} --optimizer=LARS"
    TRAIN_PARAMS="${TRAIN_PARAMS} --lr_schedule=polynomial"
    TRAIN_PARAMS="${TRAIN_PARAMS} --label_smoothing=0.1"
    TRAIN_PARAMS="${TRAIN_PARAMS} --weight_decay=0.0001"
    TRAIN_PARAMS="${TRAIN_PARAMS} --single_l2_loss_op"
fi

EXPERIMENTAL_PRELOADING=${EXPERIMENTAL_PRELOADING:-1}

TRAINING_COMMAND="python3 resnet_ctl_imagenet_main.py
    --model_dir=${MODEL_DIR}
    --data_dir=${DATA_DIR}
    --batch_size=${BATCH_SIZE}
    --num_gpus=0
    --data_format=channels_last
    --train_epochs=${EPOCHS}
    --experimental_preloading=${EXPERIMENTAL_PRELOADING}
    --log_steps=${DISPLAY_STEPS}
    --steps_per_loop=${STEPS_PER_LOOP}
    --enable_checkpoint_and_export=${ENABLE_CHECKPOINT}
    --enable_tensorboard=${ENABLE_TENSORBOARD}
    --epochs_between_evals=${EPOCHS_BETWEEN_EVALS}
    ${TRAIN_PARAMS}"

printf "*** Starting training...\n\n"

$TRAINING_COMMAND


if [ $RECOVER_TRAINING -eq 1 ]; then
  while [ ! -f $MODEL_DIR/ckpt-$STEPS.index ]
  do
    if [ ${USE_HOROVOD} == "true" ]; then
      echo "Sleep 120s"
      sleep 120s
    fi
    $TRAINING_COMMAND
  done
fi
