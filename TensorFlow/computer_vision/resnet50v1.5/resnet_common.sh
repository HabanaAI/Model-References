export LOG_LEVEL_ALL=6
DEMODIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
source $DEMODIR/../../common/common.sh

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

if [ $TF_ENABLE_BF16_CONVERSION -eq 1 ]; then
  TRAIN_PARAMS="${TRAIN_PARAMS} --data_loader_image_type=bf16"
  EVAL_PARAMS="${EVAL_PARAMS} --data_loader_image_type=bf16"
fi

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
    --enable_lars=${ENABLE_LARS_OPTIMIZER}
    ${TRAIN_PARAMS}"

printf "*** Starting training...\n\n"

$TRAINING_COMMAND

if [ $RECOVER_TRAINING -eq 1 ]; then
  while [ ! -f $MODEL_DIR/model.ckpt-$STEPS.meta ]
  do
    if [ ${USE_HOROVOD} == "true" ]; then
      echo "Sleep 120s"
      sleep 120s
    fi
    $TRAINING_COMMAND
  done
fi

if [ $NO_EVAL -eq 0 ]; then
  printf "*** Starting evaluation...\n\n"
  python3 imagenet_main.py \
    --num_gpus=1 \
    --data_dir $DATA_DIR \
    --distribution_strategy=off \
    --data_format=channels_last \
    --batch_size=32 \
    --resnet_size=${RESNET_SIZE} \
    --experimental_preloading=${EXPERIMENTAL_PRELOADING} \
    --eval_only \
    --use_horovod=$USE_HOROVOD \
    --model_dir=$MODEL_DIR \
    $EVAL_PARAMS
fi
