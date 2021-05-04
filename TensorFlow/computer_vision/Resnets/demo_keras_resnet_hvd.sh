#!/bin/bash

function usage()
{
        echo -e "usage: IMAGENET_DIR=<path/to/imagenet/database> demo_keras_resnet_hvd.sh\n"
        echo -e "\nexample:\n"
        echo -e "  IMAGENET_DIR=~/tensorflow_datasets/imagenet/tf_records/ ./demo_keras_resnet_hvd.sh"
        echo -e ""
}

# Default setting
export HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=${HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE:-false} # ART:ON by default
export TF_ENABLE_BF16_CONVERSION=1
export TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS=1
export HABANA_USE_STREAMS_FOR_HCL=true
export TF_PRELIMINARY_CLUSTER_SIZE=200

# Training hparams
export RESNET_SIZE=${RESNET_SIZE:-50}
export USE_LARS_OPTIMIZER=${USE_LARS_OPTIMIZER:-0}

export BATCH_SIZE=${BATCH_SIZE:-256}
export TRAIN_STEPS=${TRAIN_STEPS:--1}
export NO_EVAL=${NO_EVAL:-0}
export ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-0}
export DISPLAY_STEPS=100
export STEPS_PER_LOOP=100

export NUM_WORKERS_PER_HLS=${NUM_WORKERS_PER_HLS:-8}
export HLS_TYPE=${HLS_TYPE:-HLS1}
export RECOVER=0

exit_code=0

if [[ $USE_MLPERF -eq 1 ]]; then
    export STOP_THRESHOLD=${STOP_THRESHOLD:-0.759}

    export BATCH_SIZE=256
    export NUM_WORKERS_PER_HLS=8

    export NUM_TRAIN_EPOCHS_BETWEEN_EVAL=4
    export USE_LARS_OPTIMIZER=1
else
    export NUM_TRAIN_EPOCHS_BETWEEN_EVAL=$NUM_WORKERS_PER_HLS
fi

if [[ $ENABLE_CHECKPOINT -eq 1 || $ENABLE_CHECKPOINT == "true" ]]; then
    export ENABLE_CHECKPOINT="--enable-checkpoint"
else
    export ENABLE_CHECKPOINT=""
fi

if [[ $USE_LARS_OPTIMIZER -eq 1 || $USE_LARS_OPTIMIZER == "true" ]]; then
    export TRAIN_EPOCHS=${TRAIN_EPOCHS:-40}
else
    export TRAIN_EPOCHS=${TRAIN_EPOCHS:-90}
fi

if [[ ! -d ${IMAGENET_DIR} ]]; then
    echo "ImageNet image database not found"
    exit -1
fi

DEMODIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
export PYTHONPATH=$(dirname $DEMODIR)/:$PYTHONPATH

if [ -n "$__resnext" ]; then
    WORKDIR=/tmp/${USER}/resnext$RESNET_SIZE
else
    WORKDIR=/tmp/${USER}/resnet$RESNET_SIZE
fi

source $DEMODIR/../../common/common.sh

calc_optimal_cpu_resources_for_mpi

run_per_ip mkdir -p $WORKDIR
pushd $WORKDIR

export TF_RECIPE_CACHE_PATH=${TF_RECIPE_CACHE_PATH:-$WORKDIR}

printf "*** Cleaning temp files...\n\n"
run_per_ip rm -rf /tmp/checkpoint /tmp/eval /tmp/events.out.tfevents.* /tmp/graph.pbtxt /tmp/model.ckpt-*
run_per_ip rm -rf /tmp/rank_*/checkpoint /tmp/rank_*/eval /tmp/rank_*/events.out.tfevents.* /tmp/rank_*/graph.pbtxt /tmp/rank_*/model.ckpt-*

setup_preloading

if [[ ${NO_EVAL} -eq 1 ]]; then
    export NO_EVAL="--no-eval"
else
    export NO_EVAL=""
fi

if [[ -n ${STOP_THRESHOLD} ]]; then
    export STOP_THRESHOLD="--stop_threshold ${STOP_THRESHOLD}"
fi

if [[ -z ${MULTI_HLS_IPS} ]]; then
    generate_hcl_config ${WORKDIR} ${NUM_WORKERS_PER_HLS} ${HLS_TYPE}

    TRAINING_COMMAND="mpirun --allow-run-as-root \
        -np $NUM_WORKERS_PER_HLS \
        ${MPIRUN_ARGS_MAP_BY_PE} \
        --merge-stderr-to-stdout --output-filename $DEMODIR/log \
        $DEMODIR/demo_keras_resnet \
            --dtype bf16 \
            --data-dir $IMAGENET_DIR \
            --batch-size $BATCH_SIZE \
            --steps-per-loop $STEPS_PER_LOOP \
            --use_horovod \
            --epochs-between-evals $NUM_TRAIN_EPOCHS_BETWEEN_EVAL \
            --epochs $TRAIN_EPOCHS \
            --steps $TRAIN_STEPS \
            $ENABLE_CHECKPOINT \
            $STOP_THRESHOLD \
            $NO_EVAL"

    printf "*** Starting training...\n\n"
    if [ $RECOVER -eq 1 ];then
        while true;
        do
            $TRAINING_COMMAND

            if [ $? -eq 0 ]; then
                break
            else
                sleep 5
                echo "*** Recovering training...\n\n"
            fi
        done
    else
        $TRAINING_COMMAND
        exit_code=$?
    fi
fi

popd
exit $exit_code
