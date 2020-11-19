#!/bin/bash

# Default setting
export HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=true
export TF_ENABLE_BF16_CONVERSION=1
export TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS=1
export HABANA_USE_STREAMS_FOR_HCL=true
export TF_PRELIMINARY_CLUSTER_SIZE=200

# Training hparams
export RESNET_SIZE=${RESNET_SIZE:-50}
export BATCH_SIZE=256
export TRAIN_EPOCHS=90
export DISPLAY_STEPS=100
export SAVE_CHECKPOINT_STEPS=5005

export NUM_WORKERS=8
export RECOVER=0
export MPI_PE=${MPI_PE:-4}

if [[ ! -d ${IMAGENET_DIR} ]]; then
    echo "ImageNet image database not found"
    exit -1
fi

DEMODIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
export PYTHONPATH=$(dirname $DEMODIR)/:$PYTHONPATH

WORKDIR=/tmp/${USER}/resnet$RESNET_SIZE

mkdir -p $WORKDIR
pushd $WORKDIR

printf "*** Cleaning temp files...\n\n"
rm -rf /tmp/checkpoint /tmp/eval /tmp/events.out.tfevents.* /tmp/graph.pbtxt /tmp/model.ckpt-*
rm -rf /tmp/rank_*/checkpoint /tmp/rank_*/eval /tmp/rank_*/events.out.tfevents.* /tmp/rank_*/graph.pbtxt /tmp/rank_*/model.ckpt-*

source $DEMODIR/../common.sh
setup_preloading

if [[ ${USE_LARS_OPTIMIZER} -eq 1 ]]; then
    export USE_LARS_OPTIMIZER="--enable-lars-optimizer"
else
    export USE_LARS_OPTIMIZER=""
fi

if [[ -z ${MULTI_HLS_IPS} ]]; then
    generate_hcl_config ${WORKDIR} ${NUM_WORKERS}
    printf "*** Starting training...\n\n"
    if [ $RECOVER -eq 1 ];then
        while true;
        do
            mpirun -np $NUM_WORKERS --bind-to core --map-by socket:PE=$MPI_PE --merge-stderr-to-stdout --output-filename $DEMODIR/log $DEMODIR/demo_resnet \
            --dtype bf16 \
            --data-dir $IMAGENET_DIR \
            --resnet-size $RESNET_SIZE \
            --batch-size $BATCH_SIZE \
            --display-steps $DISPLAY_STEPS \
            --use_horovod \
            --epochs-between-evals $NUM_WORKERS \
            --checkpoint-steps $SAVE_CHECKPOINT_STEPS \
            --no-eval `# no_eval means disabling additional evaluation after final evaluation from training - debug feature` \
            --epochs $TRAIN_EPOCHS \
            $USE_LARS_OPTIMIZER

            if [ $? -eq 0 ]; then
                break
            else
                sleep 5
                echo "*** Recovering training...\n\n"
            fi
        done
    else
       mpirun -np $NUM_WORKERS --bind-to core --map-by socket:PE=$MPI_PE --merge-stderr-to-stdout --output-filename $DEMODIR/log $DEMODIR/demo_resnet \
            --dtype bf16 \
            --data-dir $IMAGENET_DIR \
            --resnet-size $RESNET_SIZE \
            --batch-size $BATCH_SIZE \
            --display-steps $DISPLAY_STEPS \
            --use_horovod \
            --epochs-between-evals $NUM_WORKERS \
            --checkpoint-steps $SAVE_CHECKPOINT_STEPS \
            --no-eval `# no_eval means disabling additional evaluation after final evaluation from training - debug feature` \
            --epochs $TRAIN_EPOCHS \
            $USE_LARS_OPTIMIZER
    fi

else
    if [[ -z ${HCL_CONFIG_PATH} ]]; then
        echo "Please set HCL_CONFIG_PATH and make sure it is in the same location on all HLSs."
        exit -1
    fi
    IFS=',' read -ra IPS <<< "$MULTI_HLS_IPS"
    let MPI_NP=${#IPS[@]}*8

    generate_mpi_hostfile ${WORKDIR}

    mpirun --mca plm_rsh_args "-p 3022" \
        --mca btl_tcp_if_exclude lo,docker0 \
        -np $MPI_NP -hostfile ${MPI_HOSTFILE_PATH}\
        --bind-to core --map-by socket:PE=$MPI_PE \
        --prefix /usr/lib/habanalabs/openmpi/ \
        --merge-stderr-to-stdout --output-filename $DEMODIR/log \
        -x HCL_CONFIG_PATH=${HCL_CONFIG_PATH} \
        -x IMAGENET_DIR=${IMAGENET_DIR} \
        -x HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=${HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE} \
        -x TF_ENABLE_BF16_CONVERSION=${TF_ENABLE_BF16_CONVERSION} \
        -x TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS=${TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS} \
        -x HBN_TF_REGISTER_DATASETOPS=${HBN_TF_REGISTER_DATASETOPS} \
        -x HABANA_USE_STREAMS_FOR_HCL=${HABANA_USE_STREAMS_FOR_HCL} \
        -x TF_PRELIMINARY_CLUSTER_SIZE=${TF_PRELIMINARY_CLUSTER_SIZE} \
        -x LD_PRELOAD=${LD_PRELOAD} \
        -x TF_MODULES_RELEASE_BUILD=${TF_MODULES_RELEASE_BUILD} \
        -x PYTHONPATH=${PYTHONPATH} \
        -x GC_KERNEL_PATH=${GC_KERNEL_PATH} \
        -x HABANA_LOGS=${HABANA_LOGS} \
        $DEMODIR/demo_resnet \
            --dtype bf16 \
            --data-dir $IMAGENET_DIR \
            --batch-size $BATCH_SIZE \
            --resnet-size $RESNET_SIZE \
            --display-steps $DISPLAY_STEPS \
            --use_horovod \
            --epochs-between-evals $MPI_NP \
            --checkpoint-steps $SAVE_CHECKPOINT_STEPS \
            --no-eval `# no_eval means disabling additional evaluation after final evaluation from training - debug feature` \
            --epochs $TRAIN_EPOCHS \
            $USE_LARS_OPTIMIZER

fi

popd
