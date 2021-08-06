#!/bin/bash

# Default setting
export TF_ENABLE_BF16_CONVERSION=1
export TF_ALLOW_CONTROL_EDGES_IN_HABANA_OPS=1
export TF_PRELIMINARY_CLUSTER_SIZE=200

# Training hparams
export RESNET_SIZE=${RESNET_SIZE:-50}
export BATCH_SIZE=256
export TRAIN_EPOCHS=2
export DISPLAY_STEPS=100
export SAVE_CHECKPOINT_STEPS=3000

# Script params
export RECOVER=0
export HABANA_SYNAPSE_LOGGER=RANGE

__NUM_WORKERS=(2 4 8)
__NUM_CORES_PER_RANK=(4)

if [[ ! -d ${IMAGENET_DIR} ]]; then
    echo "ImageNet image database not found"
    exit -1
fi

DEMODIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
export PYTHONPATH=$(dirname $DEMODIR)/:$PYTHONPATH

WORKDIR=/tmp/${USER}/resnet$RESNET_SIZE

mkdir -p $WORKDIR
pushd $WORKDIR

MODELDIR=/tmp/${USER}/resnet$RESNET_SIZE/model
mkdir -p $MODELDIR

source $DEMODIR/../../common/common.sh
setup_preloading

if [ $TF_ENABLE_BF16_CONVERSION -eq 1 ]; then
  DATA_LOADER_TYPE="--data_loader_image_type=bf16"
fi

printf "*** Starting training...\n\n"

for num_workers in "${__NUM_WORKERS[@]}"
do
    for num_cores in "${__NUM_CORES_PER_RANK[@]}"
    do
        printf "*** Cleaning temp files...\n\n"
        rm -rf "$MODELDIR/*"
        generate_hcl_config ${WORKDIR} ${num_workers}
        if [ $RECOVER -eq 1 ];
        then
            while true; do
                mpirun -np $num_workers --bind-to core --map-by slot:PE=$num_cores python3 $DEMODIR/imagenet_main.py \
                    --num_gpus=1 \
                    --data_dir $IMAGENET_DIR \
                    --model_dir $MODELDIR \
                    --distribution_strategy=off \
                    --data_format=channels_last \
                    --resnet_size=$RESNET_SIZE \
                    --batch_size=$BATCH_SIZE \
                    --display_steps=$DISPLAY_STEPS \
                    --use_horovod \
                    --use_synthetic_data=false \
                    --experimental_preloading=1 \
                    --save_checkpoint_steps=$SAVE_CHECKPOINT_STEPS \
                    --train_epochs=$TRAIN_EPOCHS \
                    $DATA_LOADER_TYPE

                if [ $? -eq 0 ]; then
                    break
                else
                    sleep 5
                    echo "*** Recovering training...\n\n"
                fi
            done
        else
            mpirun -np $num_workers --bind-to core --map-by slot:PE=$num_cores python3 $DEMODIR/imagenet_main.py \
                --num_gpus=1 \
                --data_dir $IMAGENET_DIR \
                --model_dir $MODELDIR \
                --distribution_strategy=off \
                --data_format=channels_last \
                --resnet_size=$RESNET_SIZE \
                --batch_size=$BATCH_SIZE \
                --display_steps=$DISPLAY_STEPS \
                --use_horovod \
                --use_synthetic_data=false \
                --experimental_preloading=1 \
                --save_checkpoint_steps=$SAVE_CHECKPOINT_STEPS \
                --train_epochs=$TRAIN_EPOCHS \
                $DATA_LOADER_TYPE
        fi
        if [ "$HABANA_SYNAPSE_LOGGER" = "RANGE" ];
        then
            echo "Collecting synapse traces...\n"
            tar -czvf syn_logs_${num_workers}_workers_${num_cores}_cores.tar.gz *.json
            LOG_FILES=worker*.json
            python ${TF_MODULES_ROOT}/synapse_logger/tools/merge_tf_and_logger.py -s $LOG_FILES
            mv merged.json.gz merged_syn_logs_${num_workers}_workers_${num_cores}_cores.json.gz
            rm *.json
            rm *.data
            sleep 10
        fi
    done
done
popd
