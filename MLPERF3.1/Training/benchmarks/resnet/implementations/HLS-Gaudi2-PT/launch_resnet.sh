#!/bin/bash

function print_synopsis()
{
    cat << EOF
NAME
        `basename $0`

SYNOPSIS
        `basename $0` [-c <config>] [-ld <log-dir>] [-wd <work-dir>] [-dd <data-dir>] [-h]

DESCRIPTION
        Runs 8-gaudi local MLPerf Resnet training on PyTorch.

        -c <config-file>, --config <config-file>
            configuration file containing series of "export VAR_NAME=value" commands
            overrides default settings for Resnet training

        -ld <log-dir>, --log-dir <log-dir>
            specify the loggin directory, used to store mllogs and outputs from all mpi processes

        -wd <work-dir>, --work-dir <work-dir>
            specify the work directory, used to store temporary files during the training

        -dd <data-dir>
            specify the data directory, containing the ImageNet dataset

        -ut <bool>, --use-torch-compile <bool>
            turn on the torch compile, default is false

        -h, --help
            print this help message

EXAMPLES
       `basename $0` -wd /data/imagenet
            MLPerf Resnet training on dataset stored in /data/imagenet

EOF
}

function parse_config()
{
    while [ -n "$1" ]; do
        case "$1" in
            -c | --config )
                CONFIG_FILE=$2
                if [[ -f ${CONFIG_FILE} ]]; then
	                source $CONFIG_FILE
                    return
                else
                    echo "Could not find ${CONFIG_FILE}"
                    exit 1
                fi
                ;;
            * )
                shift
                ;;
        esac
    done
}

function parse_args()
{
    while [ -n "$1" ]; do
        case "$1" in
            -c | --config )
                shift 2
                ;;
            -ld | --log-dir )
                LOG_DIR=$2
                shift 2
                ;;
            -wd | --work-dir )
                WORK_DIR=$2
                shift 2
                ;;
            -dd | --data-dir )
                DATA_DIR=$2
                shift 2
                ;;
            -ut | --use-torch-compile )
                USE_TORCH_COMPILE=$2
                shift 2
                ;;
            -h | --help )
                print_synopsis
                exit 0
                ;;
            * )
                echo "error: invalid parameter: $1"
                print_synopsis
                exit 1
                ;;
        esac
    done
}

# Default setting for Pytorch Resnet trainig

NUM_WORKERS_PER_HLS=8
EVAL_OFFSET_EPOCHS=3
EPOCHS_BETWEEN_EVALS=4
DISPLAY_STEPS=1000

NUM_WORKERS=8
BATCH_SIZE=256
TRAIN_EPOCHS=35
LARS_DECAY_EPOCHS=36
WARMUP_EPOCHS=3
BASE_LEARNING_RATE=9
END_LEARNING_RATE=0.0001
WEIGHT_DECAY=0.00005
LR_MOMENTUM=0.9
LABEL_SMOOTH=0.1
STOP_THRESHOLD=0.759
USE_TORCH_COMPILE=false

DATA_DIR=/mnt/weka/data/pytorch/imagenet/ILSVRC2012/

WORK_DIR=/tmp/resnet50
LOG_DIR=/tmp/resnet_log
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

# Default MPI settings
MPI_HOSTS=localhost:8
MPI_OUTPUT=/tmp/resnet_log
MPI_PATH=/opt/amazon/openmpi
SSH_PORT=3022

# MASTER_ADDR and MASTER_PORT are consumed by PyTorch c10d to establish a distributed group
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-12345}

# apply optional config, overwriting default settings
parse_config "$@"

# optional command line arguments overwrite both default and config settings
parse_args "$@"

# Use torch compile
if [ "$USE_TORCH_COMPILE" == "true" ]; then
    echo "torch.compile enabled"
    TOCH_COMPILE_FLAGS="--use_torch_compile --run-lazy-mode false"
else
    TORCH_COMPILE_FLAGS=""
fi

# Clear caches
PROC_FS=${PROC_FS:-"/proc"}
sync && echo 3 > $PROC_FS/sys/vm/drop_caches

# determine the number of available cores for each process
MPI_MAP_BY_PE=`lscpu | grep "^CPU(s):"| awk -v NUM=${NUM_WORKERS_PER_HLS} '{print int($2/NUM/2)}'`

# prepare directories
rm -rf $LOG_DIR
mkdir -p $WORK_DIR
mkdir -p $LOG_DIR

# run Pytorch Resnet training
mpirun \
--allow-run-as-root \
--np $NUM_WORKERS \
--bind-to core \
--rank-by core \
--map-by socket:PE=$MPI_MAP_BY_PE \
-H $MPI_HOSTS \
--report-bindings \
--tag-output \
--merge-stderr-to-stdout \
--output-filename $LOG_DIR \
--prefix $MPI_PATH \
-x PT_HPU_AUTOCAST_LOWER_PRECISION_OPS_LIST=$SCRIPT_DIR/PyTorch/ops_bf16_Resnet.txt \
-x PT_HPU_AUTOCAST_FP32_OPS_LIST=$SCRIPT_DIR/PyTorch/ops_fp32_Resnet.txt \
python3 $SCRIPT_DIR/PyTorch/train.py \
--model resnet50 \
--device hpu \
--print-freq $DISPLAY_STEPS \
--channels-last False \
--dl-time-exclude False \
--output-dir $WORK_DIR \
--log-dir $LOG_DIR \
--data-path $DATA_DIR \
--eval_offset_epochs $EVAL_OFFSET_EPOCHS \
--epochs_between_evals $EPOCHS_BETWEEN_EVALS \
--workers $NUM_WORKERS_PER_HLS \
--batch-size $BATCH_SIZE \
--epochs $TRAIN_EPOCHS \
--lars_decay_epochs $LARS_DECAY_EPOCHS \
--warmup_epochs $WARMUP_EPOCHS \
--base_learning_rate $BASE_LEARNING_RATE \
--end_learning_rate $END_LEARNING_RATE \
--weight-decay $WEIGHT_DECAY \
--momentum $LR_MOMENTUM \
--label-smoothing $LABEL_SMOOTH \
--target_accuracy $STOP_THRESHOLD \
--use_autocast \
 $TOCH_COMPILE_FLAGS \
--dl-worker-type HABANA

# finalize LOG_DIR folder
chmod -R 777 ${LOG_DIR}
exit 0
