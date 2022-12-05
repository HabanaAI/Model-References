#!/bin/bash

function print_synopsis()
{
    cat << EOF
NAME
        `basename $0`

SYNOPSIS
        `basename $0` [-H <hosts>] [-p <sshport>] [-id <input-dir>] [-p1 <phase1-ckpt>] [-ed <eval-dir>] [-od <output-dir>]

DESCRIPTION
        Runs MLPerf BERT pre-training training and evaluation on PyTorch with optional Habana profiling runs.

        -H, --hosts
            comma-separated list of workers' hostnames or IP addresses
            default: localhost:8

        -p, --ssh-port
            socket port number used by mpirun to establish a inter-process communication for multi-box
            default: 3022

        -id, --input-dir
            specify the training data directory, containing books-wiki packed dataset.
            default: /mnt/weka/data/pytorch/bert_mlperf/packed_data/packed

        -p1, --phase1-ckpt
            specify the phase 1 ckpt path.
            default: /mnt/weka/data/pytorch/bert_mlperf/packed_data/phase1/model.ckpt-28252.pt

        -ed, --eval-dir
            specify the evaluation data directory.
            default: /mnt/weka/data/pytorch/bert_mlperf/packed_data/hdf5/eval_varlength/

        -od, --output-dir
            specify the output directory, used to store training results.
            default: /tmp/BERT_PRETRAINING

        -h, --help
            prints this help message.

EXAMPLES
       `basename $0`                                                 # 8-Gaudi local run
       `basename $0` -dd /mnt/data/books_wiki_packed -od /tmp/output # 8-Gaudi local run, overriding data-dir and output-dir
       `basename $0` -ep true                                        # 8-Gaudi local run, with profiling enabled
       `basename $0` -H 10.111.131.28:8,10.111.131.27:8              # 16-Gaudi multi-box run
EOF
}

function parse_args()
{
    REMOTE_SSH_PORT='3022'
    REMOTE_HOSTS="localhost:8"

    while true; do
        case "$1" in
            -h | --help )
                print_synopsis
                exit 0 ;;
            -H | --hosts )
                REMOTE_HOSTS="$2"
                shift 2 ;;
            -p | --ssh-port )
                REMOTE_SSH_PORT="$2"
                shift 2 ;;
            -id | --input-dir )
                INPUT_DIR="$2"
                shift 2 ;;
            -p1 | --phase1-ckpt )
                PHASE_1_CKPT="$2"
                shift 2 ;;
            -ed | --eval-dir )
                EVAL_DIR="$2"
                shift 2 ;;
            -od | --output-dir )
                OUTPUT_DIR="$2"
                shift 2 ;;
            -- )
                shift
                break ;;
            * )
                if [[ -n "$1" ]]; then
                    echo "error: invalid parameter: $1"
                    exit -1
                fi
                break ;;
        esac
    done

    [[ "$REMOTE_HOSTS" =~ 'localhost' ]] || _multibox=true
}


SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")/../PyTorch
DATA_ROOT=/mnt/weka/data/pytorch/bert_mlperf/packed_data
INPUT_DIR=$DATA_ROOT/packed
PHASE_1_CKPT=$DATA_ROOT/phase1/model.ckpt-28252.pt
EVAL_DIR=$DATA_ROOT/hdf5/eval_varlength/
OUTPUT_DIR=/tmp/BERT_PRETRAINING

# parse arguments, possibly overwriting the default settings
parse_args "$@"

# output files for console logging for train/eval
DESC_FILE=$OUTPUT_DIR/desc.txt
TRAIN_LOG_FILE=$OUTPUT_DIR/train.log

# output directories for train/eval
RESULTS_DIR_FOR_TRAIN_EVAL=$OUTPUT_DIR/results

# checkpoint dirs for train/eval
CKPT_DIR_FOR_TRAIN_AND_EVAL=$RESULTS_DIR_FOR_TRAIN_EVAL/checkpoints

# dlloger output files for train/eval
DLLOGER_OUT_TRAIN_AND_EVAL=$RESULTS_DIR_FOR_TRAIN_EVAL/dlloger.json

# tensorboard directory for train/eval
TB_DIR_TRAIN=$OUTPUT_DIR/tb_train

# Performance flags
export ENABLE_EXPERIMENTAL_FLAGS=1
export ENABLE_SHARED_MULTIBUF_PER_SLICED_CHAIN=1

# MASTER_ADDR and MASTER_PORT are consumed by PyTorch c10d to establish a distributed group
if [ -z "$_multibox" ]; then
    _master_addr='127.0.0.1'
else
    _master_addr=`hostname -i`
fi
export MASTER_ADDR=${MASTER_ADDR:-$_master_addr}
export MASTER_PORT=${MASTER_PORT:-12345}

# build the primary mpirun command for the preparation and training
read -r -d '' MPIRUN_CMD << EOM
    mpirun \
        --allow-run-as-root \
        --tag-output \
        -H $REMOTE_HOSTS
EOM

if [ -n "$_multibox" ]; then
    read -r -d '' MPIRUN_CMD << EOM
    $MPIRUN_CMD \
        --mca plm_rsh_args "-p$REMOTE_SSH_PORT" \
        --prefix /opt/amazon/openmpi \
        -x DATA_LOADER_AEON_LIB_PATH \
        -x FI_EFA_ENABLE_SHM_TRANSFER \
        -x GC_KERNEL_PATH \
        -x HABANA_LOGS \
        -x HABANA_PLUGINS_LIB_PATH \
        -x HABANA_SCAL_BIN_PATH \
        -x HABANA_PROFILE \
        -x LD_LIBRARY_PATH \
        -x LD_PRELOAD \
        -x MASTER_ADDR \
        -x MASTER_PORT \
        -x PATH \
        -x PYTHONPATH \
        -x TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD
EOM
fi

# test mpirun invocation by counting workers
WORKER_REPORTS=`$MPIRUN_CMD echo 'WORKER_REPORTS' | grep -F 'WORKER_REPORTS'`
NUM_WORKERS=`echo "$WORKER_REPORTS" | wc -l`
IFS="," read -ra _distinc_host_arr <<< "$REMOTE_HOSTS"
NUM_NODES=${#_distinc_host_arr[@]}
NUM_LOCAL_WORKERS=`expr $NUM_WORKERS / $NUM_NODES`

# build the auxiliary mpirun command for (local) evaluation
read -r -d '' MPIRUN_LOCAL_CMD << EOM
    mpirun \
        --allow-run-as-root \
        --tag-output \
        -n $NUM_LOCAL_WORKERS
EOM

# determine key hyperparameters
case "$NUM_WORKERS" in
    8 )
        TOTAL_BATCH_SIZE=56
        TRAIN_STEPS=6700
        LEARNING_RATE=0.000425
        ;;
    16 )
        echo "warning: Using hyperparameters from BERT based on TensorFlow, which does not work for this PyTorch-based version. The model will not get trained properly."
        TOTAL_BATCH_SIZE=16
        TRAIN_STEPS=1140
        LEARNING_RATE=0.002
        ;;
    * )
        echo "error: invalid or unsupported total number of workers: $NUM_WORKERS"
        exit -1
esac

# install requirements and reset the output directory (on every node)
read -r -d '' PREPARE_CMD << EOM
    if [[ \$OMPI_COMM_WORLD_LOCAL_RANK == 0 ]]; then \
        echo "Installing requirements" && python3 -m pip install -r ${SCRIPT_DIR}/requirements.txt ; \
        rm -rf $OUTPUT_DIR ; \
        mkdir -p $CKPT_DIR_FOR_TRAIN_AND_EVAL ; \
        mkdir -p $TB_DIR_TRAIN ; \
    fi
EOM
$MPIRUN_CMD bash -c "$PREPARE_CMD"

# setup mpirun core binding
MPI_MAP_BY_PE=${MPI_MAP_BY_PE:-`lscpu | grep "^CPU(s):"| awk -v NUM=${NUM_LOCAL_WORKERS} '{print int($2/NUM/2)}'`}
read -r -d '' MPIRUN_CMD << EOM
$MPIRUN_CMD \
    --bind-to core \
    --map-by socket:PE=$MPI_MAP_BY_PE \
    --rank-by core \
    --report-bindings
EOM
read -r -d '' MPIRUN_LOCAL_CMD << EOM
$MPIRUN_LOCAL_CMD \
    --bind-to core \
    --map-by socket:PE=$MPI_MAP_BY_PE \
    --rank-by core \
    --report-bindings
EOM

# label the run (on this node)
cat > $DESC_FILE <<- EOM
Date                    : `date`

# parameters configurable from the environment
MPI_MAP_BY_PE           : $MPI_MAP_BY_PE  (numer of CPU cores assigned exclusively to each worker process)
MASTER_ADDR             : $MASTER_ADDR  (hostname or IP address of the distributed group leader)
MASTER_PORT             : $MASTER_PORT  (socket port number used by PyTorch c10d to establish a distributed group)

# input parameters
REMOTE_HOSTS            : $REMOTE_HOSTS  (comma-separated list of workers' hostnames or IP addresses)
REMOTE_SSH_PORT         : $REMOTE_SSH_PORT  (socket port number used by mpirun to establish a inter-process connections in multi-box mode)

# other parameters which are in effect
NUM_WORKERS             : $NUM_WORKERS  (total number of distributed workers)
NUM_NODES               : $NUM_NODES  (number of nodes involved)
NUM_LOCAL_WORKERS       : $NUM_LOCAL_WORKERS  (number of distributed workers per node)

# hyperparameters
TOTAL_BATCH_SIZE        : $TOTAL_BATCH_SIZE
TRAIN_STEPS             : $TRAIN_STEPS
LEARNING_RATE           : $LEARNING_RATE

dataset                 : packed
gradient_accumulation_steps : 2  (effectively 1 accumulation per update due to packed dataset)
EOM
cat $DESC_FILE

echo
echo 'Running training & evaluation'
echo
set -x
time $MPIRUN_CMD python3 $SCRIPT_DIR/run_pretraining.py \
    --bert_model bert-large-uncased \
    --config_file $SCRIPT_DIR/bert_config.json \
    --output_dir $CKPT_DIR_FOR_TRAIN_AND_EVAL \
    --do_train \
    --do_eval \
    --json-summary $DLLOGER_OUT_TRAIN_AND_EVAL \
    --use_fused_lamb \
    --use_habana \
    --hmp \
    --hmp_bf16 $SCRIPT_DIR/ops_bf16_bert_pt.txt \
    --hmp_fp32 $SCRIPT_DIR/ops_fp32_bert_pt.txt \
    --input_dir $INPUT_DIR \
    --max_seq_length 512 \
    --train_batch_size $TOTAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_predictions_per_seq 76 \
    --warmup_proportion 0 \
    --max_steps $TRAIN_STEPS \
    --gradient_accumulation_steps 2 \
    --num_steps_per_checkpoint 335 \
    --phase2 \
    --phase1_end_step 7 \
    --log_freq 10 \
    --init_checkpoint $PHASE_1_CKPT \
    --resume_step 28252 \
    --eval_dir $EVAL_DIR \
    --eval_batch_size 125 \
    --num_eval_examples 10000 \
    --enable_packed_data_mode true \
    --checkpoint_filter model \
    --use_fastddp \
    --tensorboard_dir $TB_DIR_TRAIN  2>&1 | tee $TRAIN_LOG_FILE
retval="$?"
set +x
