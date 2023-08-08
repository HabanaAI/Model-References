#!/bin/bash

function print_synopsis()
{
    cat << EOF
NAME
        `basename $0`

SYNOPSIS
        `basename $0` [-H <hosts>] [-p <sshport>] [-dd <data-dir>] [-od <output-dir>] [-ep <enable-profiling>] [-ee <enable-evaluation>]

DESCRIPTION
        Runs MLPerf BERT pre-training training and evaluation on PyTorch with optional Habana profiling runs.

        -hf, --hosts-file
            path to a file containing list of IPs for nodes.

        -H, --hosts
            comma-separated list of workers' hostnames or IP addresses
            default: localhost:8

        -p, --ssh-port
            socket port number used by mpirun to establish a inter-process communication for multi-box
            default: 3022

        -dd, --data-dir
            specify the data directory, containing books-wiki packed dataset.

        -od, --output-dir
            specify the output directory, used to store training results.

        -ee, --enable-evaluation
            if set, evaluation will be executed after training
            default: true

        -h, --help
            prints this help message.

EXAMPLES
       `basename $0`                                                 # 8-Gaudi local run
       `basename $0` -dd /mnt/data/books_wiki_packed -od /tmp/output # 8-Gaudi local run, overriding data-dir and output-dir
       `basename $0` -ep true                                        # 8-Gaudi local run, with profiling enabled
       `basename $0` -H 10.111.131.28:8,10.111.131.27:8              # 16-Gaudi multi-box run
EOF
}

function generate_hosts_list()
{
    HOSTS_PATH=$1
    local num_nodes=${2:-8}
    HOSTS_LIST=""

    while IFS= read -r ip; do
        HOSTS_LIST="$HOSTS_LIST,$ip:8"
    done < "$HOSTS_PATH"

    echo "${HOSTS_LIST:1}"
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
            -hf | --hosts-file )
                REMOTE_HOSTS=$(generate_hosts_list "$2" 8)
                shift 2 ;;
            -H | --hosts )
                REMOTE_HOSTS="$2"
                shift 2 ;;
            -p | --ssh-port )
                REMOTE_SSH_PORT="$2"
                shift 2 ;;
            -dd | --data-dir )
                DATA_ROOT="$2"
                shift 2 ;;
            -od | --output-dir )
                OUTPUT_DIR="$2"
                shift 2 ;;
            -ee | --enable-evaluation )
                ENABLE_EVALUATION="$2"
                shift 2 ;;
            -ua | --use-autocast )
                USE_AUTOCAST="$2"
                shift 2 ;;
            -ll | --lower-list )
                LOWER_LIST="$2"
                shift 2 ;;
            -fl | --f32-list )
                FP32_LIST="$2"
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
ENABLE_EVALUATION=true
USE_AUTOCAST=true

# autocast files paths
LOWER_LIST=$SCRIPT_DIR/ops_bf16_bert_pt.txt
FP32_LIST=$SCRIPT_DIR/ops_fp32_bert_pt.txt

# parse arguments, possibly overwriting the default settings
parse_args "$@"

# output files for console logging for train/eval
DESC_FILE=$OUTPUT_DIR/desc.txt
TRAIN_LOG_FILE=$OUTPUT_DIR/train.log
EVAL_LOG_FILE=$OUTPUT_DIR/eval.log

# output directories for  train/eval
RESULTS_DIR_FOR_TRAIN_EVAL=$OUTPUT_DIR/results
RESULTS_DIR_FOR_INIT_CKPT_EVAL=$OUTPUT_DIR/result_init_ckpt

# checkpoint dirs for train/eval
CKPT_DIR_FOR_TRAIN_AND_EVAL=$RESULTS_DIR_FOR_TRAIN_EVAL/checkpoints
RESULTS_DIR_FOR_INIT_CKPT_EVAL=$RESULTS_DIR_FOR_INIT_CKPT_EVAL/checkpoints

# dlloger output files for train/eval
DLLOGER_OUT_TRAIN_AND_EVAL=$RESULTS_DIR_FOR_TRAIN_EVAL/dlloger.json

# tensorboard directory for run and train
TB_DIR_TRAIN=$OUTPUT_DIR/tb_train
TB_DIR_INIT_EVAL=$OUTPUT_DIR/tb_init_eval

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
        -x LOWER_LIST=$LOWER_LIST \
        -x FP32_LIST=$FP32_LIST \
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
        -x LD_LIBRARY_PATH \
        -x LD_PRELOAD \
        -x MASTER_ADDR \
        -x MASTER_PORT \
        -x PATH \
        -x PYTHONPATH \
        -x TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD \
        -x ENABLE_EXPERIMENTAL_FLAGS \
        -x ENABLE_SHARED_MULTIBUF_PER_SLICED_CHAIN \
        -x HCL_USE_EDMA_COMMAND_V2_GAUDI2 \
        -x PT_HPU_LAZY_ACC_PAR_MODE
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
        NUM_ACCUM_STEPS=2
        TOTAL_BATCH_SIZE=$((28*$NUM_ACCUM_STEPS))
        TRAIN_STEPS=6700
        LEARNING_RATE=0.000425
        SAVE_CHECKPOINTS_STEPS=335
        LAMB_BETA_1=0.9
        LAMB_BETA_2=0.999
        LAMB_WEIGHT_DECAY=0.01
        EVAL_BATCH_SIZE=125
        MAX_EVAL_STEPS=10
        NUM_DIST_EVAL_WORKERS=8
        # 'fastddp' distribution strategy performs gradient clipping normalization before all-reduce, but effectively there is no gradient accumulation
        DISTRIBUTION_OPTION='--distribution_strategy fastddp --allreduce_dtype bf16 --use_hpu_graph true'
        ;;
    16 )
        NUM_ACCUM_STEPS=8
        TOTAL_BATCH_SIZE=$((24*$NUM_ACCUM_STEPS))
        TRAIN_STEPS=1140
        LEARNING_RATE=0.002
        SAVE_CHECKPOINTS_STEPS=57
        LAMB_BETA_1=0.71
        LAMB_BETA_2=0.998
        LAMB_WEIGHT_DECAY=0.01
        EVAL_BATCH_SIZE=125
        MAX_EVAL_STEPS=5
        NUM_DIST_EVAL_WORKERS=16
        # 'gradientbuffer' distribution strategy performs gradient clipping normalization before every accumulation
        DISTRIBUTION_OPTION='--distribution_strategy gradientbuffer --allreduce_dtype fp32'

        # Disable PT bridge's accumulation thread (see: SW-113836)
        export PT_HPU_LAZY_ACC_PAR_MODE=0
        ;;
    32 )
        NUM_ACCUM_STEPS=4
        TOTAL_BATCH_SIZE=$((24*$NUM_ACCUM_STEPS))
        TRAIN_STEPS=1140
        LEARNING_RATE=0.002
        SAVE_CHECKPOINTS_STEPS=57
        LAMB_BETA_1=0.71
        LAMB_BETA_2=0.998
        LAMB_WEIGHT_DECAY=0.01
        EVAL_BATCH_SIZE=100
        MAX_EVAL_STEPS=4
        NUM_DIST_EVAL_WORKERS=25
        # 'gradientbuffer' distribution strategy performs gradient clipping normalization before every accumulation
        DISTRIBUTION_OPTION='--distribution_strategy gradientbuffer --allreduce_dtype fp32'

        # Disable PT bridge's accumulation thread (see: SW-113836)
        export PT_HPU_LAZY_ACC_PAR_MODE=0
        ;;
    64 )
        NUM_ACCUM_STEPS=2
        TOTAL_BATCH_SIZE=$((24*$NUM_ACCUM_STEPS))
        TRAIN_STEPS=1140
        LEARNING_RATE=0.0021
        SAVE_CHECKPOINTS_STEPS=57
        LAMB_BETA_1=0.625
        LAMB_BETA_2=0.8675
        LAMB_WEIGHT_DECAY=0.0075
        EVAL_BATCH_SIZE=100
        MAX_EVAL_STEPS=2
        NUM_DIST_EVAL_WORKERS=50
        # 'gradientbuffer' distribution strategy performs gradient clipping normalization before every accumulation
        # [SW-114363] TODO: As there is effectively only 1 accumulation per step, consider using FastDDP. Check it with bf16.
        DISTRIBUTION_OPTION='--distribution_strategy fastddp --allreduce_dtype bf16 --use_hpu_graph true'
        ;;
    * )
        echo "error: invalid or unsupported total number of workers: $NUM_WORKERS"
        exit -1
esac

# NUM_EVAL_EXAMPLES is always expected to 10000.
NUM_EVAL_EXAMPLES=$(($EVAL_BATCH_SIZE*$NUM_DIST_EVAL_WORKERS*$MAX_EVAL_STEPS))
if [[ $NUM_EVAL_EXAMPLES != 10000 ]]; then
    echo "error: NUM_EVAL_EXAMPLES must be exactly 10000, but is: $NUM_EVAL_EXAMPLES"
    exit -1
fi

# TOTAL_BATCH_SIZE already contains NUM_ACCUM_STEPS this is why it is not present here
SAMPLES_BETWEEN_EVAL=$(($TOTAL_BATCH_SIZE*$NUM_WORKERS*$SAVE_CHECKPOINTS_STEPS))

PROC_FS=${PROC_FS:-"/proc"}
# install requirements, clear caches and reset the output directory (on every node)
read -r -d '' PREPARE_CMD << EOM
    if [[ \$OMPI_COMM_WORLD_LOCAL_RANK == 0 ]]; then \
        echo "Installing requirements" && python3 -m pip install -r ${SCRIPT_DIR}/requirements.txt ; \
        sync && echo 3 > $PROC_FS/sys/vm/drop_caches ; \
        rm -rf $OUTPUT_DIR ; \
        mkdir -p $CKPT_DIR_FOR_TRAIN_AND_EVAL ; \
        mkdir -p $TB_DIR_TRAIN ; \
        mkdir -p $RESULTS_DIR_FOR_INIT_CKPT_EVAL ; \
        mkdir -p $TB_DIR_INIT_EVAL ; \
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

# training
TOTAL_BATCH_SIZE        : $TOTAL_BATCH_SIZE
TRAIN_STEPS             : $TRAIN_STEPS
LEARNING_RATE           : $LEARNING_RATE
LAMB_BETA_1             : $LAMB_BETA_1
LAMB_BETA_2             : $LAMB_BETA_2
LAMB_WEIGHT_DECAY       : $LAMB_WEIGHT_DECAY
dataset                 : packed
gradient_accumulation_steps : $NUM_ACCUM_STEPS  (effectively /2 due to packed dataset)

# evaluation
SAVE_CHECKPOINTS_STEPS  : $SAVE_CHECKPOINTS_STEPS
EVAL_BATCH_SIZE         : $EVAL_BATCH_SIZE
MAX_EVAL_STEPS          : $MAX_EVAL_STEPS
NUM_DIST_EVAL_WORKERS   : $NUM_DIST_EVAL_WORKERS
NUM_EVAL_EXAMPLES       : $NUM_EVAL_EXAMPLES
EOM
cat $DESC_FILE

if [ "$ENABLE_EVALUATION" == "true" ]; then
    echo
    echo 'Running training & evaluation'
    echo
    EVALUATION_FLAG="--do_eval"
else
    echo
    echo 'Running training (no evaluation)'
    echo
    EVALUATION_FLAG=""
fi

set -x
time $MPIRUN_CMD python3 $SCRIPT_DIR/run_pretraining.py \
    --config_file $SCRIPT_DIR/bert_config.json \
    --output_dir $CKPT_DIR_FOR_TRAIN_AND_EVAL \
    --do_train \
    $EVALUATION_FLAG \
    --json-summary $DLLOGER_OUT_TRAIN_AND_EVAL \
    --use_fused_lamb \
    --use_habana \
    --use_autocast $USE_AUTOCAST \
    --input_dir $DATA_ROOT/packed_data_500_pt \
    --max_seq_length 512 \
    --train_batch_size $TOTAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --lamb_beta_1 $LAMB_BETA_1 \
    --lamb_beta_2 $LAMB_BETA_2 \
    --lamb_weight_decay $LAMB_WEIGHT_DECAY \
    --max_predictions_per_seq 76 \
    --warmup_proportion 0 \
    --max_steps $TRAIN_STEPS \
    --gradient_accumulation_steps $NUM_ACCUM_STEPS \
    --num_steps_per_checkpoint $SAVE_CHECKPOINTS_STEPS \
    --phase2 \
    --log_freq 20 \
    --init_checkpoint $DATA_ROOT/phase1/model.ckpt-28252.pt \
    --eval_dir $DATA_ROOT/hdf5/eval_varlength/ \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --num_eval_examples $NUM_EVAL_EXAMPLES \
    --num_eval_workers $NUM_DIST_EVAL_WORKERS \
    --samples_between_eval $SAMPLES_BETWEEN_EVAL \
    --enable_packed_data_mode true \
    --checkpoint_filter model \
    $DISTRIBUTION_OPTION \
    --tensorboard_dir $TB_DIR_TRAIN  2>&1 | tee $TRAIN_LOG_FILE
retval="$?"
set +x
