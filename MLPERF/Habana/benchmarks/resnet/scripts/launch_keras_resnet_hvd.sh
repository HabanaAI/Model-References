#!/bin/bash

DEBUG=${DEBUG:-0}
if [[ $DEBUG -eq 1 ]]; then
    set -x
    env
fi

# Basic paths
BASE_PATH="$( cd "$(dirname "$(readlink -f ./defaults.cfg)" )" && pwd)"
PYTHONPATH=${BASE_PATH}:$PYTHONPATH
SHARED_DIR=/root/shared
IMAGENET_DIR=/root/datasets/imagenet/tf_records
exit_code=0

# Determine OpenMPI prefix from location of mpirun in the system
OMPI_PREFIX=$(which mpirun)
OMPI_PREFIX=$(dirname $(dirname ${OMPI_PREFIX}) )

# Fixed variables needed by this script
export NUM_WORKERS_PER_HLS=${NUM_WORKERS_PER_HLS:-4}
export HLS_TYPE=${HLS_TYPE:-HLS1-H}

function help()
{
    echo  "Usage:"
    echo  "$0 [ -key1 value1 -key2 value2  .... -keyn valuen ]"
    echo  "-b      | --batch-size       Batch size"
    echo  "-c      | --config           Configuration file path (defaults to ./defaults.cfg)"
    echo  "-pc     | --profile_config   Profiling configuration file for merged HW trace (defaults to ../../scripts/synprof_mergedHW.json)"
    echo  "-p      | --cpu-pin          [ none | cpu | numa ]"
    echo  "-a      | --data-dir         Imagenet data directory"
    echo  "-dc     | --dataset_cache    Enable (true) or disable (false) dataset caching"
    echo  "-noEval | --disable_eval     Enable (0) or disable (1) evaluation"
    echo  "-bf16   | --enable_bf16"
    echo  "-mlperf | --enable_mlperf"
    echo  "-e      | --epochs           Number of epochs"
    echo  "-ebe    | --epochs_between_eval  Number of training epochs between eval"
    echo  "-eof    | --epoch_eval_offset"
    echo  "-hp     | --habana-profiler  Enable habana profiler"
    echo  "-hf     | --hostfile		Host file path (defaults to /root/shared/hosts)"
    echo  "-hwp    | --hw_profile_range TF hooks for HW profiler in <start_hook>:<end_hook> format"
    echo  "-lbs    | --label_smoothing"
    echo  "-l      | --lars-opt         Enable LARS optimizer"
    echo  "-lde    | --lars_decay_epochs"
    echo  "-mm     | --momentum"
    echo  "-md     | --modeling		Enable (1) or disable (0) TF dumpos for SPARTA modeling"
    echo  "-w      | --number-worker    Number of Gaudis"
    echo  "-rs     | --resnet-size      Resnet size"
    echo  "-slr    | --start_learning_rate"
    echo  "-sth    | --stop_thold       Target accuracy"
    echo  "-s      | --steps            Display logs for evey number of steps"
    echo  "-sl     | --steps-per-loop   Number of steps per loop in Keras implementation"
    echo  "-sd     | --synthetic_data   Enable (1) or disable (0) synthetic dataset"
    echo  "-sp     | --syn_profile      Enable (1) or disable (0) synapse logger"
    echo  "-tev    | --train_eval"
    echo  "-tps    | --tf_profile_steps TF steps to profile in <start_step>,<end_step> format"
    echo  "-ts     | --train_steps      Train steps, will be overwritten if epochs > 0"
    echo  "-u      | --use_horovod      Enable (1) or disable (0) horovod use"
    echo  "-we     | --warmup_epochs"
    echo  "-wd     | --weight_decay"
    echo  "-crc    | --clean_recipe_cache Clean TF recipe cache Enable (1) disable (0), default: 1"
    echo  "-tcp    | --mpi_tcp_include"
    echo  "-log    | --log_dir"
    echo  "-ps     | --pod_size"
}

function getmulti_hls_ips()
{
    multi_hcl_ip="MULTI_HLS_IPS="
    hostsFile=${1:-"$SHARED_DIR/hosts"}
    firstHost=1
    hostCount=0

    # Remove empty lines from host file
    sed -i '/^$/d' $hostsFile
    #  when reading hosts, exclude commented lines #
    for h in $(grep -v '^#' $hostsFile); do
        if [[ $firstHost -eq 1 ]]; then
            firstHost=0
        else
            multi_hcl_ip+=","
        fi
        multi_hcl_ip+=$h
        hostCount=$((hostCount + 1))
    done

    echo "[getmulti_hls_ips] Host Count : $hostCount"
    echo "[getmulti_hls_ips] Exporting  : $multi_hcl_ip"
    export $multi_hcl_ip
}

function generate_hcl_config()
{
    echo "Generating HCL Config..."
    local path=$1
    local devices_per_hls=$2
    local hls_type=${3:-HLS1}
    local file_name="config_$devices_per_hls.json"
    export HCL_CONFIG_PATH=$path/${file_name}
    rm -f ${HCL_CONFIG_PATH}
    echo "Path: ${HCL_CONFIG_PATH}"
    touch ${HCL_CONFIG_PATH}
    echo "{"                                             >> ${HCL_CONFIG_PATH}
    echo "    \"HCL_PORT\": 5332,"                       >> ${HCL_CONFIG_PATH}
    echo "    \"HCL_TYPE\": \"$hls_type\","              >> ${HCL_CONFIG_PATH}
    if [[ -z ${MULTI_HLS_IPS} ]]; then
        echo "    \"HCL_COUNT\": $devices_per_hls"       >> ${HCL_CONFIG_PATH}
    else
        echo "    \"HCL_RANKS\": ["                      >> ${HCL_CONFIG_PATH}
        IFS=',' read -ra IPS <<< "$MULTI_HLS_IPS"
        for ip in "${IPS[@]}"; do
            for (( c=1; c<=$devices_per_hls; c++ )); do
                if [[ $ip == ${IPS[${#IPS[@]}-1]} ]] && [[ $c == $devices_per_hls ]];
                then
                    echo "        \"$ip\""               >> ${HCL_CONFIG_PATH}
                else
                    echo "        \"$ip\","              >> ${HCL_CONFIG_PATH}
                fi
            done
        done
        echo "    ]"                                     >> ${HCL_CONFIG_PATH}
    fi
    echo "}"                                             >> ${HCL_CONFIG_PATH}
    echo "Config: "
    cat ${HCL_CONFIG_PATH}
}

function generate_mpi_hostfile()
{
    echo "Generating MPI hostfile..."
    local num_nodes=${2:-8}
    local file_name="hostfile"
    export MPI_HOSTFILE_PATH=$1/${file_name}

    rm -rf ${MPI_HOSTFILE_PATH}
    echo "PATH: ${MPI_HOSTFILE_PATH}"
    touch ${MPI_HOSTFILE_PATH}

    IFS=',' read -ra IPS <<< "$MULTI_HLS_IPS"
    for i in "${IPS[@]}"; do
        echo "$i slots=${num_nodes}" >> ${MPI_HOSTFILE_PATH}
    done

    echo "Config: "
    cat ${MPI_HOSTFILE_PATH}
}

function run_per_ip()
{
    if [ -n "$OMPI_COMM_WORLD_SIZE" ]; then
        print_error "Function run_per_ip is not meant to be ran from within an OpenMPI context. It is intended to invoke mpirun by itelf."
        exit 1
    fi

    _cmd="$@"

    # Due to technical difficulties with the following solution, the _cmd stderr shall be redirected to stdout.
    if [[ -z ${MULTI_HLS_IPS} ]]; then
        $_cmd 2>&1
    else
        if [ -n "$MPI_TCP_INCLUDE" ]; then
            _option_btl_tcp_if_include="--mca btl_tcp_if_include ${MPI_TCP_INCLUDE}"
        else
            _option_btl_tcp_if_include=""
        fi

        mpirun --allow-run-as-root \
            --mca plm_rsh_args -p3022 \
            ${_option_btl_tcp_if_include} \
            --tag-output \
            --merge-stderr-to-stdout \
            --prefix ${OMPI_PREFIX} \
            -H ${MULTI_HLS_IPS} \
            bash -c "`declare`; `declare -x`; ($_cmd 2>&1)" 2>/dev/null
    fi
}

# Parse command line options
unset __bsize
unset __config
unset __profile_cfg
unset __data_dir
unset __dataset_cache
unset __disable_eval
unset __enable_bf16_conversion
unset __enable_habana_profile
unset __enable_lars
unset __enable_mlperf
unset __epochs
unset __epochs_between_eval
unset __eval_offset_epochs
unset __hostfile
unset __hw_profile_range
unset __label_smoothing
unset __lars_decay_epochs
unset __momentum
unset __modeling
unset __number_Worker
unset __tf_profile_steps
unset __resnetSize
unset __run_number
unset __start_learning_rate
unset __steps
unset __steps_per_loop
unset __stop_thold
unset __syn_profile
unset __train_eval
unset __train_steps
unset __use_horovod
unset __synthetic_data
unset __warmup_epochs
unset __weight_decay
unset __workload_to_cpu_pin_type
unset __dataset_cache
unset __clean_recipe_cache
unset __mpi_tcp_include
unset __log_dir
unset __pod_size

while [ -n "$1" ]; do
    case $1 in
        -rs | --resnet-size )
            shift
            __resnetSize=$1
            ;;
	-c | --config )
	    shift
	    __config=$1
	    ;;
	-pc | --profile_config )
	    shift
	    __profile_cfg=$1
	    ;;
        -a | --data-dir )
            shift
            __data_dir=$1
            ;;
        -b | --batch-size )
            shift
            __bsize=$1
            ;;
        -s | --steps )
            shift
            __steps=$1
            ;;
        -sl | --steps-per-loop )
            shift
            __steps_per_loop=$1
            ;;
        -e | --epochs )
            shift
            __epochs=$1
            ;;
        -w | --number-worker )
            shift
            __number_Worker=$1
            ;;
	-hf | --hostfile)
	    shift
	    __hostfile=$1
	    ;;
        -hp | --habana-profiler)
            shift
            __enable_habana_profile=$1
            ;;
        -l | --lars-opt)
            shift
            __enable_lars=$1
            ;;
	-dc | --dataset_cache)
	    shift
	    __dataset_cache=$1
	    ;;
        -ebe | --epochs_between_eval)
            shift
            __epochs_between_eval=$1
            ;;
	-ts | --train_steps)
	    shift
	    __train_steps=$1
	    ;;
        -we | --warmup_epochs)
            shift
            __warmup_epochs=$1
            ;;
        -wd | --weight_decay)
            shift
            __weight_decay=$1
            ;;
        -lbs | --label_smoothing)
            shift
            __label_smoothing=$1
            ;;
        -slr | --start_learning_rate)
            shift
            __start_learning_rate=$1
            ;;
	-sd  | --synthetic_data)
	    shift
	    __synthetic_data=$1
	    ;;
        -mlperf | --enable_mlperf)
            shift
            __enable_mlperf=$1
            ;;
        -noEval | --disable_eval)
            shift
            __disable_eval=$1
            ;;
        -tev | --train_eval)
            shift
            __train_eval=$1
            ;;
        -bf16 | --enable_bf16)
            shift
            __enable_bf16_conversion=$1
            ;;
        -eof | --epoch_eval_offset)
            shift
            __eval_offset_epochs=$1
            ;;
        -sth | --stop_thold)
            shift
            __stop_thold=$1
            ;;
        -mm | --momentum)
            shift
            __momentum=$1
            ;;
	-md | --modeling)
	    shift
	    __modeling=$1
	    ;;
        -rn | --run-number )
            shift
            __run_number=$1
            ;;
        -p | --cpu-pin)
            shift
            __workload_to_cpu_pin_type=$1
            case ${__workload_to_cpu_pin_type} in
                numa | cpu | none )
                         ;;
                *)
                echo "--cpu-pin must be one of the following numa | cpu | none "
                exit 1
            esac
            ;;
	-u | --use_horovod)
	    shift
	    __use_horovod=$1
	    ;;
	-tps | --tf_profile_steps)
	     shift
	     __tf_profile_steps=$1
	    ;;
	-hwp | --hw_profile_range)
	     shift
	     __hw_profile_range=$1
	     ;;
	-sp  | --syn_profile)
	     shift
	     __syn_profile=$1
	     ;;
	-dc | --dataset_cache)
	      shift
	      __dataset_cache=$1
	      ;;
	-lde | --lars_decay_epochs)
		shift
		__lars_decay_epochs=$1
		;;
	-crc | --clean_recipe_cache)
            shift
	    __clean_recipe_cache=$1
	    ;;
	-tcp | --mpi_tcp_include)
	    shift
	    __mpi_tcp_include=$1
	    ;;
	-log | --log_dir)
	    shift
	    __log_dir=$1
	    ;;
	-ps  | --pod_size)
	    shift
	    __pod_size=$1
	    ;;
        -h | --help)
            help
	    exit 1
            ;;
            * )
            echo "The parameter $1 is not allowed"
            help
            ;;
    esac
    shift
done

# Set default values for environmental variable
export CFG_FILE=${__config:-"${BASE_PATH}/defaults.cfg"}
export HOST_FILE=${__hostfile:-"/root/shared/hosts"}
export PROFILE_CONFIG_FILE=${__profile_cfg:-"synprof_mergedHW.json"}
export PROFILE_CONFIG="${BASE_PATH}/../../../profile/${PROFILE_CONFIG_FILE}"
if [[ -f ${CFG_FILE} ]]; then
        source ${CFG_FILE}
else
        echo "Could not find ${CFG_FILE}"
        exit 1
fi

# set default LOG_DIR
export testdate=`date +%Y-%m-%d`
export testtime=`date +%H%M%S`
export LOG_DIR=/root/scratch/resnet/resnet_gaudi${NUM_WORKERS}_${testdate}_${testtime}

# Override defaults with command line options if needed
export RESNET_SIZE=${__resnetSize:-"$RESNET_SIZE"}
export IMAGENET_DIR=${__data_dir:-"$IMAGENET_DIR"}
export BATCH_SIZE=${__bsize:-"$BATCH_SIZE"}
export TRAIN_EPOCHS=${__epochs:-"$TRAIN_EPOCHS"}
export TRAIN_STEPS=${__train_steps:-"$TRAIN_STEPS"}
export DISPLAY_STEPS=${__steps:-"$DISPLAY_STEPS"}
export STEPS_PER_LOOP=${__steps_per_loop:-"$STEPS_PER_LOOP"}
export NUM_WORKERS=${__number_Worker:-"$NUM_WORKERS"}
export USE_LARS_OPTIMIZER=${__enable_lars:-"$USE_LARS_OPTIMIZER"}
export CPU_BIND_TYPE=${__workload_to_cpu_pin_type:-"$CPU_BIND_TYPE"}
export HABANA_PROFILE=${__enable_habana_profile:-"$HABANA_PROFILE"}
export EPOCHS_BETWEEN_EVALS=${__epochs_between_eval:-"$EPOCHS_BETWEEN_EVALS"}
export WEIGHT_DECAY=${__weight_decay:-"$WEIGHT_DECAY"}
export LABEL_SMOOTH=${__label_smoothing:-"$LABEL_SMOOTH"}
export BASE_LEARNING_RATE=${__start_learning_rate:-"$BASE_LEARNING_RATE"}
export WARMUP_EPOCHS=${__warmup_epochs:-"$WARMUP_EPOCHS"}
export USE_MLPERF=${__enable_mlperf:-"$USE_MLPERF"}
export NO_EVAL=${__disable_eval:-"$NO_EVAL"}
export STOP_THRESHOLD=${__stop_thold:-"$STOP_THRESHOLD"}
export LR_MOMENTUM=${__momentum:-"$LR_MOMENTUM"}
export EVAL_OFFSET_EPOCHS=${__eval_offset_epochs:-"$EVAL_OFFSET_EPOCHS"}
export TF_ENABLE_BF16_CONVERSION=${_enable_bf16_conversion:-"$TF_ENABLE_BF16_CONVERSION"}
export USE_HOROVOD=${__use_horovod:-"$USE_HOROVOD"}
export DATASET_CACHE=${__dataset_cache:-"$DATASET_CACHE"}
export LARS_DECAY_EPOCHS=${__lars_decay_epochs:-"$LARS_DECAY_EPOCHS"}
export SYNTHETIC_DATA=${__synthetic_data:-"$SYNTHETIC_DATA"}
# Only used in model garden tf (no keras)
if [ -z ${__train_eval} ]; then
    export TRAIN_AND_EVAL=${__train_eval:-"$TRAIN_AND_EVAL"}
fi
export TRAIN_STEPS=${TRAIN_STEPS:--1}
export TF_PROFILE_STEPS=${__tf_profile_steps:-"$TF_PROFILE_STEPS"}
export HW_PROFILE_RANGE=${__hw_profile_range:-"$HW_PROFILE_RANGE"}
export SYN_PROFILE=${__syn_profile:-"$SYN_PROFILE"}
export MODELING=${__modeling:-"$MODELING"}
export CLEAN_RECIPE_CACHE=${__clean_recipe_cache:-1}
export MPI_TCP_INCLUDE=${__mpi_tcp_include:-$MPI_TCP_INCLUDE}
export LOG_DIR=${__log_dir:-"$LOG_DIR"}
export POD_SIZE=${__pod_size:-"$POD_SIZE"}

echo "[launch_keras_resnet_hvd] General Settings:"
echo "[launch_keras_resnet_hvd] CFG_FILE"  $CFG_FILE
echo "[launch_keras_resnet_hvd] HOST_FILE"  $HOST_FILE
echo "[launch_keras_resnet_hvd] NUM_WORKERS"  $NUM_WORKERS
echo "[launch_keras_resnet_hvd] RESNET_SIZE" $RESNET_SIZE
echo "[launch_keras_resnet_hvd] IMAGENET_DIR" $IMAGENET_DIR
echo "[launch_keras_resnet_hvd] BATCH_SIZE"  $BATCH_SIZE
echo "[launch_keras_resnet_hvd] TRAIN_EPOCHS" $TRAIN_EPOCHS
echo "[launch_keras_resnet_hvd] TRAIN_STEPS" $TRAIN_STEPS
echo "[launch_keras_resnet_hvd] DISPLAY_STEPS" $DISPLAY_STEPS
echo "[launch_keras_resnet_hvd] USE_LARS_OPTIMIZER" $USE_LARS_OPTIMIZER
echo "[launch_keras_resnet_hvd] CPU_BIND_TYPE" $CPU_BIND_TYPE
echo "[launch_keras_resnet_hvd] HABANA_PROFILE" $HABANA_PROFILE
echo "[launch_keras_resnet_hvd] EPOCHS_BETWEEN_EVALS" $EPOCHS_BETWEEN_EVALS
echo "[launch_keras_resnet_hvd] TRAIN_AND_EVAL" $TRAIN_AND_EVAL
echo "[launch_keras_resnet_hvd] TF_ENABLE_BF16_CONVERSION" $TF_ENABLE_BF16_CONVERSION
echo "[launch_keras_resnet_hvd] USE_HOROVOD" $USE_HOROVOD
echo "[launch_keras_resnet_hvd] DATASET_CACHE" $DATASET_CACHE
echo "[launch_keras_resnet_hvd] TF_PROFILE_STEPS" $TF_PROFILE_STEPS
echo "[launch_keras_resnet_hvd] HW_PROFILE_RANGE" $HW_PROFILE_RANGE
echo "[launch_keras_resnet_hvd] SYN_PROFILE" $SYN_PROFILE
echo "[launch_keras_resnet_hvd] PROFILE_CONFIG" $PROFILE_CONFIG
echo "[launch_keras_resnet_hvd] MODELING" $MODELING
echo "[launch_keras_resnet_hvd] MPI_TCP_INCLUDE" $MPI_TCP_INCLUDE
echo "[launch_keras_resnet_hvd] LOG_DIR" $LOG_DIR
echo "[launch_keras_resnet_hvd] POD_SIZE" $POD_SIZE
echo
echo "[launch_keras_resnet_hvd] Learning Setting:"
echo "[launch_keras_resnet_hvd] WEIGHT_DECAY" $WEIGHT_DECAY
echo "[launch_keras_resnet_hvd] LABEL_SMOOTH" $LABEL_SMOOTH
echo "[launch_keras_resnet_hvd] BASE_LEARNING_RATE" $BASE_LEARNING_RATE
echo "[launch_keras_resnet_hvd] WARMUP_EPOCHS" $WARMUP_EPOCHS
echo "[launch_keras_resnet_hvd] USE_MLPERF" $USE_MLPERF
echo "[launch_keras_resnet_hvd] NO_EVAL" $NO_EVAL
echo "[launch_keras_resnet_hvd] STOP_THRESHOLD" $STOP_THRESHOLD
echo "[launch_keras_resnet_hvd] LR_MOMENTUM" $LR_MOMENTUM
echo "[launch_keras_resnet_hvd] EVAL_OFFSET_EPOCHS" $EVAL_OFFSET_EPOCHS
echo "[launch_keras_resnet_hvd] LARS_DECAY_EPOCHS" $LARS_DECAY_EPOCHS
echo "[launch_keras_resnet_hvd] SYNTHETIC_DATA" $SYNTHETIC_DATA
echo "[launch_keras_resnet_hvd] WORK_DIR" $WORK_DIR

# This check always needs to go after all environment variable proccessing is complete.
if [[ ! -d ${IMAGENET_DIR} ]]; then
    echo "[launch_keras_resnet_hvd] ImageNet image database not found on ${IMAGENET_DIR}"
    exit -1
fi

getmulti_hls_ips ${HOST_FILE}

# Create recipes directory if it does not exist and adjust dirctory name
# if we are collecting traces - which require debug information
WORK_DIR=/tmp/resnet50
if [[ -n ${HW_PROFILE_RANGE} ]]; then
        WORK_DIR=${WORK_DIR}_trace
fi
# Setup the cahe directory and create ramdisk
export TF_RECIPE_CACHE_PATH=${WORK_DIR}/graph_dump_recipes
if [[ $CLEAN_RECIPE_CACHE -eq 1 ]]; then
    run_per_ip rm -rf ${TF_RECIPE_CACHE_PATH}
fi
run_per_ip mkdir -p ${TF_RECIPE_CACHE_PATH}
#run_per_ip mkdir -p /mnt/${WORK_DIR}
#run_per_ip mount -t tmpfs -o size=4g tmpfs /mnt/${WORK_DIR}
# Copy existing recipes from work directory to tmpfs
#printf "[launch_keras_resnet_hvd] Copying recipes to TF_DATA_CACHE_PATH - this could take some time...\n"
#run_per_ip cp -r ${WORK_DIR}/*.metadata /mnt/${WOR_KDIR}  2>/dev/null
#run_per_ip cp -r ${WORK_DIR}/*.recipe /mnt/${WORK_DIR}   2>/dev/null
#run_per_ip rm -rf ${WORK_DIR}/worker_*

run_per_ip 'cd ${BASE_PATH} && git clone https://github.com/mlperf/logging.git mlperf-logging'
run_per_ip 'cd ${BASE_PATH} && python3 -m pip install -e mlperf-logging'
run_per_ip 'mkdir -p ${BASE_PATH}/log'

printf "[launch_keras_resnet_hvd] Cleaning temp files...\n\n"
run_per_ip rm -rf /tmp/checkpoint /tmp/eval /tmp/events.out.tfevents.* /tmp/graph.pbtxt /tmp/model.ckpt-*
run_per_ip rm -rf /tmp/rank_*/checkpoint /tmp/rank_*/eval /tmp/rank_*/events.out.tfevents.* /tmp/rank_*/graph.pbtxt /tmp/rank_*/model.ckpt-*

if [[  -z ${MULTI_HLS_IPS} ]]; then
    echo "[launch_keras_resnet_hvd] MULTI_HLS_IPS undefined - maybe a missing /root/shared/hosts file?"
    exit -1
else

    run_per_ip generate_hcl_config '${WORK_DIR}' ${NUM_WORKERS_PER_HLS} ${HLS_TYPE} #&> /dev/null
    generate_hcl_config ${WORK_DIR} ${NUM_WORKERS_PER_HLS} ${HLS_TYPE}

    generate_mpi_hostfile ${WORK_DIR} ${NUM_WORKERS_PER_HLS}

    # Substituted this by the calculation below
    #calc_optimal_cpu_resources_for_mpi veces leri
    MPI_MAP_BY=socket
    MPI_MAP_BY_PE=`lscpu | grep "^CPU(s):"| awk -v NUM=${NUM_WORKERS_PER_HLS} '{print int($2/NUM/2)}'`
    if [[ "$CPU_BIND_TYPE" == "numa" ||  "$CPU_BIND_TYPE" == "none" ]]; then
        MPIRUN_ARGS_MAP_BY_PE="-bind-to none"
    else
        MPIRUN_ARGS_MAP_BY_PE="--bind-to core --map-by $MPI_MAP_BY:PE=$MPI_MAP_BY_PE"
    fi

    TRAINING_COMMAND="mpirun --allow-run-as-root \
            	-np $NUM_WORKERS --hostfile ${MPI_HOSTFILE_PATH} \
            	--prefix ${OMPI_PREFIX} \
            	--mca plm_rsh_args -p3022 \
            	--mca btl_tcp_if_include ${MPI_TCP_INCLUDE} \
		-x BASE_PATH=${BASE_PATH} \
		-x PYTHONPATH=${PYTHONPATH} \
		-x DATASET_CACHE=${DATASET_CACHE} \
		-x DEBUG=${DEBUG} \
		-x RESNET_SIZE=${RESNET_SIZE} \
		-x HABANA_PROFILE=${HABANA_PROFILE} \
            	-x HCL_CONFIG_PATH=${HCL_CONFIG_PATH} \
                -x IMAGENET_DIR=${IMAGENET_DIR} \
                -x TF_ENABLE_BF16_CONVERSION=${TF_ENABLE_BF16_CONVERSION} \
                -x TF_RECIPE_CACHE_PATH=${TF_RECIPE_CACHE_PATH} \
		-x LD_PRELOAD=${LD_PRELOAD} \
                -x TF_MODULES_RELEASE_BUILD=${TF_MODULES_RELEASE_BUILD} \
                -x GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
                -x HABANA_LOGS=${HABANA_LOGS} \
                -x CPU_BIND_TYPE=${CPU_BIND_TYPE} \
                -x WORK_DIR=${WORK_DIR} \
                -x BATCH_SIZE=${BATCH_SIZE} \
                -x TRAIN_EPOCHS=${TRAIN_EPOCHS} \
		-x TRAIN_STEPS=${TRAIN_STEPS} \
                -x DISPLAY_STEPS=${DISPLAY_STEPS} \
                -x STEPS_PER_LOOP=${STEPS_PER_LOOP} \
		-x NUM_WORKERS=${NUM_WORKERS} \
                -x EPOCHS_BETWEEN_EVALS=${EPOCHS_BETWEEN_EVALS} \
                -x EVAL_OFFSET_EPOCHS=${EVAL_OFFSET_EPOCHS} \
                -x WARMUP_EPOCHS=${WARMUP_EPOCHS} \
                -x LABEL_SMOOTH=${LABEL_SMOOTH} \
                -x WEIGHT_DECAY=${WEIGHT_DECAY} \
                -x LR_MOMENTUM=${LR_MOMENTUM} \
       		-x USE_LARS_OPTIMIZER=${USE_LARS_OPTIMIZER} \
		-x SYNTHETIC_DATA=${SYNTHETIC_DATA} \
       		-x BASE_LEARNING_RATE=${BASE_LEARNING_RATE} \
		-x USE_MLPERF=${USE_MLPERF} \
       		-x ENABLE_BARRIERS=0 \
       		-x SCALE_OUT_PORTS=1 \
       		-x STOP_THRESHOLD=${STOP_THRESHOLD} \
       		-x NO_EVAL=${NO_EVAL} \
		-x USE_HOROVOD=${USE_HOROVOD} \
		-x TRAIN_STEPS=${TRAIN_STEPS} \
		-x TF_PROFILE_STEPS=${TF_PROFILE_STEPS} \
		-x HW_PROFILE_RANGE=${HW_PROFILE_RANGE} \
		-x PROFILE_CONFIG=${PROFILE_CONFIG} \
		-x SYN_PROFILE=${SYN_PROFILE} \
		-x LARS_DECAY_EPOCHS=${LARS_DECAY_EPOCHS} \
		-x LOG_DIR=${LOG_DIR} \
		-x POD_SIZE=${POD_SIZE} \
                ${MPIRUN_ARGS_MAP_BY_PE} \
                -x MODELING=${MODELING} \
		--merge-stderr-to-stdout --output-filename ${LOG_DIR} \
                ${BASE_PATH}/run.sh"

    echo "TRAINING COMMAND = ${TRAINING_COMMAND}"
    printf "[launch_keras_resnet_hvd] Starting training...\n\n"
    $TRAINING_COMMAND
fi

#run_per_ip cp -r /mnt/${WORK_DIR}/* ${WORK_DIR}
rm -rf ${BASE_PATH}/log
cp /root/build_log.csv ${LOG_DIR}/
cp ${MPI_HOSTFILE_PATH} ${LOG_DIR}/
cp -r ${LOG_DIR} ${BASE_PATH}/log
chmod -R 777 ${LOG_DIR}
exit $exit_code
