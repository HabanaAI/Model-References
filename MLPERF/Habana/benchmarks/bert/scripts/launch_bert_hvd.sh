#!/bin/bash

#DEBUG=${DEBUG:-0}
#if [[ $DEBUG -eq 1 ]]; then
#    set -x
#    env
#fi

# Basic paths
export BASE_PATH="$( cd "$(dirname "$(readlink -f ./defaults.cfg)" )" && pwd)"
SHARED_DIR=/root/shared
exit_code=0

OMPI_PREFIX=$(which mpirun)
export OMPI_PREFIX=$(dirname $(dirname ${OMPI_PREFIX}) )

function help()
{
    echo  "Usage:"
    echo  "$0 [ -key1 value1 -key2 value2  .... -keyn valuen ]"
    echo  "-c      | --config           Configuration file path (defaults to ./defaults.cfg)"
    echo  "-hf     | --hostfile         Host file path (defaults to /root/shared/hosts)"
    echo  "-u      | --use_horovod      Enable (0) or disable (1) horovod use"
    echo  "-ws     | --warmup_steps"
    echo  "-lr     | --learning_rate"
    echo  "-st     | --stop_threshold"
    echo  "-acs    | --num_accumul_steps"
    echo  "-tbs    | --train_batchsize"
    echo  "-ebs    | --eval_batchsize"
    echo  "-ts     | --train_steps"
    echo  "-lb1    | --lamb_beta_1"
    echo  "-lb2    | --lamb_beta_2"
    echo  "-ep     | --epsilon"
    echo  "-lwd    | --lamb_weight_decay_rate"
    echo  "-ldp    | --lamb_lr_decay_poly_power"
    echo  "-sbe    | --samples_btw_eval"
    echo  "-sse    | --samples_start_eval"
    echo  "-mes    | --max_eval_steps"
    echo  "-w      | --num_workers_total"
    echo  "-p      | --packed_data       Packed (0) or unpacked (1)"
    echo  "-sch    | --save_checkpoints_steps"
    echo  "-pc     | --profile_config   Profiling configuration file for merged HW trace (defaults to ../../profile/synprof_mergedHW.json)"
    echo  "-hwp    | --hw_profile_range TF hooks for HW profiler in <start_hook>:<end_hook> format"
    echo  "-sp     | --syn_profile      Enable (1) or disable (0) synapse logger"
    echo  "-tps    | --tf_profile_steps TF steps to profile in <start_step>,<end_step> format"
    echo  "-cpu    | --cpu_bind_type"
    echo  "-inputf | --input_files_dir"
    echo  "-evalf  | --eval_files_dir"
    echo  "-od     | --output_dir"
    echo  "-ckpt   | --initial_checkpoint"
    echo  "-config | --config_dir"
    echo  "-hls    | --hls_type"
    echo  "-tcp    | --mpi_tcp_include"
    echo  "-dram   | --use_dram_output"
    echo  "-lw     | --light_weight"
}
#echo  "-sws    | --start_warmup_steps"

# Parse command line options
unset __config
unset __hostfile
unset __use_horovod
unset __warmup_steps
unset __learning_rate
unset __stop_threshold
unset __num_accumul_steps
unset __train_batchsize
unset __eval_batchsize
unset __train_steps
#unset __start_warmup_steps
unset __lamb_beta_1
unset __lamb_beta_2
unset __epsilon
unset __lamb_weight_decay_rate
unset __lamb_lr_decay_poly_power
unset __samples_btw_eval
unset __samples_start_eval
unset __max_eval_steps
unset __num_workers_total
unset __packed_data
unset __save_checkpoints_steps
unset __profile_cfg
unset __hw_profile_range
unset __syn_profile
unset __tf_profile_steps
unset __cpu_bind_type
unset __input_files_dir
unset __eval_files_dir
unset __output_dir
unset __initial_checkpoint
unset __config_dir
unset __hls_type
unset __mpi_tcp_include
unset __use_dram_output
unset __light_weight

while [ -n "$1" ]; do
    case $1 in
    -c | --config )
        shift
        __config=$1
        ;;
    -hf | --hostfile)
        shift
        __hostfile=$1
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
    -u | --use_horovod )
        shift
        __use_horovod=$1
        ;;
    -ws | --warmup_steps )
        shift
        __warmup_steps=$1
        ;;
    -lr | --learning_rate )
        shift
        __learning_rate=$1
        ;;
    -st | --stop_threshold )
        shift
        __stop_threshold=$1
        ;;
    -acs | --num_accumul_steps )
        shift
        __num_accumul_steps=$1
        ;;
    -tbs | --train_batchsize )
        shift
        __train_batchsize=$1
        ;;
    -ebs | --eval_batchsize)
        shift
        __eval_batchsize=$1
        ;;
    -ts | --train_steps)
        shift
        __train_steps=$1
        ;;
 #       -sws | --start_warmup_steps)
 #           shift
 #           __start_warmup_steps=$1
 #           ;;
    -lb1 | --lamb_beta_1)
        shift
        __lamb_beta_1=$1
        ;;
    -lb2 | --lamb_beta_2)
        shift
        __lamb_beta_2=$1
        ;;
    -ep  | --epsilon)
        shift
        __epsilon=$1
        ;;
    -lwd | --lamb_weight_decay_rate)
        shift
        __lamb_weight_decay_rate=$1
        ;;
    -ldp | --lamb_lr_decay_poly_power)
        shift
        __lamb_lr_decay_poly_power=$1
        ;;
    -sbe | --samples_btw_eval)
        shift
        __samples_btw_eval=$1
        ;;
    -sse | --samples_start_eval)
        shift
        __samples_start_eval=$1
        ;;
    -mes | --max_eval_steps)
        shift
        __max_eval_steps=$1
        ;;
    -w | --num_workers_total)
        shift
        __num_workers_total=$1
        ;;
    -p | --packed_data)
        shift
        __packed_data=$1
        ;;
    -sch | --save_checkpoints_steps)
        shift
        __save_checkpoints_steps=$1
        ;;
    -pc | --profile_config )
        shift
        __profile_cfg=$1
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
    -cpu | --cpu_bind_type)
        shift
        __cpu_bind_type=$1
        ;;
    -inputf | --input_files_dir)
        shift
        __input_files_dir=$1
        ;;
    -evalf | --eval_files_dir)
        shift
        __eval_files_dir=$1
        ;;
    -od | --output_dir)
        shift
        __output_dir=$1
        ;;
    -ckpt | --initial_checkpoint)
        shift
        __initial_checkpoint=$1
        ;;
    -config | --config_dir)
        shift
        __config_dir=$1
        ;;
    -hls | --hls_type)
        shift
        __hls_type=$1
        ;;
    -tcp | --mpi_tcp_include)
        shift
        __mpi_tcp_include=$1
        ;;
    -dram | --use_dram_output)
        shift
        __use_dram_output=$1
        ;;
    -lw  | --light_weight)
        shift
	__light_weight=$1
	;;
    -h | --help)
        help
        exit 1
        ;;
    * )
        echo "The parameter $1 is not allowed"
        help
	    exit 1
        ;;
    esac
    shift
done

export CFG_FILE=${__config:-"${BASE_PATH}/defaults.cfg"}
if [[ -f ${CFG_FILE} ]]; then
	source ${CFG_FILE}
else
        echo "Could not find ${CFG_FILE}"
        exit 1
fi

# Set default values for environmental variable
export HOST_FILE=${__hostfile:-"/root/shared/hosts"}
export PROFILE_CONFIG_FILE=${__profile_cfg:-"synprof_mergedHW.json"}
export PROFILE_CONFIG="${BASE_PATH}/../../../profile/${PROFILE_CONFIG_FILE}"

export EVAL_FILES_DIR=${EVAL_FILES_DIR}
export OUTPUT_DIR=${OUTPUT_DIR}
export PHASE1_CKPT=${INITIAL_CHECKPOINT}
export INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT}
export BERT_CONFIG_DIR=${BERT_CONFIG_DIR}
export NUM_WORKERS_PER_HLS=${NUM_WORKERS_PER_HLS}
export ENABLE_EXPERIMENTAL_FLAGS=${ENABLE_EXPERIMENTAL_FLAGS}
export OPTIMIZE_DMA_ENGINES_ALLOCATION=${OPTIMIZE_DMA_ENGINES_ALLOCATION}
export RUN_TPC_FUSER=${RUN_TPC_FUSER}
export TF_CPU_RUNTIME_FALLBACK=${TF_CPU_RUNTIME_FALLBACK}
export TF_HCCL_MEMORY_ALLOWANCE_MB=${TF_HCCL_MEMORY_ALLOWANCE_MB}
export HABANA_INITIAL_WORKSPACE_SIZE_MB=${HABANA_INITIAL_WORKSPACE_SIZE_MB}

# Override defaults with command line options if needed
export MPI_TCP_INCLUDE=${__mpi_tcp_include:-$MPI_TCP_INCLUDE}
export USE_HOROVOD=${__use_horovod:-$USE_HOROVOD}
export WARMUP_STEPS=${__warmup_steps:-$WARMUP_STEPS}
export LEARNING_RATE=${__learning_rate:-$LEARNING_RATE}
export STOP_THRESHOLD=${__stop_threshold:-$STOP_THRESHOLD}
export NUM_ACCUMULATION_STEPS=${__num_accumul_steps:-$NUM_ACCUMULATION_STEPS}
export TRAIN_BATCH_SIZE=${__train_batchsize:-$TRAIN_BATCH_SIZE}
export EVAL_BATCH_SIZE=${__eval_batchsize:-$EVAL_BATCH_SIZE}
export TRAIN_STEPS=${__train_steps:-$TRAIN_STEPS}
#export START_WARMUP_STEPS=${__start_warmup_steps:-$START_WARMUP_STEPS}
export LAMB_BETA_1=${__lamb_beta_1:-$LAMB_BETA_1}
export LAMB_BETA_2=${__lamb_beta_2:-$LAMB_BETA_2}
export EPSILON=${__epsilon:-$EPSILON}
export LAMB_WEIGHT_DECAY_RATE=${__lamb_weight_decay_rate:-$LAMB_WEIGHT_DECAY_RATE}
export LAMB_LEARNING_RATE_DECAY_POLY_POWER=${__lamb_lr_decay_poly_power:-$LAMB_LEARNING_RATE_DECAY_POLY_POWER}
export SAMPLES_START_EVAL=${__samples_start_eval:-$SAMPLES_START_EVAL}
export MAX_EVAL_STEPS=${__max_eval_steps:-$MAX_EVAL_STEPS}
export NUM_WORKERS_TOTAL=${__num_workers_total:-$NUM_WORKERS_TOTAL}
export PACKED_DATA=${__packed_data:-$PACKED_DATA}
export SAVE_CHECKPOINTS_STEPS=${__save_checkpoints_steps:-$SAVE_CHECKPOINTS_STEPS}
SAMPLES_BETWEEN_EVAL=$(($TRAIN_BATCH_SIZE*$NUM_WORKERS_TOTAL*$NUM_ACCUMULATION_STEPS*$SAVE_CHECKPOINTS_STEPS))
export SAMPLES_BETWEEN_EVAL=${__samples_btw_eval:-$SAMPLES_BETWEEN_EVAL}
export TF_PROFILE_STEPS=${__tf_profile_steps:-"$TF_PROFILE_STEPS"}
export HW_PROFILE_RANGE=${__hw_profile_range:-"$HW_PROFILE_RANGE"}
export SYN_PROFILE=${__syn_profile:-"$SYN_PROFILE"}
export CPU_BIND_TYPE=${__workload_to_cpu_pin_type:-$CPU_BIND_TYPE}
export EVAL_FILES_DIR=${__eval_files_dir:-$EVAL_FILES_DIR}
export OUTPUT_DIR=${__output_dir:-$OUTPUT_DIR}
export PHASE1_CKPT=${__initial_checkpoint:-$INITIAL_CHECKPOINT}
export BERT_CONFIG_DIR=${__config_dir:-$BERT_CONFIG_DIR}
export HLS_TYPE=${__hls_type:-$HLS_TYPE}
export USE_DRAM_OUTPUT=${__use_dram_output:-"True"}
export USE_LIGHTWEIGHT_CHECKPOINT=${__light_weight:-$USE_LIGHTWEIGHT_CHECKPOINT}

if [ "$HLS_TYPE" == "HLS1-H" ]; then
   export NUM_WORKERS_PER_HLS=4
elif [ "$HLS_TYPE" == "HLS1" ]; then
   export NUM_WORKERS_PER_HLS=8
elif [ "$HLS_TYPE" == "OCP1" ]; then
   export NUM_WORKERS_PER_HLS=8
else
   "============== WRONG HLS TYPE!! ==============="
   exit -1
fi

if [ "$PACKED_DATA" == "False" ]; then
   export INPUT_FILES_DIR=${__input_files_dir:-$INPUT_FILES_DIR_UNPACKED}
else
   export INPUT_FILES_DIR=${__input_files_dir:-$INPUT_FILES_DIR_PACKED}
fi

if [ "$USE_HOROVOD" == "True" ]; then
    export HOROVOD_STALL_CHECK_DISABLE=1
    echo HOROVOD_STALL_CHECK_DISABLE=$HOROVOD_STALL_CHECK_DISABLE

    # HCL Streams:ON by default
    export HABANA_USE_STREAMS_FOR_HCL=${HABANA_USE_STREAMS_FOR_HCL:-True}
    echo HABANA_USE_STREAMS_FOR_HCL=${HABANA_USE_STREAMS_FOR_HCL}

    # ART:ON by default
    export HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=${HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE:-False}
    echo HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE=${HABANA_USE_PREALLOC_BUFFER_FOR_ALLREDUCE}

    # SAO:ON by default
    export TF_DISABLE_SCOPED_ALLOCATOR=${TF_DISABLE_SCOPED_ALLOCATOR:-False}
    echo TF_DISABLE_SCOPED_ALLOCATOR=$TF_DISABLE_SCOPED_ALLOCATOR
fi

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

getmulti_hls_ips ${HOST_FILE}

# Create recipes directory if it does not exist and adjust dirctory name
# if we are collecting traces - which require debug information
run_per_ip mkdir -p ${OUTPUT_DIR} # 2>/dev/null
run_per_ip rm -rf ${OUTPUT_DIR}/* # 2>/dev/null

run_per_ip 'cd ${BASE_PATH} && git clone https://github.com/mlperf/logging.git mlperf-logging'
run_per_ip 'cd ${BASE_PATH} && python3 -m pip install -e mlperf-logging'

#run_per_ip rm -rf /tmp/checkpoint /tmp/eval /tmp/events.out.tfevents.* /tmp/graph.pbtxt /tmp/model.ckpt-*
#run_per_ip rm -rf /tmp/rank_*/checkpoint /tmp/rank_*/eval /tmp/rank_*/events.out.tfevents.* /tmp/rank_*/graph.pbtxt /tmp/rank_*/model.ckpt-*

function setup_libjemalloc()
{
	local libjemalloc_1_lib="libjemalloc.so.1"
	local libjemalloc_2_lib="libjemalloc.so.2"
	local is_v2_not_present=`LD_PRELOAD=${libjemalloc_2_lib} head -0 2>&1 > /dev/null`

	if [ -z "${is_v2_not_present}" ]; then
	    export LD_PRELOAD=${libjemalloc_2_lib}:$LD_PRELOAD
	else
	    export LD_PRELOAD=${libjemalloc_1_lib}:$LD_PRELOAD
        fi
}
run_per_ip setup_libjemalloc

if [[  -z ${MULTI_HLS_IPS} ]]; then
    echo "[launch_bert_hvd] MULTI_HLS_IPS undefined - maybe a missing /root/shared/hosts file?"
    exit -1
else
    run_per_ip generate_hcl_config '${OUTPUT_DIR}' ${NUM_WORKERS_PER_HLS} ${HLS_TYPE} #&> /dev/null
    generate_hcl_config ${OUTPUT_DIR} ${NUM_WORKERS_PER_HLS} ${HLS_TYPE}

    IFS=',' read -ra IPS <<< "$MULTI_HLS_IPS"
    let MPI_NP=${#IPS[@]}*${NUM_WORKERS_PER_HLS}
    export NUM_WORKERS_TOTAL=${NUM_WORKERS_TOTAL:-$MPI_NP}

    if [[ $NUM_WORKERS_TOTAL != $MPI_NP ]]; then
       echo $NUM_WORKERS_TOTAL $MPI_NP
       echo "===============   WRONG NUMBER_WORKERS_TOTAL!!   ==============="
       exit -1
    fi

    echo NUM_WORKERS_TOTAL=$NUM_WORKERS_TOTAL

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

    generate_mpi_hostfile ${OUTPUT_DIR} ${NUM_WORKERS_PER_HLS}

    export testdate=`date +%Y-%m-%d`
    export testtime=`date +%H%M%S`
    export OUTPUT_DIR=${__output_dir:-/root/scratch/bert/bert_gaudi${NUM_WORKERS_TOTAL}_${testdate}_${testtime}}
    LOGFILE=$OUTPUT_DIR/tf_bert_pretraining_lamb.log
    export TF_RECIPE_CACHE_PATH=/tmp/bert_pretrain/phase_2
    run_per_ip mkdir -p $TF_RECIPE_CACHE_PATH

    MPI_MAP_BY=socket
    MPI_MAP_BY_PE=`lscpu | grep "^CPU(s):"| awk -v NUM=${NUM_WORKERS_PER_HLS} '{print int($2/NUM/2)}'`
    if [[ "$CPU_BIND_TYPE" == "numa" ||  "$CPU_BIND_TYPE" == "none" ]]; then
        MPIRUN_ARGS_MAP_BY_PE="-bind-to none"
    else
        MPIRUN_ARGS_MAP_BY_PE="--bind-to core --map-by $MPI_MAP_BY:PE=$MPI_MAP_BY_PE"
    fi

    TRAINING_COMMAND="mpirun --allow-run-as-root \
        --display-map \
        --report-bindings \
        --bind-to none \
        -np ${NUM_WORKERS_TOTAL}\
        --hostfile ${MPI_HOSTFILE_PATH} \
        --prefix ${OMPI_PREFIX} \
        --mca plm_rsh_args -p3022 \
        --mca btl_tcp_if_include ${MPI_TCP_INCLUDE} \
        --merge-stderr-to-stdout \
        --output-filename ${OUTPUT_DIR}/bert_log \
        -x USE_HOROVOD=${USE_HOROVOD} \
        -x HCL_CONFIG_PATH=${HCL_CONFIG_PATH} \
        -x TF_MODULES_RELEASE_BUILD=/usr/lib/habanalabs/ \
        -x GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
        -x HABANA_LOGS=/var/log/habana_logs/ \
        -x LEARNING_RATE=${LEARNING_RATE} \
        -x STOP_THRESHOLD=${STOP_THRESHOLD} \
        -x NUM_ACCUMULATION_STEPS=${NUM_ACCUMULATION_STEPS} \
        -x TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
        -x EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE} \
        -x TRAIN_STEPS=${TRAIN_STEPS} \
        -x NUM_WORKERS_TOTAL=${NUM_WORKERS_TOTAL} \
        -x WARMUP_STEPS=${WARMUP_STEPS} \
        -x LAMB_BETA_1=${LAMB_BETA_1} \
        -x LAMB_BETA_2=${LAMB_BETA_2} \
        -x EPSILON=${EPSILON} \
        -x LAMB_WEIGHT_DECAY_RATE=${LAMB_WEIGHT_DECAY_RATE} \
        -x LAMB_LEARNING_RATE_DECAY_POLY_POWER=${LAMB_LEARNING_RATE_DECAY_POLY_POWER} \
        -x SAMPLES_BETWEEN_EVAL=${SAMPLES_BETWEEN_EVAL} \
        -x SAMPLES_START_EVAL=${SAMPLES_START_EVAL} \
        -x MAX_EVAL_STEPS=${MAX_EVAL_STEPS} \
        -x INPUT_FILES_DIR=${INPUT_FILES_DIR} \
        -x EVAL_FILES_DIR=${EVAL_FILES_DIR} \
        -x OUTPUT_DIR=${OUTPUT_DIR} \
        -x PHASE1_CKPT=${PHASE1_CKPT} \
        -x BERT_CONFIG_DIR=${BERT_CONFIG_DIR} \
        -x ENABLE_EXPERIMENTAL_FLAGS=${ENABLE_EXPERIMENTAL_FLAGS} \
        -x OPTIMIZE_DMA_ENGINES_ALLOCATION=${OPTIMIZE_DMA_ENGINES_ALLOCATION} \
        -x RUN_TPC_FUSER=${RUN_TPC_FUSER} \
        -x TF_CPU_RUNTIME_FALLBACK=${TF_CPU_RUNTIME_FALLBACK} \
        -x TF_HCCL_MEMORY_ALLOWANCE_MB=${TF_HCCL_MEMORY_ALLOWANCE_MB} \
        -x HABANA_INITIAL_WORKSPACE_SIZE_MB=${HABANA_INITIAL_WORKSPACE_SIZE_MB} \
        -x HLS_TYPE=${HLS_TYPE} \
        -x MPI_TCP_INCLUDE=${MPI_TCP_INCLUDE} \
        -x SAVE_CHECKPOINTS_STEPS=${SAVE_CHECKPOINTS_STEPS} \
        -x PACKED_DATA=${PACKED_DATA} \
        -x TESTDATE=${testdate} \
        -x TESTTIME=${testtime} \
        -x PROFILE_CONFIG=${PROFILE_CONFIG} \
        -x SYN_PROFILE=${SYN_PROFILE} \
        -x TF_PROFILE_STEPS=${TF_PROFILE_STEPS} \
        -x HW_PROFILE_RANGE=${HW_PROFILE_RANGE} \
        -x CPU_BIND_TYPE=${CPU_BIND_TYPE} \
        -x NUM_WORKERS_PER_HLS=${NUM_WORKERS_PER_HLS} \
        ${MPIRUN_ARGS_MAP_BY_PE} \
        -x USE_DRAM_OUTPUT=${USE_DRAM_OUTPUT} \
        -x USE_LIGHTWEIGHT_CHECKPOINT=${USE_LIGHTWEIGHT_CHECKPOINT} \
        -x TF_ENABLE_DYNAMIC_SHAPES=${TF_ENABLE_DYNAMIC_SHAPES} \
        ${BASE_PATH}/run.sh"

    echo "TRAINING COMMAND = ${TRAINING_COMMAND}"
    printf "[launch_bert_hvd] Starting training...\n\n"
    time $TRAINING_COMMAND |& tee -a $LOGFILE
fi
run_per_ip rm -rf $OUTPUT_DIR/*/model.ckpt-*
# -x START_WARMUP_STEPS=${START_WARMUP_STEPS} \
rm -rf $BASE_PATH/log
cp /root/build_log.csv ${OUTPUT_DIR}/
cp ${MPI_HOSTFILE_PATH} ${OUTPUT_DIR}/
cp -r $OUTPUT_DIR/bert_log $BASE_PATH/log
chmod -R 777 ${OUTPUT_DIR}
exit $exit_code
