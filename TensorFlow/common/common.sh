#!/bin/bash
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

function print_error()
{
    >&2 printf "\033[0;31mError: $1\n\033[0m"
}

function print_warning()
{
    >&2 printf "\033[0;33mWarning: $1\n\033[0m"
}

function find_library()
{
    local name="${1}"
    local path
    local libpath
    local LOCATIONS

    LOCATIONS="${LD_LIBRARY_PATH}:${BUILD_ROOT_LATEST}:${TF_MODULES_RELEASE_BUILD}:${TF_MODULES_DEBUG_BUILD}"

    OLD_IFS="${IFS}"
    IFS=":"
    for path in ${LOCATIONS}; do
        if [ ! -z "${path}" ]; then
            libpath="${path}/${name}"
            if [ -e "${libpath}" ]; then
                readlink -f "${libpath}"
                break
            fi
        fi
    done
    IFS="${OLD_IFS}"
}


function setup_preloading()
{
    local dynpatch_name="dynpatch_prf_remote_call.so"
    local dynpatch_path=`find_library ${dynpatch_name}`
    if [ -z "${dynpatch_path}" ]; then
        echo "Preloading lib ${dynpatch_name} not found in the system. Searching in wheel.."
        dynpatch_path=$(python3 -c "import habana_frameworks.tensorflow as htf; print(htf.sysconfig.get_lib_dir())")/${dynpatch_name}
        if [ -z "${dynpatch_path}" ]; then
            echo "Data-preloading feature has been requested, but cannot find ${dynpatch_name} in wheel neither.."
            exit 1
        fi
    fi
    export LD_PRELOAD="${dynpatch_path}:${LD_PRELOAD}"
    export HBN_TF_REGISTER_DATASETOPS=1
}

#######################################
# Generate hcl config file and set HCL_CONFIG_PATH
# Globals:
#   MULTI_HLS_IPS
# Arguments:
#   Path to save the file
#   Number of devices per HLS
#   HLS type, default HLS1
#######################################
function generate_hcl_config()
{
    if [ -z ${HCL_CONFIG_PATH} ]; then
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
    else
        echo "HCL Config Path: ${HCL_CONFIG_PATH}"
        echo "Config: "
        cat ${HCL_CONFIG_PATH}
    fi
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
            --prefix $MPI_ROOT \
            -H ${MULTI_HLS_IPS} \
            bash -c "`declare`; `declare -x`; ($_cmd 2>&1)" 2>/dev/null
    fi
}

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

function calc_optimal_cpu_resources_for_mpi()
{
    # OpenMPI process bind resource type.
    export MPI_MAP_BY=${MPI_MAP_BY:-"socket"}
    echo MPI_MAP_BY=$MPI_MAP_BY

    # Determine the optimal value of resources per process of OpenMPI binding based on local lscpu.
    if [ "$MPI_MAP_BY" == "socket" ]; then
        __mpi_map_by_pe=`lscpu | grep "CPU(s):" | python3 -c "print(int(input().split()[1])//${NUM_WORKERS_PER_HLS}//2)"`
    elif [ "$MPI_MAP_BY" == "slot" ]; then
        __mpi_map_by_pe=`lscpu | grep "CPU(s):" | python3 -c "print(int(input().split()[1])//${NUM_WORKERS_PER_HLS})"`
    else
        print_error "MPI_MAP_BY must be either 'socket' or 'slot'."
        exit 1;
    fi
    export MPI_MAP_BY_PE=${MPI_MAP_BY_PE:-$__mpi_map_by_pe}
    echo MPI_MAP_BY_PE=$MPI_MAP_BY_PE

    if [ "$MPI_MAP_BY_PE" -gt "0" ]; then
        __mpirun_args_map_by_pe=" --bind-to core --map-by $MPI_MAP_BY:PE=$MPI_MAP_BY_PE"
    else
        unset __mpirun_args_map_by_pe
    fi
    export MPIRUN_ARGS_MAP_BY_PE=$__mpirun_args_map_by_pe
}