#!/bin/bash

function print_error()
{
    >&2 printf "\033[0;31mError: $1\n\033[0m"
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
        echo "Data-preloading feature has been requested, but cannot find ${dynpatch_name}"
        exit 1
    fi
    export LD_PRELOAD="${dynpatch_path}:${LD_PRELOAD}"
    export HBN_TF_REGISTER_DATASETOPS=1
}


function generate_hcl_config()
{
    echo "Generating HCL Config..."
    local file_name="config_$2_local.json"
    export HCL_CONFIG_PATH=$1/${file_name}
    rm -f ${HCL_CONFIG_PATH}
    echo "Path: ${HCL_CONFIG_PATH}"
    touch ${HCL_CONFIG_PATH}
    echo "{"                                             >> ${HCL_CONFIG_PATH}
    echo "    \"HCL_PORT\": 5332,"                       >> ${HCL_CONFIG_PATH}
    echo "    \"HCL_TYPE\": \"HLS1\","                   >> ${HCL_CONFIG_PATH}
    if [[ -z ${MULTI_HLS_IPS} ]]; then
        echo "    \"HCL_COUNT\": $2"                     >> ${HCL_CONFIG_PATH}
    else
        echo "    \"HCL_RANKS\": ["                      >> ${HCL_CONFIG_PATH}
        IFS=',' read -ra IPS <<< "$MULTI_HLS_IPS"
        for i in "${IPS[@]}"; do
            echo "        \"$i\","                       >> ${HCL_CONFIG_PATH}
            echo "        \"$i\","                       >> ${HCL_CONFIG_PATH}
            echo "        \"$i\","                       >> ${HCL_CONFIG_PATH}
            echo "        \"$i\","                       >> ${HCL_CONFIG_PATH}
            echo "        \"$i\","                       >> ${HCL_CONFIG_PATH}
            echo "        \"$i\","                       >> ${HCL_CONFIG_PATH}
            echo "        \"$i\","                       >> ${HCL_CONFIG_PATH}
            if [[ $i == ${IPS[${#IPS[@]}-1]} ]]; then
                echo "        \"$i\""                    >> ${HCL_CONFIG_PATH}
            else
                echo "        \"$i\","                   >> ${HCL_CONFIG_PATH}
            fi
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
    local file_name="hostfile"
    export MPI_HOSTFILE_PATH=$1/${file_name}

    rm -rf ${MPI_HOSTFILE_PATH}
    echo "PATH: ${MPI_HOSTFILE_PATH}"
    touch ${MPI_HOSTFILE_PATH}

    IFS=',' read -ra IPS <<< "$MULTI_HLS_IPS"
    for i in "${IPS[@]}"; do
        echo "$i slots=8" >> ${MPI_HOSTFILE_PATH}
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
        mpirun --tag-output --merge-stderr-to-stdout --prefix ${HOME}/.openmpi/ -H ${MULTI_HLS_IPS} bash -c "`declare`; ($_cmd 2>&1)" 2>/dev/null
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
