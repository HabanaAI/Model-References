#!/bin/bash
# USAGE1: HOST_LIST=#<dot separated 16 ips> ./run_hvd_16gaudi.sh
# For usage1, the "IP1" and "IP2" in the file "hcl_config.16cards.json" need to
# be replaced by the 2 host ip addresses
#
# USAGE2: ./run_hvd_16gaudi.sh #<ip1> #<ip2>

export INPUT_IPS=$@
export SSHD_PORT=${SSHD_PORT:-3022}
export HCL_CONFIG_PATH=${HCL_CONFIG_PATH:-${SCRIPT_DIR}/hcl_config.16cards.json}

SCRIPT_DIR=`readlink -e ${BASH_SOURCE} | xargs -I {} dirname {}`
source ${SCRIPT_DIR}/hvd_common.sh
MPI_PREFIX=${MPI_PREFIX:-/usr/local/openmpi/}

if [ ! -z $1 ] && [ -z $HOST_LIST ]; then
    gen_host_ip_list $INPUT_IPS
    gen_multinode_hclconfig $INPUT_IPS
fi
gen_if_include_list $HOST_LIST

echo "Using config file $HCL_CONFIG_PATH"

if [ ! -z ${MPI_TCP_INCLUDE} ]; then
    EXTRA_MPI_ARGS="${EXTRA_MPI_ARGS} --mca btl_tcp_if_include ${MPI_TCP_INCLUDE}"
fi

for e in `env |grep -i 'habanalabs\|^HABANA\|_proxy'`; do
    VARNAME=`echo $e | tr '=' ' ' | awk '{print $1}'`
    EXTRA_MPI_ARGS="${EXTRA_MPI_ARGS} -x $VARNAME"
done

mpirun --allow-run-as-root ${EXTRA_MPI_ARGS} -host $HOST_LIST \
    --mca plm_rsh_args -p${SSHD_PORT} --prefix ${MPI_PREFIX} \
    -x PATH -x PYTHONPATH -x HCL_CONFIG_PATH \
    python3 ${SCRIPT_DIR}/example_hvd.py
