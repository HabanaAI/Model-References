#!/bin/bash
# USAGE1: HOST_LIST=#<dot separated 16 ips> ./run_hvd_16gaudi_hostnic.sh
# USAGE2: ./run_hvd_16gaudi_hostnic.sh #<ip1> #<ip2>

export INPUT_IPS=$@
export SSHD_PORT=${SSHD_PORT:-3022}

SCRIPT_DIR=`readlink -e ${BASH_SOURCE} | xargs -I {} dirname {}`
source ${SCRIPT_DIR}/hvd_common.sh
MPI_PREFIX=${MPI_PREFIX:-/usr/local/openmpi/}

if [ ! -z $1 ] && [ -z $HOST_LIST ]; then
    gen_host_ip_list $INPUT_IPS
fi
gen_if_include_list $HOST_LIST

if [ ! -z ${MPI_TCP_INCLUDE} ]; then
    EXTRA_MPI_ARGS="${EXTRA_MPI_ARGS} --mca btl_tcp_if_include ${MPI_TCP_INCLUDE}"
fi

for e in `env |grep -i 'habanalabs\|^HABANA\|_proxy'`; do
    VARNAME=`echo $e | tr '=' ' ' | awk '{print $1}'`
    EXTRA_MPI_ARGS="${EXTRA_MPI_ARGS} -x $VARNAME"
done

mpirun --allow-run-as-root ${EXTRA_MPI_ARGS} -host $HOST_LIST \
    --mca plm_rsh_args -p${SSHD_PORT} --prefix ${MPI_PREFIX} \
    -x PATH -x PYTHONPATH -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
    python3 ${SCRIPT_DIR}/example_hvd.py
