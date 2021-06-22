#!/bin/bash
# USAGE1: HOST_LIST=#<dot separated 16 ips> ./run_hvd_16gaudi.sh
# USAGE2: ./run_hvd_16gaudi.sh #<ip1> #<ip2>

export INPUT_IPS=$@
export HOROVOD_HIERARCHICAL_ALLREDUCE=1
source `readlink -e ${BASH_SOURCE} | xargs -I {} dirname {}`/hvd_common.sh

if [ ! -z $1 ] && [ -z $HOST_LIST ]; then
    gen_host_ip_list $INPUT_IPS
fi

export HCL_CONFIG_PATH=${HCL_CONFIG_PATH:-`pwd`/hcl_config.json}
echo "Using config file $HCL_CONFIG_PATH"

mpirun --allow-run-as-root --mca btl_tcp_if_exclude lo,docker -host $HOST_LIST \
    -x PATH -x HCL_CONFIG_PATH -x HOROVOD_HIERARCHICAL_ALLREDUCE \
    python3 example_hvd.py
