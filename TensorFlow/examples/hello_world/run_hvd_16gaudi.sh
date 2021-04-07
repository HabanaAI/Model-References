#!/bin/bash
# USAGE1: HOST_LIST=#<dot separated 16 ips> ./run_hvd_16gaudi.sh
# For usage1, the "IP1" and "IP2" in the file "hcl_config.16cards.json" need to
# be replaced by the 2 host ip addresses
#
# USAGE2: ./run_hvd_16gaudi.sh #<ip1> #<ip2>

export INPUT_IPS=$@
source `readlink -e ${BASH_SOURCE} | xargs -I {} dirname {}`/hvd_common.sh

if [ ! -z $1 ] && [ -z $HOST_LIST ]; then
    gen_host_ip_list $INPUT_IPS
    gen_multinode_hclconfig $INPUT_IPS
fi

export HCL_CONFIG_PATH=${HCL_CONFIG_PATH:-`pwd`/hcl_config.16cards.json}
echo "Using config file $HCL_CONFIG_PATH"

mpirun --allow-run-as-root --mca btl_tcp_if_exclude lo,docker -host $HOST_LIST \
    -x PATH -x HCL_CONFIG_PATH \
    python3 example_hvd.py
