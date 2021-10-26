#!/bin/bash
# USAGE ./run_multi_hvd_4_4.sh

function run_job()
{
    export HLS1_MODULE_ID_LIST=$1
    MPI_LOG=$2
    if [ "$#" -lt 2 ] ; then
        echo "not enough arguments for function run_job!!!"
        exit 1
    fi
    NUM=$((`echo ${HLS1_MODULE_ID_LIST} | wc -c` - 1))
    SCRIPT_PATH=`readlink -e ${BASH_SOURCE[0]}`
    SCRIPT_DIR=`dirname ${SCRIPT_PATH}`

    mpirun --allow-run-as-root --merge-stderr-to-stdout -np $NUM --output-filename $MPI_LOG \
        -x HLS1_MODULE_ID_LIST -x HCL_CONFIG_PATH python3 ${SCRIPT_DIR}/example_hvd.py&
    export JOB_WAIT_LIST="$JOB_WAIT_LIST $!"
}

# USAGE: run_job #<HLS1_MODULE_ID_LIST> #<MPI_LOG_DIR>
run_job 0123 log.job1
run_job 4567 log.job2

for pid in $JOB_WAIT_LIST; do
    wait $pid
done

echo "================================Done================================"
echo "Logs are saved in folder log.job1 and log.job2 for the 2 workloads respectively"