#!/bin/bash
# USAGE ./run_multi_hvd_4_4.sh

function run_job()
{
    export HABANA_VISIBLE_MODULES=$1
    MPI_LOG=$2
    if [ "$#" -lt 2 ] ; then
        echo "not enough arguments for function run_job!!!"
        exit 1
    fi
    NUM=$((`echo ${HABANA_VISIBLE_MODULES} | sed s/,/""/g | wc -c` - 1))
    SCRIPT_DIR=`dirname $(readlink -e ${BASH_SOURCE[0]})`

    mpirun --tag-output --allow-run-as-root --merge-stderr-to-stdout -np $NUM --output-filename $MPI_LOG \
        -x HABANA_VISIBLE_MODULES $PYTHON ${SCRIPT_DIR}/example_hvd.py&
    export JOB_WAIT_LIST="$JOB_WAIT_LIST $!"
}

# USAGE: run_job #<HABANA_VISIBLE_MODULES> #<MPI_LOG_DIR>
run_job "0,1,2,3" log.job1
run_job "4,5,6,7" log.job2

for pid in $JOB_WAIT_LIST; do
    wait $pid
done

echo "================================Done================================"
echo "Logs are saved in folder log.job1 and log.job2 for the 2 workloads respectively"