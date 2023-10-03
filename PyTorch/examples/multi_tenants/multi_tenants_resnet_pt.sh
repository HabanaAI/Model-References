#!/bin/bash
export MASTER_ADDR=localhost

SCRIPT_DIR=`dirname $(readlink -e ${BASH_SOURCE[0]})`
TRAIN_EPOCHS=1
WORKERS=10
MODEL=resnet50
LR_VALUES="0.275 0.45 0.625 0.8 0.08 0.008 0.0008"
LR_MILESTONES="1 2 3 4 30 60 80"

DATA_DIR=${DATA_DIR:=/data/pytorch/imagenet/ILSVRC2012}

if [ ! -d $DATA_DIR ]; then
    echo "Need to specify the ImageNet data in the env variable DATA_DIR"
    exit 1
fi

JOB_ID=1

function run() {
    if [ -z "$5" ]; then
        echo "5 arguments needed for function run()!!"
        echo "Usage of function run(): run #<HABANA_VISIBLE_MODULES> #<MASTER_PORT> #<MODEL_DIR> #<STDOUT_LOG> #<STDERR_LOG> "
        return
    fi
    export HABANA_VISIBLE_MODULES=$1
    export MASTER_PORT=$2
    MODEL_DIR=$3
    STDOUT_LOG=$4
    STDERR_LOG=$5
    NUM=$((`echo ${HABANA_VISIBLE_MODULES} | sed s/,/""/g | wc -c` - 1))
    MAX_CORE_PER_PROC=$(($(lscpu|grep "^CPU(s):" | tr -s ' '|cut -d ' ' -f 2) / $NUM / 2))
    CORE_PER_PROC=${CORE_PER_PROC:-${MAX_CORE_PER_PROC}}
    CORE_PER_PROC=$((${CORE_PER_PROC} > ${MAX_CORE_PER_PROC} ? ${MAX_CORE_PER_PROC} : ${CORE_PER_PROC}))

    DEMO_SCRIPT=${SCRIPT_DIR}/../../computer_vision/classification/torchvision/train.py
 
    # --map-by slot if VM is used
    mpirun --allow-run-as-root --bind-to core --rank-by core -np $NUM --map-by socket:PE=${CORE_PER_PROC} \
        --report-bindings  \
        $PYTHON ${DEMO_SCRIPT} \
        --model=${MODEL} \
        --device=hpu \
        --autocast \
        --epochs=${TRAIN_EPOCHS} \
        --workers=${WORKERS} \
        --dl-worker-type=MP \
        --batch-size=256 \
        --print-freq=10 \
        --data-path=${DATA_DIR} \
        --output-dir=. \
        --deterministic \
        --dl-time-exclude=False \
        --custom-lr-values ${LR_VALUES} \
        --custom-lr-milestones  ${LR_MILESTONES} \
        --seed=123 1> $STDOUT_LOG 2> $STDERR_LOG &

    echo "Job ${JOB_ID} starts with ${NUM} cards, stdout: ${STDOUT_LOG}, stderr: ${STDERR_LOG}"
    JOB_ID=$((JOB_ID+1))
    export JOB_WAIT_LIST="$JOB_WAIT_LIST $!"
}

if [ $# == 2 ]
then
    if echo $1 | grep -Eq '^[0-9]+(,[0-9]+)*$' && echo $2 | grep -Eq '^[0-9]+(,[0-9]+)*$'; then
        echo "User user input cards numbers"
        run $1 12355 /tmp/resnet/1/ job1.log job1.err
        run $2 12356 /tmp/resnet/2/ job2.log job2.err
    else
        echo "Wrong input arguments"
        exit 2
    fi
else
    echo "Using default card numbers"
    run "0,1,2,3" 12355 /tmp/resnet/1/ job1.log job1.err
    run "4,5,6,7" 12356 /tmp/resnet/2/ job2.log job2.err
fi

echo "=======================Training in background======================="

for pid in $JOB_WAIT_LIST; do
    wait $pid
done

echo "================================Done================================"
echo "Logs are saved in files job1.log and job2.log for the 2 workloads respectively"
