#!/bin/bash
SCRIPT_DIR=`dirname $(readlink -e ${BASH_SOURCE[0]})`
TRAIN_EPOCHS=${TRAIN_EPOCHS:-40}

DATA_DIR=${DATA_DIR:-/root/tf_records}

if [ ! -d $DATA_DIR ]; then
    echo "Need to specify the ImageNet data in the env variable DATA_DIR"
    exit 1
fi

JOB_ID=1

function run() {
    if [ -z "$4" ]; then
        echo "4 arguments needed for function run()!!"
        echo "Usage of function run(): run #<HABANA_VISIBLE_MODULES> #<MODEL_DIR> #<STDOUT_LOG> #<STDERR_LOG> "
        return
    fi
    export HABANA_VISIBLE_MODULES=$1
    MODEL_DIR=$2
    STDOUT_LOG=$3
    STDERR_LOG=$4
    NUM=$((`echo ${HABANA_VISIBLE_MODULES} | sed s/,/""/g | wc -c` - 1))
    MAX_CORE_PER_PROC=$(($(lscpu|grep "^CPU(s):" | tr -s ' '|cut -d ' ' -f 2) / $NUM / 2))
    CORE_PER_PROC=${CORE_PER_PROC:-${MAX_CORE_PER_PROC}}
    CORE_PER_PROC=$((${CORE_PER_PROC} > ${MAX_CORE_PER_PROC} ? ${MAX_CORE_PER_PROC} : ${CORE_PER_PROC}))

    DEMO_SCRIPT=${SCRIPT_DIR}/../../computer_vision/Resnets/resnet_keras/resnet_ctl_imagenet_main.py

    mpirun --allow-run-as-root --bind-to core -np $NUM --map-by socket:PE=${CORE_PER_PROC} \
        $PYTHON ${DEMO_SCRIPT} \
        --dtype bf16 \
        --data_loader_image_type bf16 \
        --use_horovod \
        -te ${TRAIN_EPOCHS} \
        -ebe ${TRAIN_EPOCHS} \
        -bs 256 \
        --optimizer LARS \
        --base_learning_rate 9.5 \
        --warmup_epochs 3 \
        --lr_schedule polynomial \
        --label_smoothing 0.1 \
        --weight_decay 0.0001 \
        --single_l2_loss_op \
        --enable_tensorboard=true \
        --data_dir ${DATA_DIR} \
        --model_dir ${MODEL_DIR} 1> $STDOUT_LOG 2> $STDERR_LOG &

    echo "Job ${JOB_ID} starts with ${NUM} cards, stdout: ${STDOUT_LOG}, stderr: ${STDERR_LOG}"
    JOB_ID=$((JOB_ID+1))
    export JOB_WAIT_LIST="$JOB_WAIT_LIST $!"
}

$PYTHON -m pip install -r ${SCRIPT_DIR}/../../computer_vision/Resnets/resnet_keras/requirements.txt

run "0,1,2,3" /tmp/resnet_keras_lars/1/ job1.log job1.err
run "4,5,6,7" /tmp/resnet_keras_lars/2/ job2.log job2.err

echo "=======================Training in background======================="

for pid in $JOB_WAIT_LIST; do
    wait $pid
done

echo "================================Done================================"
echo "Logs are saved in files job1.log and job2.log for the 2 workloads respectively"
