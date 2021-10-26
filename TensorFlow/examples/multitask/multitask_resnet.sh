SCRIPT_PATH=`readlink -e ${BASH_SOURCE[0]}`
SCRIPT_DIR=`dirname ${SCRIPT_PATH}`
DEMO_SCRIPT=${SCRIPT_DIR}/../../computer_vision/Resnets/resnet_keras/demo_resnet_keras.py
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
        echo "Usage of function run(): run #<HLS1_MODULE_ID_LIST> #<MODEL_DIR> #<STDOUT_LOG> #<STDERR_LOG> "
        return
    fi
    export HLS1_MODULE_ID_LIST=$1
    MODEL_DIR=$2
    STDOUT_LOG=$3
    STDERR_LOG=$4
    NUM=$((`echo $HLS1_MODULE_ID_LIST | wc -c` - 1))

    python3 ${DEMO_SCRIPT} --dtype bf16 --data_loader_image_type bf16 --use_horovod \
        --num_workers_per_hls $NUM -te $TRAIN_EPOCHS -bs 256 --optimizer LARS --experimental_preloading \
        -dd $DATA_DIR -md $MODEL_DIR 1> $STDOUT_LOG 2> $STDERR_LOG &
    echo "Job ${JOB_ID} starts with ${NUM} cards, stdout: ${STDOUT_LOG}, stderr: ${STDERR_LOG}"
    JOB_ID=$((JOB_ID+1))
    export JOB_WAIT_LIST="$JOB_WAIT_LIST $!"
}

run 0123 /tmp/resnet_keras_lars/1/ job1.log job1.err
run 4567 /tmp/resnet_keras_lars/2/ job2.log job2.err

echo "=======================Training in background======================="

for pid in $JOB_WAIT_LIST; do
    wait $pid
done

echo "================================Done================================"
echo "Logs are saved in files job1.log and job2.log for the 2 workloads respectively"
