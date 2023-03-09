#!/bin/bash
SCRIPT_DIR=`dirname $(readlink -e ${BASH_SOURCE[0]})`
TRAIN_EPOCHS=${TRAIN_EPOCHS:-40}

DATA_DIR=/root/software/lfs/data/pytorch/imagenet/ILSVRC2012/

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
    export MASTER_PORT=$5
    export MASTER_ADDR=127.0.0.1
    export DATALOADER_FALLBACK_EN=False
    MODEL_DIR=$2
    STDOUT_LOG=$3
    STDERR_LOG=$4
    NUM=8
    MAX_CORE_PER_PROC=$(($(lscpu|grep "^CPU(s):" | tr -s ' '|cut -d ' ' -f 2) / $NUM / 2))
    CORE_PER_PROC=${CORE_PER_PROC:-${MAX_CORE_PER_PROC}}
    CORE_PER_PROC=$((${CORE_PER_PROC} > ${MAX_CORE_PER_PROC} ? ${MAX_CORE_PER_PROC} : ${CORE_PER_PROC}))

    DEMO_SCRIPT=/root/Model-References/PyTorch/computer_vision/classification/torchvision/train.py


    if [ "$1" = "0,1,2,3" ]; then

   mpirun -n 4 --bind-to core --map-by socket:PE=6  --rank-by core --report-bindings --allow-run-as-root python3 /root/Model-References/PyTorch/computer_vision/classification/torchvision/train.py --batch-size 256 --model resnet50 --device hpu --workers 8 --print-freq 10 --channels-last False --dl-time-exclude False --deterministic --num-eval-steps 1 --data-path /root/software/lfs/data/pytorch/imagenet/ILSVRC2012 --num-train-steps 500 --epochs 3 --hmp --hmp-bf16 /root/Model-References/PyTorch/computer_vision/classification/torchvision/ops_bf16_Resnet.txt --hmp-fp32 /root/Model-References/PyTorch/computer_vision/classification/torchvision/ops_fp32_Resnet.txt --dl-worker-type HABANA 1> $STDOUT_LOG 2> $STDERR_LOG &

    else
	      mpirun -n 4 --bind-to core --map-by socket:PE=6  --rank-by core --report-bindings --allow-run-as-root python3 /root/Model-References/PyTorch/computer_vision/classification/torchvision/train.py --batch-size 256 --model resnet50 --device hpu --workers 8 --print-freq 10 --channels-last False --dl-time-exclude False --deterministic --num-eval-steps 1 --data-path /root/software/lfs/data/pytorch/imagenet/ILSVRC2012 --num-train-steps 500 --epochs 3 --hmp --hmp-bf16 /root/Model-References/PyTorch/computer_vision/classification/torchvision/ops_bf16_Resnet.txt --hmp-fp32 /root/Model-References/PyTorch/computer_vision/classification/torchvision/ops_fp32_Resnet.txt --dl-worker-type HABANA 1> $STDOUT_LOG 2> $STDERR_LOG &

    fi

    echo "Job ${JOB_ID} starts with ${NUM} cards, stdout: ${STDOUT_LOG}, stderr: ${STDERR_LOG}"
    JOB_ID=$((JOB_ID+1))
    export JOB_WAIT_LIST="$JOB_WAIT_LIST $!"
}

PYTHON=python3
$PYTHON -m pip install -r /root/Model-References/PyTorch/computer_vision/classification/torchvision/requirements.txt

run "0,1,2,3" /tmp/resnet_keras_lars/1/ job1.log job1.err 12345
run "4,5,6,7" /tmp/resnet_keras_lars/2/ job2.log job2.err 12346

echo "=======================Training in background======================="

for pid in $JOB_WAIT_LIST; do
    wait $pid
done

echo "================================Done================================"
echo "Logs are saved in files job1.log and job2.log for the 2 workloads respectively"
