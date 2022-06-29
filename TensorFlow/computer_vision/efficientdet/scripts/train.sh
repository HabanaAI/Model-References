#!/usr/bin/env bash
num_gpus=$1

#export NCCL_P2P_DISABLE=1
export PYTHONPATH=`pwd`:$PYTHONPATH

mpirun -np $num_gpus -H localhost:$num_gpus \
--allow-run-as-root -bind-to none -map-by slot -x NCCL_DEBUG=INFO \
-x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib \
--mca btl_smcuda_use_cuda_ipc 0 \
python main.py --training_file_pattern=/share/liupeng/data/cv/coco/coco_tfrecord/train* \
    --model_name=$MODEL \
    --model_dir=/tmp/$MODEL \
    --hparams="use_bfloat16=false" \
    --use_tpu=False \
    --train_batch_size 4
