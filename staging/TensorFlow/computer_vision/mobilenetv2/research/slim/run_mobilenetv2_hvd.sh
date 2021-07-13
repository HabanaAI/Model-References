export DATASET_DIR=/software/data/tf/data/mobilenetv2/train
export NUM_GPUS=1
export TRAIN_DIR=/tmp/mobilenetv2
export LR=0.045
export NUM_EPOCHS_PER_DECAY=2.5
export HCL_CONFIG_PATH=$HOME/hcl_config.json
mpirun -np 8 python3 train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name="mobilenet_v2" \
    --learning_rate=$LR \
    --clone_on_cpu=False \
    --preprocessing_name="inception_v2" \
    --label_smoothing=0.1 \
    --moving_average_decay=0.9999 \
    --batch_size=96 \
    --num_clones=$NUM_GPUS \
    --num_workers=8 \
    --use_horovod=True \
    --learning_rate_decay_factor=0.98 \
    --max_number_of_steps=100000 \
    --num_epochs_per_decay=$NUM_EPOCHS_PER_DECAY 2>&1 | tee output_multi.log

