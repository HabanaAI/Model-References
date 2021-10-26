export DATASET_DIR=/software/data/tf/data/mobilenetv2/train
export NUM_GPUS=1
export TRAIN_DIR=/tmp/mobilenetv2
export LR=0.045
export NUM_EPOCHS_PER_DECAY=2.5
python3 train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name="mobilenet_v2" \
    --learning_rate=$LR \
    --clone_on_cpu=False \
    --preprocessing_name="inception_v2" \
    --log_every_n_steps=20 \
    --label_smoothing=0.1 \
    --moving_average_decay=0.9999 \
    --batch_size=96 \
    --num_clones=$NUM_GPUS \
    --learning_rate_decay_factor=0.98 \
    --max_number_of_steps=1000000 \
    --num_epochs_per_decay=$NUM_EPOCHS_PER_DECAY 2>&1 | tee output_single.log

