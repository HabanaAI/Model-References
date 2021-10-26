DATASET_DIR=/software/data/tf/data/mobilenetv2/validation
CHECKPOINT_DIR=/tmp/mobilenetv2      #/model.ckpt
EVAL_DIR=/tmp/my_eval_logs
NUM_GPUS=1
python3 eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_DIR} \
    --eval_dir=${EVAL_DIR} \
    --max_num_batches=400 \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --dataset_dir=${DATASET_DIR} \
    --model_name="mobilenet_v2" \
    --batch_size=96 \
    --label_smoothing=0.1 \
    --moving_average_decay=0.9999 \
    --num_epochs_per_decay=2.5     # / $NUM_GPUS # train_image_classifier does per clone epochs
