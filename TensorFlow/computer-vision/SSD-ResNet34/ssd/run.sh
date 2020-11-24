#!/bin/bash
# Run script for OSS SSD

set -e

CLOUD_TPU="cloud_tpu"
DATA_DIR="/tmp/coco"
MODEL_DIR="/tmp/ssd_test_model_dir"
# MODEL_DIR="gs://garden-team-scripts/ssd_test_model_dir"

PYTHONPATH=""

# TODO(taylorrobie): properly open source backbone checkpoint
RESNET_CHECKPOINT="gs://garden-team-scripts/resnet34_ssd_checkpoint"

chmod +x open_source/.setup_env.sh
./open_source/.setup_env.sh

if [ ! -d $DATA_DIR ]; then
  pushd ${CLOUD_TPU}/tools/datasets
  bash download_and_preprocess_coco.sh $DATA_DIR
  popd
fi

if [ -d $MODEL_DIR ]; then
  rm -r $MODEL_DIR
fi

export PYTHONPATH="$(pwd)/cloud_tpu/models/official/retinanet:${PYTHONPATH}"
python3 ssd_main.py  --use_tpu=False \
                     --device=gpu \
                     --mode=train \
                     --train_batch_size=32 \
                     --training_file_pattern="${DATA_DIR}/train-*" \
                     --resnet_checkpoint=${RESNET_CHECKPOINT} \
                     --model_dir=${MODEL_DIR} \
                     --num_epochs=60

python3 ssd_main.py  --use_tpu=False \
                     --device=gpu \
                     --mode=eval \
                     --eval_batch_size=32 \
                     --validation_file_pattern="${DATA_DIR}/val-*" \
                     --val_json_file="${DATA_DIR}/raw-data/annotations/instances_val2017.json" \
                     --model_dir=${MODEL_DIR}\
                     --eval_timeout=0
