#!/bin/bash
# Run script for OSS SSD on TPU

set -e

# Example settings:
# TPU="taylorrobie-tpu-0"
# BUCKET="gs://taylorrobie-tpu-test-bucket-2"

# Remove IDE "not assigned" warning highlights.
TPU=${TPU:-""}
BUCKET=${BUCKET:-""}
DATE=$(date '+%Y-%m-%d_%H:%M:%S')

if [[ -z ${TPU} ]]; then
  echo "Please set 'TPU' to the name of the TPU to be used."
  exit 1
fi

if [[ -z ${BUCKET} ]]; then
  echo "Please set 'BUCKET' to the GCS bucket to be used."
  exit 1
fi


# TODO(taylorrobie): Remove duplicated code.

CLOUD_TPU="cloud_tpu"
DATA_SUBDIR="ssd_coco"
DATA_DIR="${BUCKET}/${DATA_SUBDIR}"
MODEL_DIR="${BUCKET}/ssd_oss_test/model_dir_${DATE}"

PYTHONPATH=""

# TODO(taylorrobie): properly open source backbone checkpoint
RESNET_CHECKPOINT="gs://taylorrobie-tpu-test-bucket-2/resnet34_ssd_checkpoint"

chmod +x open_source/.setup_env.sh
./open_source/.setup_env.sh

if ! gsutil ls "${DATA_DIR}/"; then
  TEMP_DIR="/tmp/${DATA_SUBDIR}"
  pushd ${CLOUD_TPU}/tools/datasets
  bash download_and_preprocess_coco.sh $TEMP_DIR
  gsutil -m cp -r $TEMP_DIR $BUCKET
  popd
fi

if gsutil ls ${MODEL_DIR}; then
  gsutil -m rm -r ${MODEL_DIR}
fi

export PYTHONPATH="$(pwd)/cloud_tpu/models/official/retinanet:${PYTHONPATH}"
python3 ssd_main.py  --use_tpu=True \
                     --tpu_name=${TPU} \
                     --device=tpu \
                     --train_batch_size=32 \
                     --training_file_pattern="${DATA_DIR}/train-*" \
                     --resnet_checkpoint=${RESNET_CHECKPOINT} \
                     --model_dir=${MODEL_DIR} \
                     --num_epochs=60

python3 ssd_main.py  --use_tpu=True \
                     --tpu_name=${TPU} \
                     --device=tpu \
                     --mode=eval \
                     --eval_batch_size=256 \
                     --validation_file_pattern="${DATA_DIR}/val-*" \
                     --val_json_file="${DATA_DIR}/raw-data/annotations/instances_val2017.json" \
                     --model_dir=${MODEL_DIR} \
                     --eval_timeout=0
