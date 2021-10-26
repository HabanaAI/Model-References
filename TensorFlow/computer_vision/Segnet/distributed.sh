#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -d DATATYPE -b BATCH -w WORKERS -e TRAINING_ENGINE -p DATA_PATH"
   echo -e "\t-d datatype e.g. fp32 or bf16"
   echo -e "\t-b batch to train"
   echo -e "\t-w workers to train"
   echo -e "\t-e training engine - hpu or gpu"
   echo -e "\t-p dataset location"
   exit 1 # Exit script after printing help
}

while getopts ":d:b:w:e:p:" opt
do
   case "$opt" in
      d ) DATATYPE="$OPTARG" ;;
      b ) BATCH="$OPTARG" ;;
      w ) WORKERS="$OPTARG" ;;
      e ) TRAINING_ENGINE="$OPTARG" ;;
      p ) DATAPATH="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if  [[ -z "$DATATYPE" ]]; then
    DATATYPE="bf16"
fi

if  [[ -z "$BATCH" ]]; then
    BATCH=16
fi

if [[ -z "$WORKERS" ]]; then
    WORKERS="4"
fi

if [[ -z "$TRAINING_ENGINE" ]]; then
    TRAINING_ENGINE="hpu"
fi

if [[ -z "$DATAPATH" ]]; then
    DATAPATH=/home/dataset1
fi

if [ "$DATATYPE" == "amp" ]; then
  if [ "$TRAINING_ENGINE" == "gpu" ]; then
    export TF_ENABLE_AUTO_MIXED_PRECISION=1
  fi
fi

export HCL_CONFIG_PATH=hcl_config.json
#export LOG_LEVEL_ALL=0
str="Running distributed training on ${WORKERS} workers for ${DATATYPE} with batch size ${BATCH}"
echo "$str"
if [ "$TRAINING_ENGINE" == "hpu" ]; then
    mpirun -np ${WORKERS} --allow-run-as-root python3 -m keras_segmentation train --train_images="${DATAPATH}/images_prepped_train/" \
        --train_annotations="${DATAPATH}/annotations_prepped_train/" --val_images="${DATAPATH}/images_prepped_test/" \
        --val_annotations="${DATAPATH}/annotations_prepped_test/" --n_classes=12 --input_height=320 \
	--input_width=640 --model_name="vgg_segnet" --data_type=${DATATYPE} --distributed --batch_size=${BATCH}  --epoch 125 --train_engine=hpu
elif [ "$TRAINING_ENGINE" == "gpu" ]; then
    horovodrun -np ${WORKERS} python3 -m keras_segmentation train --train_images="${DATAPATH}/images_prepped_train/" \
        --train_annotations="${DATAPATH}/annotations_prepped_train/" --val_images="${DATAPATH}/images_prepped_test/" \
        --val_annotations="${DATAPATH}/annotations_prepped_test/" --n_classes=12 --input_height=320 \
        --input_width=640 --model_name="vgg_segnet" --data_type=${DATATYPE} --distributed  --batch_size=${BATCH} --epoch 125 --train_engine=gpu --loss_type=2
fi
