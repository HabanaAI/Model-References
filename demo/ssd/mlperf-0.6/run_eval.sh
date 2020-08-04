#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD
export OUTPUT=$HOME/ssd_output

DATASETPATH1=/data/ssd/coco
DATASETPATH2=/igk-datasets/coco
DATASETPATH3=/software/data/tf/ssd/coco

if [ -d "$DATASETPATH" ]; then
	echo "Using user specified paths"
elif [ -d "$DATASETPATH1" ]; then
	# Check if datasets are in /data
	export DATASETPATH=$DATASETPATH1
	export INIT_PATH="/data/ssd/ssd_r34-mlperf/mlperf_artifact"
elif [ -d "$DATASETPATH2" ]; then
	# Check if mounted IGK and workloads
	export DATASETPATH=$DATASETPATH2
	export INIT_PATH="/workloads/mszutenberg/resnet34_ssd_checkpoint" # SW-18911
else	# Use /software
	export DATASETPATH=$DATASETPATH3
	export INIT_PATH="/software/data/tf/ssd_r34-mlperf/mlperf_artifact"
fi

echo "DATASETPATH = $DATASETPATH"
echo "INIT_PATH   = $INIT_PATH"

python3 ssd/ssd_main.py \
	--lr_warmup_epoch=5 \
	--base_learning_rate=3e-3 \
	--iterations_per_loop=625 \
	--mode=eval \
	--model_dir=$OUTPUT \
	--num_epochs=64 \
	--train_batch_size=32 \
	--training_file_pattern=$DATASETPATH/train* \
	--validation_file_pattern=$DATASETPATH/val* \
	--val_json_file=$DATASETPATH/raw-data/annotations/instances_val2017.json \
	--resnet_checkpoint=$INIT_PATH

