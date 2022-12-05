###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################

# script for this section: https://github.com/tensorflow/tpu/tree/01b67067be64987e9519b35342ba91d3179daeed/tools/datasets

# Inputs:
# $1 is IMAGENET_HOME: A folder containing zipped imagenet tar files. Should contain 2 tar files: ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar
# $2 is WORK_DIR: Location where temp files might be downloaded, and where final tf records will be created

# Outputs: workspace/tf_records

# sample run command:
# ./prep_mlperf_resnet_imagenet_data_fromtar.sh imagenet workspace

if [ "$#" -ne 2 ]; then
    echo "Expected 2 arguments, imagenet folder (with 2 subfolders, train and validation) and a workspace folder, got $# arguments instead"
    echo "Sample command line: ./prep_mlperf_resnet_imagenet_data.sh imagenet workspace"
    exit 1
fi

if ! command -v rename &> /dev/null
then
    echo "rename could not be found. Try: sudo apt install rename"
    exit 1
fi
IMAGENET_HOME=$1
WORK_DIR=$2


if [ ! -d "$IMAGENET_HOME" ]; then
    echo "Did not find imagenet folder $IMAGENET_HOME"
    exit 1
fi
if [ -d "$WORK_DIR" ]; then
    echo "Working dir $WORK_DIR already exists. Please delete it or give path to a non-existing directory"
    exit 1
fi

mkdir -p $WORK_DIR
git clone https://github.com/tensorflow/tpu.git $WORK_DIR/tpu
CURRENT=`pwd`
cd $WORK_DIR/tpu
git checkout r2.7
cd $CURRENT
wget -O $IMAGENET_HOME/synset_labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt

mkdir -p $IMAGENET_HOME/validation
mkdir -p $IMAGENET_HOME/train
# Extract validation and training
echo "Start extracting validation"
tar xf $IMAGENET_HOME/ILSVRC2012_img_val.tar -C $IMAGENET_HOME/validation
echo "Start extracting train"
tar xf $IMAGENET_HOME/ILSVRC2012_img_train.tar -C $IMAGENET_HOME/train
CURRENT=`pwd`
cd $IMAGENET_HOME/train
i=0
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  tar xf $f -C $d
  echo "extracted $i directory: $d"
  ((i=i+1))
done
cd $CURRENT
rm $IMAGENET_HOME/train/*.tar

python3.7 -m venv $WORK_DIR/venv
source $WORK_DIR/venv/bin/activate

# with the venv, pip version is 9.0.1, and tf 1.14.0 is the latest TF for that pip.
# upgrading
pip3 install --upgrade pip setuptools
pip install tensorflow==2.7.0
pip install absl-py
# To solve these dependency errors for the line: from google.cloud import storage
pip install google-cloud-storage

python $WORK_DIR/tpu/tools/datasets/imagenet_to_gcs.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$WORK_DIR/tf_records \
  --nogcs_upload


mv $WORK_DIR/tf_records/train $WORK_DIR/tf_records/img_train
mv $WORK_DIR/tf_records/validation $WORK_DIR/tf_records/img_val



# change names from train -> img_name, same for val
# like so: validation-00054-of-00128 img_val-00054-of-00128
# shortcut: rename  's/validation/img_val/' *

cd $WORK_DIR/tf_records/img_train
rename  's/train/img_train/' *
cd $CURRENT
cd $WORK_DIR/tf_records/img_val
rename  's/validation/img_val/' *

deactivate
