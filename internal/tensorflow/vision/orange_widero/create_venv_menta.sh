#!/bin/bash

VENV_BASE="/mobileye/algo_STEREO3/stereo/venv"
GIT_BASE=`pwd`

if [ "$#" -ne 1 ]; then
    echo "Usage: create_venv.sh [venv name]"
    echo ""
    echo "This will generate a new python virtual environment based on the requirements.txt file"
    echo "and the extra commands in this script."
    exit 1
fi

VENV_NAME=$1

if [ -d "$VENV_BASE/$VENV_NAME" ]; then
    echo "Failed: a virtual environment of this name already exists in $VENV_BASE"
    exit 1
fi

cd $VENV_BASE

/mobileye/algo_STEREO3/davidn/Python-3.6.6/python -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# menta requirements
pip install -r /mobileye/algo_OBJD/benas/menta_tutorial/requirements.txt

# stereo packages
pip install pip setuptools --upgrade
pip install tensorflow==1.12
pip install s3fs ipykernel tfmpl open3d==0.9 pptk
pip install sagemaker --upgrade

# dlo packages
pip install networkx tabulate dill psutil hyperopt lxml jsonpath_rw Pillow progressbar awesome-slugify
# me packages
pip install me-nebula file2path mdblib devkit

# Extra commands to manually add non pip installable packages
cd $VENV_BASE/$VENV_NAME/lib/python3.6/site-packages
ln -s /mobileye/algo_STEREO3/davidn/menta


# Back to the general script
cd $GIT_BASE
