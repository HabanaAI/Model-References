#!/bin/bash

PYTHON=$1
VENV_BASE=$2
VENV_NAME=$3

GIT_BASE=`pwd`

if [ "$#" -ne 3 ]; then
    echo "Usage: ./create_venv_habana.sh <python 3.6 interpeter> <venv base> <venv name>"
    echo ""
    echo "This will generate a new python virtual environment"
    exit 1
fi


if [ -d "$VENV_BASE/$VENV_NAME" ]; then
    echo "Failed: a virtual environment of this name already exists in $VENV_BASE"
    exit 1
fi

cd $VENV_BASE

$PYTHON -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# menta requirements

# stereo packages
pip install pip setuptools --upgrade
pip install tensorflow==1.13.1 s3fs==0.4.2 sagemaker==1.70.2

# me packages

# Extra commands to manually add non pip installable packages

# Back to the general script
cd $GIT_BASE
