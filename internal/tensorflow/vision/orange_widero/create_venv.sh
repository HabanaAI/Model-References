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

python -m virtualenv $VENV_NAME
source $VENV_NAME/bin/activate

pip install pip --upgrade
pip install -r $GIT_BASE/requirements.txt

# Extra commands to manually add non pip installable packages

pip install git+http://gitlab.mobileye.com/adih/distcorrection.git

cd $VENV_BASE/$VENV_NAME/lib/python2.7/site-packages
ln -s /usr/lib/python2.7/dist-packages/PyQt4
ln -s tables PyTables
ln -s /mobileye/algo_STEREO3/simulator/code/carla_scripts
ln -s /usr/lib/python2.7/dist-packages/sip.so

git clone git@gitlab.mobileye.com:dl/menta.git
cd menta
git reset --hard 4e03783b11af9bd9b5c3ce63952b17585486f2d4

# Back to the general script
cd $GIT_BASE
