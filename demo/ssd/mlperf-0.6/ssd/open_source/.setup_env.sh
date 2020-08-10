#!/bin/bash

set -e

CLOUD_TPU=${CLOUD_TPU:-"cloud_tpu"}
PYTHONPATH=""

if [ ! -d $CLOUD_TPU ]; then
  git clone https://github.com/tensorflow/tpu.git $CLOUD_TPU
fi

pushd $CLOUD_TPU
# TODO(taylorrobie): Change to 'git checkout SHA' for official submission.
git pull
popd

sudo apt-get update
sudo apt-get install -y python3-tk unzip

if ! pip --version | grep -q -E --regexp="\(python 3.+)$";  then
  echo "Expected Python 3 pip. Found: " `pip --version`
  exit 1
fi

pip install --upgrade pip
pip install Cython==0.28.4 \
            matplotlib==2.2.2
pip install pycocotools==2.0.0
pip install Pillow==5.2.0

alias protoc="/usr/local/bin/protoc"
INSTALL_PROTO="yes"
if protoc --version | grep -q -E --regexp="3.6.1$"; then
  INSTALL_PROTO=""
fi

if [ ! -z $INSTALL_PROTO ]; then
  pushd /tmp
  curl -OL https://github.com/google/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
  unzip protoc-3.6.1-linux-x86_64.zip -d protoc3

  # Move protoc to /usr/local/bin/
  sudo mv protoc3/bin/* /usr/local/bin/

  # Move protoc3/include to /usr/local/include/
  if [ -d /usr/local/include/google/protobuf ]; then
    sudo rm -r /usr/local/include/google/protobuf
  fi
  sudo mv protoc3/include/* /usr/local/include/

  # Optional: change owner
  sudo chown $USER /usr/local/bin/protoc
  sudo chown -R $USER /usr/local/include/google
  popd
fi
