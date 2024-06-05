#!/bin/bash
set -x
apt update -y && apt-get install -y psmisc
script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
pushd $HOME
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
# install protobuf
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
# prepare TGI with Gaudi support
mkdir repos
cp -r "$script_dir/tgi-gaudi/" repos/
# build server
cd repos/tgi-gaudi/server
make gen-server
pip install pip --upgrade
pip install -r requirements.txt
pip install -e ".[bnb, accelerate]"
cd ..
# build router
make install-router
# build launcher
make install-launcher
# build benchmark
make install-benchmark
popd
pip install --force --no-deps git+https://github.com/huggingface/optimum-habana.git@ci_31032024
pip list
pip install transformers==4.38.2
pip list
# workaround for https://github.com/huggingface/text-generation-inference/issues/1876
pip install huggingface_hub==0.20.0
pip list
