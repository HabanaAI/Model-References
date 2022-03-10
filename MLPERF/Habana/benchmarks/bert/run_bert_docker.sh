###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
#!/bin/bash

DOCKER_IMAGE=vault.habana.ai/gaudi-docker/1.2.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.7.0:1.2.0-585
CONTAINER_NAME=mlperf1_1

# =========================================================
# Modify below 3 env variables:
#  - CODE_DIR: path to the MLPERF code folder on the host
#  - DATASET_DIR: path to the TF packed mlperf bert dataset folder on the host
#  - RESULTS_DIR: needs to be a writable folder, no special restriction
# =========================================================
RESULTS_DIR=/data/mlperf_result
DATASET_DIR=/data/mlperf_bert
CODE_DIR=/data/MLPERF

sudo docker run --privileged --security-opt seccomp=unconfined \
	--name $CONTAINER_NAME -td                         \
	--device=/dev/hl_controlD0:/dev/hl_controlD0 \
	--device=/dev/hl_controlD1:/dev/hl_controlD1 \
	--device=/dev/hl_controlD2:/dev/hl_controlD2 \
	--device=/dev/hl_controlD3:/dev/hl_controlD3 \
	--device=/dev/hl0:/dev/hl0                   \
	--device=/dev/hl1:/dev/hl1                   \
	--device=/dev/hl2:/dev/hl2                   \
	--device=/dev/hl3:/dev/hl3                   \
	-e DISPLAY=$DISPLAY                          \
	-e LOG_LEVEL_ALL=6                           \
	-v $CODE_DIR:/root/MLPERF	                 \
	-v  /data/shared:/root/shared                \
	-v /sys/kernel/debug:/sys/kernel/debug       \
	-v /tmp/.X11-unix:/tmp/.X11-unix:ro          \
	-v /tmp:/tmp                                 \
	-v $RESULTS_DIR:/root/scratch                \
	-v $DATASET_DIR:/root/datasets/              \
	--cap-add=sys_nice --cap-add=SYS_PTRACE      \
	--user root --workdir=/root --net=host       \
	--ulimit memlock=-1:-1 ${DOCKER_IMAGE}

docker exec -ti $CONTAINER_NAME bash