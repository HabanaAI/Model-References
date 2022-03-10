###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
DOCKER_IMAGE=vault.habana.ai/gaudi-docker/1.2.0/ubuntu18.04/habanalabs/tensorflow-installer-tf-cpu-2.7.0:1.2.0-585
CONTAINER_NAME=mlperf_resnet.${USER}

# =========================================================
# Modify below 3 env variables:
#  - MLPERF_HOST: path to the MLPERF folder on the host
#  - IMAGENET_HOST: path to the TF imagenet folder on the host
#  - RESULTS_DIR: needs to be a writable folder, no special restriction
# =========================================================
MLPERF_HOST=${HOME}/MLPERF/
IMAGENET_HOST=/data/imagenet2012
RESULTS_DIR=/data/scratch

MLPERF_CONTAINER=/root/MLPERF/
IMAGENET_CONTAINER=/root/datasets/imagenet/tf_records/

docker stop $CONTAINER_NAME

sleep 3

docker run --privileged --security-opt seccomp=unconfined \
           --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
           --name $CONTAINER_NAME -d --rm               \
           -e DISPLAY=$DISPLAY                          \
           -e LOG_LEVEL_ALL=6                           \
           -v /sys/kernel/debug:/sys/kernel/debug       \
           -v /tmp/.X11-unix:/tmp/.X11-unix:ro          \
           -v $RESULTS_DIR:/root/scratch                \
           -v ${IMAGENET_HOST}:${IMAGENET_CONTAINER}    \
           -v ${MLPERF_HOST}:${MLPERF_CONTAINER}        \
           --cap-add=sys_nice --cap-add=SYS_PTRACE      \
           --user root --workdir=/root --net=host       \
           --ulimit memlock=-1:-1 ${DOCKER_IMAGE} sleep infinity

docker exec $CONTAINER_NAME bash -c "service ssh start"

docker exec $CONTAINER_NAME mkdir /root/shared
docker exec $CONTAINER_NAME bash -c "echo 127.0.0.1 > /root/shared/hosts"
docker exec $CONTAINER_NAME apt install -y numactl

# If running on AWS DL1 instance, use HLS_TYPE=OCP1
docker exec -w ${MLPERF_CONTAINER}/Habana/benchmarks/resnet/implementations/HLS-1-N1 -e NUM_WORKERS_PER_HLS=8 -e HLS_TYPE=OCP1 $CONTAINER_NAME bash ./launch_keras_resnet_hvd.sh --cpu-pin cpu
# Otherwise, use HLS_TYPE=HLS1
# docker exec -w ${MLPERF_CONTAINER}/Habana/benchmarks/resnet/implementations/HLS-1-N1 -e NUM_WORKERS_PER_HLS=8 -e HLS_TYPE=HLS1 $CONTAINER_NAME bash ./launch_keras_resnet_hvd.sh --cpu-pin cpu