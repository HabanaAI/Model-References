###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - Added calibration dir argument

#!/bin/bash

: "${DOWNLOAD_PATH:=../coco2014}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --download-path )       shift
                                     DOWNLOAD_PATH=$1
                                     ;;
    esac
    case $1 in
        -i | --images )
                                     IMAGES=1
                                     ;;
    esac
    case $1 in
        -n | --num-workers  )        shift
                                      NUM_WORKERS=$1
                                      ;;
    esac
    case $1 in
        --calibration-dir  )         shift
                                     CALIBRATION_DIR=$1
                                     ;;
    esac
    shift
done

if [ -z ${IMAGES} ];
then
    python3 coco_calibration.py \
        --dataset-dir ${DOWNLOAD_PATH} \
        --num-workers ${NUM_WORKERS} \
        --calibration-dir ${CALIBRATION_DIR}

else
    python3 coco_calibration.py \
        --dataset-dir ${DOWNLOAD_PATH} \
        --download-images \
        --num-workers ${NUM_WORKERS}
fi
