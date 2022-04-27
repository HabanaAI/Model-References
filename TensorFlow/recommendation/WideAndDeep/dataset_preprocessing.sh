#!/bin/bash
#
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###############################################################################

DEFAULT_DATASET_PATH="/outbrain/orig/"
DEFAULT_OUTPUT_PATH="/outbrain/tfrecords/"

PRINT_HELP=false
export DATA_BUCKET_FOLDER=$DEFAULT_DATASET_PATH
export LOCAL_DATA_TFRECORDS_DIR=$DEFAULT_OUTPUT_PATH

function install_deps() {
    echo "Installing all required dependencies..."
    apt-get -y update && \
    apt-get install --no-install-recommends -y openjdk-8-jre-headless ca-certificates-java time && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    pip install --upgrade pip && \
    pip install pyspark==3.2.0 pandas==1.3.5 && \
    pip install --no-deps -r requirements-no-deps.txt
    echo -e "\nDone: all required dependencies have been installed!"
}

function start_data_preprocessing() {
    export PYTHONPATH=$PYTHONPATH:`pwd`
    echo -e "\nStarting data preprocessing 1/3..."
    $PYTHON data/outbrain/spark/preproc1.py
    echo -e "\nStarting data preprocessing 2/3..."
    $PYTHON data/outbrain/spark/preproc2.py
    echo -e "\nStarting data preprocessing 3/3..."
    $PYTHON data/outbrain/spark/preproc3.py
}

while (( "$#" )); do
    case "$1" in
        -h|--help)
            PRINT_HELP=true
            shift
        ;;
        --dataset_path)
            if [ ! -z "$2" ]
            then
                export DATA_BUCKET_FOLDER=$2
                shift 2
            else
                echo "Unrecognized argument for --dataset_path: $2"
                PRINT_HELP=true
                break
            fi
        ;;
        --output_path)
            if [ ! -z "$2" ]
            then
                export LOCAL_DATA_TFRECORDS_DIR=$2
                shift 2
            else
                echo "Unrecognized argument for --output_path: $1"
                PRINT_HELP=true
                break
            fi
            shift
        ;;
        -*|--*=) # unsupported flags
            echo -e $"Error: Unsupported flag $1\n" >&2
            exit 1
            shift
        ;;
    esac
done

if $PRINT_HELP; then
    printf "
    data_preprocessing.sh is a script for preprocessing already downloaded dataset.
    The script installs all required dependencies and starts preprocessing of the dataset using Spark.

    Usage:  bash $(basename $0) [--dataset_path <DATASET_PATH>] [--output_path <OUTPUT_PATH>]

    --dataset_path <DATASET_PATH>   - Path, where original dataset is stored (default: $DEFAULT_DATASET_PATH)
    --output_path <OUTPUT_PATH>     - Path, where preprocessed dataset should be saved (default: $DEFAULT_OUTPUT_PATH)
"
    exit 1
fi

install_deps
start_data_preprocessing
echo -e "\nData preprocessing is finished! You can find the data in $LOCAL_DATA_TFRECORDS_DIR."
