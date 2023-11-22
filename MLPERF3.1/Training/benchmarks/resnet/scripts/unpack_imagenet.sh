#!/bin/bash
###############################################################################
# Copyright (c) 2023, Habana Labs Ltd.  All rights reserved.
###############################################################################

function print_synopsis()
{
    cat << HEREDOC

    Usage: $program -ta|--train-archive PATH -va|--validation-archive PATH
                    -o|--output-path PATH [-j|--jobs-number COUNT] [-h|--help]

    Required arguments:
        -ta, --train-archive PATH       path to ImageNet training archive
        -va, --validation-archive PATH  path to ImageNet validation archive
        -o, --output-path PATH          path to folder where ImageNet will be upacked

    Optional arguments:
        -j, --jobs-number COUNT          number of jobs used when unpacking training archive
                                        default value is 16
        -h, --help                      print this help message

HEREDOC
}

function parse_args()
{
    export TRAIN_ARCHIVE_PATH=""
    export VAL_ARCHIVE_PATH=""
    export OUTPUT_PATH=""
    export JOBS_NUMBER=16

    while [ -n "$1" ]; do
        case "$1" in
            -ta | --train-archive )
                export TRAIN_ARCHIVE_PATH=$2
                shift 2
                ;;
            -va | --validation-archive )
                export VAL_ARCHIVE_PATH=$2
                shift 2
                ;;
            -o | --output-path )
                export OUTPUT_PATH=$2
                shift 2
                ;;
            -j | --jobs-number )
                export JOBS_NUMBER=$2
                shift 2
                ;;
            -h | --help )
                print_synopsis
                exit 0
                ;;
            * )
                echo "error: invalid parameter: $1"
                print_synopsis
                exit 1
                ;;
        esac
    done

    if [[ ! -f "$TRAIN_ARCHIVE_PATH" ]]; then
        echo "Please specify correct path to traing archive using -ta, --train-archive."
        print_synopsis
        exit 1
    fi

    if [[ ! -f "$VAL_ARCHIVE_PATH" ]]; then
        echo "Please specify correct path to  validation archive using -va, --validation-archive."
        print_synopsis
        exit 1
    fi

    if [[ -z "$OUTPUT_PATH" ]]; then
        echo "Please specify output path using -o, --output-path."
        print_synopsis
        exit 1
    fi
}

function reset_folder()
{
    rm -rf $1
    mkdir -p $1
}

function upack_train_subarchive()
{
    ARCHIVE_NAME=$1
    ARCHIVE_INDEX=$2
    NO_OF_ARCHIVES=$3
    PRINT="$ARCHIVE_INDEX/$NO_OF_ARCHIVES: $ARCHIVE_NAME"
    echo "Upacking $PRINT."

    pushd $TRAIN_PATH > /dev/null

    DIR=`basename $ARCHIVE_NAME .tar`
    mkdir $DIR
    tar xf $ARCHIVE_NAME -C $DIR
    rm $ARCHIVE_NAME

    popd > /dev/null
    echo "Finished upacking $PRINT."
}

function unpack_train()
{
    export TRAIN_PATH="$OUTPUT_PATH/train"
    export TMP_PATH="$OUTPUT_PATH/tmp"
    reset_folder $TRAIN_PATH
    reset_folder $TMP_PATH

    echo "Unpacking training data."
    tar xf $TRAIN_ARCHIVE_PATH -C $TMP_PATH

    echo "Unpacking subarchives."
    pushd $TMP_PATH > /dev/null
    ARCHIVES_COUNT=$(ls *.tar | wc -l)
    ARCHIVE_IDX=0
    for ARCHIVE in *.tar; do
        ((ARCHIVE_IDX++))

        while : ; do
            JOBS_COUNT=$(ls $TRAIN_PATH/*.tar 2> /dev/null | wc -l)
            if [ "$JOBS_COUNT" -lt "$JOBS_NUMBER" ]; then
                break
            fi
            sleep 1s
        done

        mv $ARCHIVE $TRAIN_PATH/
        upack_train_subarchive $ARCHIVE $ARCHIVE_IDX $ARCHIVES_COUNT &
    done
    popd > /dev/null

    wait
    rm -rf $TMP_PATH
    echo "Imagenet training data ready."
}

function unpack_val()
{
    export VAL_PATH="$OUTPUT_PATH/val"
    reset_folder $VAL_PATH

    echo "Unpacking validation data."
    tar xf $VAL_ARCHIVE_PATH -C $VAL_PATH

    echo "Reorganizing validation folder."
    export VALPREP_ADDR=https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
    pushd $VAL_PATH > /dev/null
    wget -qO- $VALPREP_ADDR | bash
    popd > /dev/null

    echo "Imagenet validation data ready."
}

parse_args "$@"

unpack_train &
unpack_val &

wait
