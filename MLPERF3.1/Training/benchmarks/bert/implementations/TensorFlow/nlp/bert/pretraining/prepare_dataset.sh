#!/bin/bash

###############################################################################
# Copyright (c) 2023, Habana Labs Ltd.  All rights reserved.
###############################################################################

function print_synopsis()
{
    cat << HEREDOC

    Usage: $program -d|--data-path PATH -s|--scripts-path PATH
                    [-od|--only-download] [-op|--only-preprocessing]
                    [-j|--jobs-limit COUNT] [-h|--help]

    Required arguments:
        -d, --data-path PATH        path were dataset will be stored
        -s, --scripts-path PATH     path to BERT scripts
                                    required only when prerocessing is performed (--only-download is not used)

    Optional arguments:
        -od, --only-download        only download packages from google drives (skip preprocessing)
        -op, --only-preprocessing   only preprocess dataset (skip download)
                                    downloads must already be available in --data-path
        -j, --jobs-limit COUNT      number of jobs used for tfrecords parallel processing, default value:25
        -h, --help                  print this help message

HEREDOC
}

function reset_folder()
{
    rm -rf $1
    mkdir -p $1
}

function parse_args()
{
    export SCRIPT_PATH=""
    export DATA_PATH=""
    export JOBS_LIMIT=25
    export DO_DOWNLOAD=1
    export DO_PREPROCESSING=1

    while [ -n "$1" ]; do
        case "$1" in
            -d | --data-path )
                export DATA_PATH=$2
                shift 2
                ;;
            -s | --scripts-path )
                export SCRIPT_PATH=$2
                shift 2
                ;;
            -od | --only-download )
                export DO_PREPROCESSING=0
                shift 1
                ;;
            -op | --only-preprocessing )
                export DO_DOWNLOAD=0
                shift 1
                ;;
            -j | --jobs-limit )
                export JOBS_LIMIT=$2
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

    if [[ -z "$SCRIPT_PATH" && $DO_PREPROCESSING -eq 1 ]]; then
        echo "Please specify bert scripts path using -s, --scripts-path."
        print_synopsis
        exit 1
    fi

    if [[ -z "$DATA_PATH" ]]; then
        echo "Please specify output path using -o, --output-path."
        print_synopsis
        exit 1
    fi

    if [[ $DO_DOWNLOAD -eq 0 && $DO_PREPROCESSING -eq 0 ]]; then
        echo "Cannot run with --only-download and --only-preprocessing both set."
        print_synopsis
        exit 1
    fi
}

function set_paths()
{
    export INPUT_PATH=$DATA_PATH/input
    export TMP_PATH=$DATA_PATH/tmp
    export VOCAB_FILE=$INPUT_PATH/vocab.txt
    export BERT_CONFIG_FILE=$INPUT_PATH/bert_config.json
    export RESULTS_TEXT_FILE=$INPUT_PATH/results_text.tar.gz
    export EVAL_FILE=$TMP_PATH/eval.txt
    export INPUT_RECORDS_PATH=$TMP_PATH/results4
    export OUTPUT_RECORDS_PATH=$DATA_PATH/unpacked_data
    export PACKED_RECORDS_PATH=$DATA_PATH/packed_data_500
    export TMP_EVAL_FILE=$TMP_PATH/tmp_eval_10k
    export OUTPUT_EVAL_PATH=$DATA_PATH/eval_dataset
    export OUTPUT_EVAL_FILE=$OUTPUT_EVAL_PATH/eval_10k
    export CHECKPOINT_PATH=$DATA_PATH/checkpoint
    export TMP_PATH=$DATA_PATH/tmp
}

function download_file()
{
    FILE_NAME=$1
    FILE_ID=$2

    gdown $FILE_ID

    if [ ! -f "$FILE_NAME" ]; then
        echo "Failed to download file $FILE_NAME or downloaded a file with the wrong name."
        exit 1
    fi
}

function download_input()
{
    FILES=( "results_text.tar.gz" "14xV2OUGSQDG_yDBrmbSdcDC-QGeqpfs_"
            "bert_config.json" "1fbGClQMi2CoMv7fwrwTC5YYPooQBdcFW"
            "vocab.txt" "1USK108J6hMM_d27xCHi738qBL8_BT1u1"
            "model.ckpt-28252.index" "1oVBgtSxkXC9rH2SXJv85RXR9-WrMPy-Q"
            "model.ckpt-28252.meta" "1vAcVmXSLsLeQ1q7gvHnQUSth5W_f_pwv"
            "model.ckpt-28252.data-00000-of-00001" "1pJhVkACK3p_7Uc-1pAzRaOXodNeeHZ7F")

    reset_folder $INPUT_PATH
    pushd $INPUT_PATH > /dev/null

    for ((i = 0; i < ${#FILES[@]}; i += 2)); do
        download_file ${FILES[$i]} ${FILES[$((i + 1))]}
    done

    echo "Downloaded files:"
    echo "$(ls)"
    popd > /dev/null
}

function unpack_results_text()
{
    echo "Unpacking $RESULTS_TEXT_FILE"
    rm -r $INPUT_RECORDS_PATH 2>/dev/null
    tar -xvf $RESULTS_TEXT_FILE -C $TMP_PATH/
    mv $INPUT_RECORDS_PATH/eval.txt $EVAL_FILE
    rm $INPUT_RECORDS_PATH/eval.md5
}

function prepare_checkpoint_dir()
{
    reset_folder $CHECKPOINT_PATH

    echo "Prepare checkpoint directory."
    cp $INPUT_PATH/model.ckpt-* $CHECKPOINT_PATH/
    cp $BERT_CONFIG_FILE $CHECKPOINT_PATH/
    echo 'model_checkpoint_path: "model.ckpt-28252"' > $CHECKPOINT_PATH/checkpoint
}

function prepare_eval_dir()
{
    rm -f $TMP_EVAL_FILE 2>/dev/null
    reset_folder $OUTPUT_EVAL_PATH

    pushd $SCRIPT_PATH > /dev/null

    python3 pretraining/create_pretraining_data.py \
        --input=$EVAL_FILE \
        --output=$TMP_EVAL_FILE \
        --vocab_file=$VOCAB_FILE \
        --do_lower_case=True \
        --max_seq_length=512 \
        --max_predictions_per_seq=76 \
        --masked_lm_prob=0.15 \
        --random_seed=12345 \
        --dupe_factor=10

    python3 pretraining/pick_eval_samples.py \
        --input_tfrecord=$TMP_EVAL_FILE \
        --output_tfrecord=$OUTPUT_EVAL_FILE \
        --num_examples_to_pick=10000

    popd > /dev/null
}

function prepare_packed_data()
{
    reset_folder $OUTPUT_RECORDS_PATH
    reset_folder $PACKED_RECORDS_PATH

    pushd $SCRIPT_PATH > /dev/null

    python3 pretraining/create_pretraining_data.py \
        --input=$INPUT_RECORDS_PATH \
        --output=$OUTPUT_RECORDS_PATH \
        --jobs=$JOBS_LIMIT \
        --vocab_file=$VOCAB_FILE \
        --do_lower_case=True \
        --max_seq_length=512 \
        --max_predictions_per_seq=76 \
        --masked_lm_prob=0.15 \
        --random_seed=12345 \
        --dupe_factor=10

    export TF_CPP_MIN_LOG_LEVEL=2
    python3 pretraining/pack_pretraining_data_tfrec.py \
        --input-dir "$OUTPUT_RECORDS_PATH/" \
        --output-dir $PACKED_RECORDS_PATH \
        --jobs $JOBS_LIMIT \
        --max-files 500
    unset TF_CPP_MIN_LOG_LEVEL

    popd > /dev/null
}

parse_args "$@"
set_paths

if [ "$DO_DOWNLOAD" -eq 1 ]; then
    download_input
fi

if [ "$DO_PREPROCESSING" -eq 1 ]; then
    reset_folder $TMP_PATH

    prepare_checkpoint_dir
    unpack_results_text
    prepare_eval_dir
    prepare_packed_data

    rm -r $TMP_PATH
fi
