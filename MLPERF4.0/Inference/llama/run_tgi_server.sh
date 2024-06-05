#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

set -e

# Kill TGI if present
killall -q -15 text-generation-launcher && sleep 60

usage() {
    echo "Usage: $0 --model model --bs batch_size"
    echo "Options:"
    echo "  --model, -m             Specify the model"
    echo "  --bs                    Specify the batch size"
    echo "  --scenario              Specify the scenario, possible values: Offline, Server"
    echo "  --fp8                   Use the fp8 quantization"
    echo "  --output_dir, -o        Specify the output dir for logs if RESULT_DIR is not set, default: ./results"
    echo "  --help                  Display this help message"
    exit 1
}

wait_for_server() {
    local model="$1"

    timeout=3600
    step=10
    current_time=0

    set +x
    while [ "$current_time" -lt "$timeout" ]; do
        output=$(curl -s http://localhost:8080/info | grep Llama-2-$model-chat-hf | wc -l)
        if (( $output > 0 )); then
            set -x
            return
        fi
        sleep $step
        current_time=$((current_time + step))
    done

    set -x
    echo "TGI server didn't start"
    exit -1
}

model="70b"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-m)
            model=$2
            shift 2
            ;;
        --bs)
            batch_size=$2
            shift 2
            ;;
        --scenario)
            scenario=$2
            shift 2
            ;;
        --fp8)
            fp8=true
            shift
            ;;
        --output_dir|-o)
            output_dir=$2
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

if [[ -n $HELP || -z $model || -z $batch_size || -z $scenario ]]; then
    usage
fi

script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
output_dir=${RESULT_DIR:-$output_dir}
output_dir=${output_dir:-$script_dir/results}
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

if [ "$fp8" = true ]; then
    export QUANT_CONFIG=hqt/llama2-70b-8x/config_meas_maxabs_quant_MAXABS_HW.json
fi

waiting_served_ratio=0.006
if [ "$scenario" = "Offline" ]; then
    if [ "$fp8" = true ]; then
        prefill_batch_size=16
        PREFILL_BATCH_BUCKET_SIZE=16
        waiting_served_ratio=0.017
        export PAD_SEQUENCE_TO_MULTIPLE_OF=64
    else
        prefill_batch_size=4
    fi
elif [ "$fp8" = true ]; then
    prefill_batch_size=4
    PREFILL_BATCH_BUCKET_SIZE=4
else
    prefill_batch_size=2
fi

source "$HOME/.cargo/env"
MAX_INPUT_SEQ_LEN=${MAX_INPUT_SEQ_LEN:-1024}
export MAX_INPUT_SEQ_LEN
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-2048}
export  MAX_TOTAL_TOKENS
PAD_SEQUENCE_TO_MULTIPLE_OF=${PAD_SEQUENCE_TO_MULTIPLE_OF:-32}
export PAD_SEQUENCE_TO_MULTIPLE_OF
PT_HPU_ENABLE_LAZY_COLLECTIVES=${PT_HPU_ENABLE_LAZY_COLLECTIVES:-true}
export PT_HPU_ENABLE_LAZY_COLLECTIVES
SKIP_TOKENIZER_IN_TGI=${SKIP_TOKENIZER_IN_TGI:-true}
export SKIP_TOKENIZER_IN_TGI
PREFILL_BATCH_BUCKET_SIZE=${PREFILL_BATCH_BUCKET_SIZE:-1}
export PREFILL_BATCH_BUCKET_SIZE
export BATCH_BUCKET_SIZE=$batch_size
max_batch_total_tokens=$(($batch_size*$MAX_TOTAL_TOKENS))
max_batch_prefill_tokens=$(($prefill_batch_size*$MAX_INPUT_SEQ_LEN))

text-generation-launcher --port 8080 \
    --model-id /mnt/weka/data/pytorch/llama2/Llama-2-$model-chat-hf --sharded true --num-shard 8 \
    --max-total-tokens 2048 --max-input-length 1024 \
    --max-batch-prefill-tokens $max_batch_prefill_tokens --max-batch-total-tokens $max_batch_total_tokens \
     --shard-uds-path /tmp/text-generation-server-$scenario \
    --max-concurrent-requests 1024 --max-waiting-tokens 20 --waiting-served-ratio $waiting_served_ratio \
     --dtype bfloat16 &>> ${output_dir}/text-generation-launcher.log &

wait_for_server "${model}"
