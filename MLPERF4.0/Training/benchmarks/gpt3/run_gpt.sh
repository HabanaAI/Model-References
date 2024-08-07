#!/bin/bash
# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.

set -ex
function parse_args()
{
    while true; do
        case "$1" in
            --data-dir )
                DATA_DIR="$2"
                shift 2 ;;
            --num-nodes )
                NUM_NODES="$2"
                shift 2 ;;
            --devices-per-node )
                DEVICES_PER_NODE="$2"
                shift 2 ;;
            --data-parallel-size )
                DP="$2"
                shift 2 ;;
            --tensor-model-parallel-size )
                TP="$2"
                shift 2 ;;
            --pipeline-model-parallel-size )
                PP="$2"
                shift 2 ;;
            --num-layers )
                NUM_LAYERS="$2"
                shift 2 ;;
            --hidden-size )
                HIDDEN_SIZE="$2"
                shift 2 ;;
            --num-attention-heads )
                NUM_ATTENTION_HEADS="$2"
                shift 2 ;;
            --seq-length )
                SEQ_LENGTH="$2"
                shift 2 ;;
            --dropout )
                DROPOUT="$2"
                shift 2 ;;
            --micro-batch-size )
                MICRO_BATCH="$2"
                shift 2 ;;
            --eval-micro-batch-size )
                EVAL_MICRO_BATCH="$2"
                shift 2 ;;
            --global-batch-size )
                GLOBAL_BATCH="$2"
                shift 2 ;;
            --train-samples )
                TRAIN_SAMPLES="$2"
                shift 2 ;;
            --lr )
                LR="$2"
                shift 2 ;;
            --min-lr )
                MIN_LR="$2"
                shift 2 ;;
            --lr-decay-samples )
                LR_DECAY_SAMPLES="$2"
                shift 2 ;;
            --lr-warmup-samples )
                LR_WARMUP_SAMPLES="$2"
                shift 2 ;;
            --seed )
                SEED="$2"
                shift 2 ;;
            --eval-iters )
                EVAL_ITERS="$2"
                shift 2 ;;
            --eval-interval )
                EVAL_INTERVAL="$2"
                shift 2 ;;
            --exit-interval )
                EXIT_INTERVAL="$2"
                shift 2 ;;
            --output-dir )
                OUTPUT_DIR="$2"
                shift 2 ;;
            --start-from-ckpt )
                START_FROM_CKPT="$2"
                shift 2 ;;
            --universal-ckpt-path )
                UNIVERSAL_CKPT_PATH="$2"
                shift 2 ;;
            --save-checkpoints )
                SAVE_CKPT="$2"
                shift 2 ;;
            --save-checkpoints-dir )
                SAVE_CKPT_DIR="$2"
                shift 2 ;;
            --save-interval )
                SAVE_INTERVAL="$2"
                shift 2 ;;
            --log-interval )
                LOG_INTERVAL="$2"
                shift 2 ;;
            --tensorboard-dir )
                TENSORBOARD_DIR="$2"
                shift 2 ;;
            --kill-switch-file )
                KILL_SWITCH_FILE="$2"
                shift 2 ;;
            --hosts )
                HOSTS="$2"
                shift 2 ;;
            --hostsfile )
                HOSTSFILE="$2"
                shift 2 ;;
            --mllog-output-path )
                MLLOG_FILE="$2"
                shift 2 ;;
            --eval-loss-exit-value )
                EVAL_LOSS_EXIT_VALUE="$2"
                shift 2 ;;
            --profile )
                PROFILE_FLAG="--profile $2"
                shift 2 ;;
            --profile-steps )
                PROFILE_STEPS_FLAG="--profile-steps $2"
                shift 2 ;;
            -te | --use-fp8-transformer-engine )
                TRANSFORMER_ENGINE_FLAG="--use-hpu-fp8-transformer-engine"
                shift 1 ;;
            -fsdpa | --use-fused-sdpa )
                USE_FUSED_SDPA="--use-fused-sdpa $2"
                shift 2 ;;
            -fsdpa-recompute | --use-fused-sdpa-with-recompute )
                USE_FUSED_SDPA_WITH_RECOMPUTE_ARG="$2"
                shift 2 ;;
            --fp8-measure-interval )
                FP8_MEASURE_INTERVAL="$2"
                shift 2 ;;
            --use-hpu-graphs )
                HPU_GRAPHS_FLAG="--use-hpu-graphs $2"
                shift 2 ;;
            --cache-fp8-weight-fwd )
                HPU_GRAPHS_FLAG="--cache-fp8-weight-fwd $2"
                shift 2 ;;
            --ext-train-iters )
                EXTERNAL_TRAINING_ITERATIONS="$2"
                shift 2 ;;
            -sp | --sequence-parallel )
                SEQUENCE_PARALLEL="$2"
                shift 2 ;;
            --device-warmup )
                DEVICE_WARMUP=$2
                shift 2 ;;
            --device-warmup-dataset-path )
                WARMUP_DATASET_PATH=$2
                shift 2 ;;
            --device-warmup-iterations )
                WARMUP_ITERATIONS=$2
                shift 2 ;;
            -- )
                shift
                break ;;
            * )
                if [[ -n "$1" ]]; then
                    echo "error: invalid parameter: $1"
                    exit -1
                fi
                break ;;
        esac
    done

}

function generate_hostsfile()
{
    HOSTS_PATH=$1
    HOSTSFILE_PATH=$2
    local num_nodes=${3:-8}

    rm -rf $HOSTSFILE_PATH
    touch $HOSTSFILE_PATH

    while IFS= read -r ip; do
        echo "$ip slots=$num_nodes" >> $HOSTSFILE_PATH
    done < "$HOSTS_PATH"

    echo "hostsfile: "
    cat $HOSTSFILE_PATH
}


# Default values for arguments, that can be overridden from cmd by parse_args func or env variable
DATA_DIR="/mnt/weka/data/mlperf_datasets/gpt-3/c4_mlperf_19_12_2022/preprocessed_c4_spm"
NUM_NODES=8
DEVICES_PER_NODE=8
DP=1
TP=8
PP=8
NUM_LAYERS=96
HIDDEN_SIZE=12288
NUM_ATTENTION_HEADS=96
SEQ_LENGTH=2048
DROPOUT=0.0
MICRO_BATCH=2
EVAL_MICRO_BATCH=8
GLOBAL_BATCH=2048
CLIP_GRAD=1.0
ZERO_STAGE=0
TRAIN_SAMPLES=84500000
LR=2.0e-5
MIN_LR=2.0e-6
LR_DECAY_SAMPLES=166809600
LR_WARMUP_SAMPLES=407040
SEED=${RANDOM}
EVAL_ITERS=-1
EVAL_INTERVAL=12
EXIT_INTERVAL=500
START_FROM_CKPT=true
SAVE_CKPT=true
SAVE_INTERVAL=500
LOG_INTERVAL=1
UNIVERSAL_CKPT_PATH="/mnt/weka/data/pytorch/gpt3/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101_universal4000"
OUTPUT_DIR=${OUTPUT_DIR:-"/tmp"}
HOSTS=""
HOSTSFILE="/root/shared/hostsfile"
MLLOG_FILE="/tmp/result_0.txt"
EVAL_LOSS_EXIT_VALUE=2.69
TRANSFORMER_ENGINE_FLAG=""
USE_FUSED_SDPA="--use-fused-sdpa true"
USE_FUSED_SDPA_WITH_RECOMPUTE_ARG="false"
FP8_MEASURE_INTERVAL=16
CACHE_FP8_WEIGHT_FWD_FLAG="--cache-fp8-weight-fwd true"
HPU_GRAPHS_FLAG="--use-hpu-graphs false"
ACCUMULATE_GRADS_VIA_HOOKS="true"
EXTERNAL_TRAINING_ITERATIONS=4000
EXTERNAL_GBS=1536
SEQUENCE_PARALLEL=true
DEVICE_WARMUP=true
WARMUP_DATASET_PATH="/mnt/weka/data/mlperf_datasets/gpt-3/synthetic_dataset/warmup_dataset"
WARMUP_ITERATIONS=5
CACHE_FP8_WEIGHT_FLAG="--cache-fp8-weight"

parse_args "$@"

if [ -f "$HOSTS" ]; then
    generate_hostsfile $HOSTS $HOSTSFILE 8
fi

# data and model dir paths
DATA_PATH_6=$DATA_DIR/c4_en_6_c4_spm_text_document
DATA_PATH_7=$DATA_DIR/c4_en_7_c4_spm_text_document
VALID_DATA_PATH=$DATA_DIR/c4_en_validation_c4_spm_text_document
MODEL_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# allow to override /proc file system in case it is mounted with different name on docker container
PROC_FS=${PROC_FS:-"/proc"}

# output log path
if [ -z "$OUTPUT_DIR" ]; then
  RUNTIME=`date +"%Y%m%d_%H%M"`
  OUTPUT_DIR=out/gpt3/ds_z${ZERO_STAGE}_nl${NUM_LAYERS}_hs${HIDDEN_SIZE}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_D${DP}_T${TP}_P${PP}_${RUNTIME}
fi
if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
fi

# saving checkpoint args
if [ $SAVE_CKPT = true ] || [ $SAVE_CKPT = 1 ]; then
    if [ -z "$SAVE_CKPT_DIR" ]; then
        SAVE_CKPT_DIR=$OUTPUT_DIR/checkpoints
    fi
    SAVE_CKPT_ARGS=" --save $SAVE_CKPT_DIR --save-interval $SAVE_INTERVAL "
fi

if [ "$DEVICE_WARMUP" == "true" ]; then
    DEVICE_WARMUP_ARG=" --device-warmup --warmup-dataset-path $WARMUP_DATASET_PATH --device-warmup-iterations $WARMUP_ITERATIONS"
fi

# handle kill switch argument
if [ -n "$KILL_SWITCH_FILE" ]; then
    KILL_SWITCH_ARG="--kill-switch-path $KILL_SWITCH_FILE"
fi

# Checkpoint loading configure
LOAD_CHECKPOINT_ARGS=""
if [ $START_FROM_CKPT = true ] || [ $START_FROM_CKPT = 1 ]; then
    CHECKPOINTS_BACKUP="$OUTPUT_DIR/../../checkpoints"
    if [ "$(ls -A $CHECKPOINTS_BACKUP 2>/dev/null)" ]; then
        LOAD_CHECKPOINT_ARGS=" --load $CHECKPOINTS_BACKUP "
    else
        LOAD_CHECKPOINT_ARGS=" --load $UNIVERSAL_CKPT_PATH --universal-checkpoint --no-load-rng "
    fi
fi

# Sequence parallelism
SEQUENCE_PARALLEL_ARG="--sequence-parallel"
PARTITIONED_MODE="false"
if [ $SEQUENCE_PARALLEL = false ]; then
    SEQUENCE_PARALLEL_ARG=""
    PARTITIONED_MODE="true"
fi

# Activation checkpointing or recompute
if [[ $USE_FUSED_SDPA_WITH_RECOMPUTE_ARG == "false" ]]; then
    ACTIVATION_CHECKPOINTING="--checkpoint-activations \
                              --checkpoint-activations-granularity=selective "
else
    ACTIVATION_CHECKPOINTING=""
fi

mkdir -p ${OUTPUT_DIR}
# create DS config
DS_CONFIG=${OUTPUT_DIR}/ds_config.json
cat << EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": $LOG_INTERVAL,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "gradient_clipping": $CLIP_GRAD,
  "bf16": {
    "enabled": true,
    "accumulate_grads_via_hooks": $ACCUMULATE_GRADS_VIA_HOOKS
    },

  "wall_clock_breakdown" : false,

  "pipeline": {
    "pipe_partitioned": $PARTITIONED_MODE,
    "grad_partitioned": $PARTITIONED_MODE
  }
}
EOT

echo "*******************************************************"
echo "Deepspeed config:"
cat $DS_CONFIG
echo "*******************************************************"

# DeepSpeed args
ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

CMD="sync && \
    if [ \"\$LOCAL_RANK\" -eq \"0\" ]; then echo 3 > $PROC_FS/sys/vm/drop_caches ; fi && \
    python -u $MODEL_DIR/pretrain_gpt.py \
    --use_hpu \
    --distributed-backend=hccl \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --optimizer fusedadamw \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --loss-scale 1 \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH \
    --eval-micro-batch-size $EVAL_MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-decay-style cosine \
    --train-samples $TRAIN_SAMPLES \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --log-interval $LOG_INTERVAL \
    --train-data-path 0.5 $DATA_PATH_6 0.5 $DATA_PATH_7 \
    --valid-data-path 1.0 $VALID_DATA_PATH \
    --eval-iters $EVAL_ITERS \
    --eval-interval $EVAL_INTERVAL \
    --vocab-file $DATA_DIR/vocab.json \
    --merge-file $DATA_DIR/merges.txt \
    --split 100,0,0 \
    --clip-grad $CLIP_GRAD \
    --attention-dropout $DROPOUT \
    --hidden-dropout $DROPOUT \
    --no-query-key-layer-scaling \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.1 \
    --init-method-std 0.006 \
    --seed $SEED \
    --bf16 \
    $ACTIVATION_CHECKPOINTING \
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-validation-ppl-to-tensorboard \
    --no-bias-gelu-fusion \
    --no-masked-softmax-fusion \
    --no-bias-dropout-fusion \
    --mask-tensor-adding \
    --fix-position-emb-redundant-alloc \
    --no-scaled-init \
    --no-seq-len-plus-one-tokens \
    --apply-layernorm-weight-plus-one \
    --do-layernorm-bias-weight-decay \
    --exit-interval $EXIT_INTERVAL \
    --DDP-impl local  \
    --mllog-output-path $MLLOG_FILE \
    --eval-loss-exit-value $EVAL_LOSS_EXIT_VALUE \
    --ext-lr-steps $(($EXTERNAL_TRAINING_ITERATIONS*$EXTERNAL_GBS)) \
    $LOAD_CHECKPOINT_ARGS \
    $SAVE_CKPT_ARGS \
    $KILL_SWITCH_ARG \
    $TRANSFORMER_ENGINE_FLAG \
    $USE_FUSED_SDPA \
    $DEVICE_WARMUP_ARG \
    --hpu-fp8-measure-interval $FP8_MEASURE_INTERVAL \
    $CACHE_FP8_WEIGHT_FWD_FLAG \
    $HPU_GRAPHS_FLAG \
    $CACHE_FP8_WEIGHT_FLAG \
    $PROFILE_FLAG \
    $PROFILE_STEPS_FLAG \
    $SEQUENCE_PARALLEL_ARG \
    $ds_args"


# configure multinode
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]
then
    MULTINODE_CMD="--force_multi \
                    --hostfile=$HOSTSFILE \
                    --master_addr $(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p) "
fi

# run gpt3
deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${DEVICES_PER_NODE} \
          --no_local_rank \
          --no_python \
          $MULTINODE_CMD \
          /usr/bin/bash -c "$CMD"
