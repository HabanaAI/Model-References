# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.

#!/bin/bash


set -ex

# ----------------------
# Configurable parameters
DATA_DIR=${HL_DATA_DIR_ROOT:-/data/bigscience/oscar-en/}
DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-meg-gpt2_text_document}
NUM_NODES=${HL_NUM_NODES:-1}
DP=${HL_DP:-1}
TP=${HL_TP:-4}
PP=${HL_PP:-2}
MICRO_BATCH=${HL_MICRO_BATCH:-1}
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH:-}
HOSTSFILE=${HL_HOSTSFILE:-}
USE_HPU=${HL_USE_HPU:-1}
CKP_ACT=${HL_CKP_ACT:-0}
RAMPUP_BS=${HL_RAMPUP_BS:-1}
UNIV_CP=${HL_UNIV_CP:-0}
QNPU_DIR=${HL_QNPU_DIR:-}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
LLAMA_VER=${HL_LLAMA_VER:-13}
# ----------------------

if [[ -z "$MODEL_REFERENCES_ROOT" ]]; then
    echo "Must provide MODEL_REFERENCES_ROOT in environment" 1>&2
    exit 1
fi

DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}
MODEL_DIR=$MODEL_REFERENCES_ROOT/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed

# Scaling
NGPU_PER_NODE=8
NUM_GPUs=$(($DP * $TP * $PP))

if [ $LLAMA_VER -eq 13 ]; then
    # Llama-13B model architecture
    NLAYERS=40 # must be divisible by PP
    NHIDDEN=5120
    NHEADS=40 # must be divisible by TP
    FFN_HIDDEN_SIZE=13824
else
    # Llama-7B model architecture
    NLAYERS=32 # must be divisible by PP
    NHIDDEN=4096
    NHEADS=32 # must be divisible by TP
    FFN_HIDDEN_SIZE=11008
fi
SEQ_LEN=2048

# Training parameters
GLOBAL_BATCH=256
ZERO_STAGE=0

# output paths
if [ -z "$OUTPUT_DIR" ]; then
    RUNTIME=`date +"%Y%m%d_%H%M"`
    OUTPUT_DIR=out/llama13b/ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_D${DP}_T${TP}_P${PP}_GPUs${NUM_GPUs}_${RUNTIME}
fi

if [ -z "$CHECKPOINTS_DIR" ]; then
    CHECKPOINTS_DIR=$OUTPUT_DIR/checkpoints
fi

if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${TENSORBOARD_DIR}

# handle kill switch argument
if [ -z "$KILL_SWITCH_FILE"]; then
    KILL_SWITCH_ARG=""
else
    KILL_SWITCH_ARG="--kill-switch-path $KILL_SWITCH_FILE"
fi

# create DS config
DS_CONFIG=${OUTPUT_DIR}/ds_config.json
cat << EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": $LOG_INTERVAL,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {"enabled": true},
  "fp16": {"enabled": false},
  "wall_clock_breakdown": false
}
EOT

# configure multi-node
MULTINODE_CMD=""
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]; then
    MULTINODE_CMD="--hostfile=$HOSTSFILE \
                   --master_addr $(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p) "
fi

# training script command
CMD=""
if [ ! -z "$QNPU_DIR" ]; then
    CMD="source ${QNPU_DIR}/activate ;"
fi

CMD="${CMD} \
    cd $MODEL_DIR && \
    ENABLE_EXPERIMENTAL_FLAGS=1 SPILL_PERSISTENT_TENSORS=0 python -u ./pretrain_llama.py \
    --deepspeed \
    --verify-checkpoint-model-type LLAMA \
    --verify-checkpoint \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --position-embedding-type rotary \
    --no-bias \
    --layernorm-type rmsnorm \
    --activation-func-type swiglu \
    --layernorm-epsilon 1e-6 \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --micro-batch-size ${MICRO_BATCH} \
    --global-batch-size ${GLOBAL_BATCH} \
    --train-iters 10000 \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters 10 \
    --eval-interval 100 \
    --data-path ${DATA_PATH} \
    --vocab-file $DATA_DIR/gpt2-vocab.json \
    --merge-file $DATA_DIR/gpt2-merges.txt \
    --optimizer adamw \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-6 \
    --lr 3e-4 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-validation-ppl-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --load $CHECKPOINTS_DIR \
    --deepspeed_config=$DS_CONFIG  \
    --zero-stage=$ZERO_STAGE \
    --exit-interval $EXIT_INTERVAL \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    $KILL_SWITCH_ARG \
    --bf16"

if [ $USE_HPU -eq 1 ]
then
    CMD="${CMD} --use_hpu --distributed-backend=hccl"
fi

if [ $UNIV_CP -eq 1 ]
then
    echo "Loading Universal Checkpoint from ${CHECKPOINTS_DIR}"
    CMD="${CMD} --universal-checkpoint"
fi

#if [ $RAMPUP_BS -eq 1 ]
#then
#    CMD="${CMD} --rampup-batch-size 16 16 5_000_000"
#fi

if [ $CHECKPOINT_SAVE -eq 1 ]
then
    mkdir -p ${CHECKPOINTS_DIR}
    CMD="${CMD} --save $CHECKPOINTS_DIR --save-interval $SAVE_INTERVAL"
fi

if [ $CKP_ACT -eq 1 ]
then
    CMD="${CMD} --checkpoint-activations --deepspeed-activation-checkpointing"
fi

if [ ! -z "$QNPU_DIR" ]; then
    rm -rf $HOME/.deepspeed_env
fi
# configure deepspeed env
#echo "PT_ENABLE_COMM_GROUP_CACHE=true" >> $HOME/.deepspeed_env
#echo "PT_HPU_LAZY_ACC_PAR_MODE=0" >> $HOME/.deepspeed_env

# run!
deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          $MULTINODE_CMD \
          /usr/bin/bash -c "$CMD"

