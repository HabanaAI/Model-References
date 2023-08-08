# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.

#!/bin/bash

#This is based on https://github.com/bigscience-workshop/bigscience/blob/master/train/tr1-13B-base/tr1-13B-round1.slurm

set -ex

# ----------------------
# Configurable parameters
DATA_DIR=${HL_DATA_DIR_ROOT:-/data/bigscience/oscar-en}
NUM_NODES=${HL_NUM_NODES:-1}
DP=${HL_DP:-2}
TP=${HL_TP:-4}
PP=${HL_PP:-1}
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
# ----------------------

if [[ -z "$MODEL_REFERENCES_ROOT" ]]; then
    echo "Must provide MODEL_REFERENCES_ROOT in environment" 1>&2
    exit 1
fi

DATA_PATH=${DATA_DIR}/meg-gpt2_text_document
MODEL_DIR=$MODEL_REFERENCES_ROOT/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed

# Scaling
NGPU_PER_NODE=8
NUM_GPUs=$(($DP * $TP * $PP))

# Bloom-13B model architecture
NLAYERS=40 # must be divisible by PP
NHIDDEN=5120
NHEADS=32 # must be divisible by TP
FFN_HIDDEN_SIZE=$(($NHIDDEN * 4))
SEQ_LEN=2048

# Training parameters
GLOBAL_BATCH=1024
ZERO_STAGE=0

# output paths
if [ -z "$OUTPUT_DIR" ]; then
    RUNTIME=`date +"%Y%m%d_%H%M"`
    OUTPUT_DIR=out/bloom13b/ds_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_D${DP}_T${TP}_P${PP}_GPUs${NUM_GPUs}_${RUNTIME}
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
    ENABLE_EXPERIMENTAL_FLAGS=1 SPILL_PERSISTENT_TENSORS=0 python -u ./pretrain_gpt.py \
    --deepspeed \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size ${MICRO_BATCH} \
    --global-batch-size ${GLOBAL_BATCH} \
    --train-samples 300_000_000 \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters 5 \
    --eval-interval 100 \
    --data-path ${DATA_PATH} \
    --data-impl mmap \
    --split 949,50,1 \
    --vocab-file $DATA_DIR/gpt2-vocab.json \
    --merge-file $DATA_DIR/gpt2-merges.txt \
    --optimizer adamw \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples 126_953_125 \
    --lr-warmup-samples 216_320 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-validation-ppl-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --load $CHECKPOINTS_DIR \
    --deepspeed_config=$DS_CONFIG  \
    --zero-stage=$ZERO_STAGE \
    --seed 42 \
    --exit-interval $EXIT_INTERVAL \
    --no-masked-softmax-fusion \
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

if [ $RAMPUP_BS -eq 1 ]
then
    CMD="${CMD} --rampup-batch-size 16 16 5_000_000"
fi

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
          /usr/bin/bash -c "$CMD" #2>&1 | tee ${OUTPUT_DIR}/output.log
