#!/bin/bash

##########################################################################################
# Example: Bloom inferfence
##########################################################################################

# Params:
HOSTSFILE=${HL_HOSTSFILE:-"./scripts/hostsfile"}
WEIGHTS=$HL_WEIGHTS_ROOT/data/pytorch/bloom
MODEL="bloom"
DTYPE="bf16"
PROMPT="Does he know about phone hacking"
REPEAT=10
USE_GRAPHS="true"

BASE_DIR=`dirname -- "$0"`/..

# Params: DeepSpeed
NUM_NODES=${HL_NUM_NODES:-2}
NGPU_PER_NODE=8
DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
CMD="source ~/qnpu/pt/activate && \
     cd $DIR/.. && \
     PT_HPU_LAZY_ACC_PAR_MODE=0 PT_HPU_ENABLE_LAZY_COLLECTIVES=true python -u bloom.py \
     --weights $WEIGHTS \
     --model $MODEL \
     --dtype $DTYPE \
     -r $REPEAT \
     --use_graphs $USE_GRAPHS \
     --ignore_eos false \
     \"$PROMPT\""

#Configure multinode
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]
then
    MULTINODE_CMD="--hostfile=$HOSTSFILE \
                   --master_addr $(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p) "
fi


deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          $MULTINODE_CMD \
          /bin/bash -c "$CMD"
