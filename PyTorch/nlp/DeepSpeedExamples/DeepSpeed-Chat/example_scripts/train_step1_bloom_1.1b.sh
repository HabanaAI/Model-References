#!/bin/bash
# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company


# -----------------------------------------------------------------------
# RLHF step1 reference training script for Bloom-1.1B model
#
# The script contains FIXED training parameters.
#
# Arguments:
#  tag - tag name added to the artifacts of this run
#  base_out_path - base path for artifacts
#  n_nodes - number of nodes
#  n_devices_per_node - number of devices at each node
#  ckp_act - checkpoint activations. 0 for disabled, 1 for enabled
#  seed - seed value
#  mbs - micro batch size
#  tensorboard_path - tensorboard path. empty string for default
#  log_file - log full filename. empty string for default
#  model_name_or_path - HF-style model name or path
#  master_port - deepspeed runner master_port
#  dataset_path - HF path or list of paths to the dataset
# -----------------------------------------------------------------------

set -ex

DATA_DIR_ROOT=${HL_DATA_DIR_ROOT:-/mnt/weka}
tag=${HL_TAG:-default_tag}
base_out_path=${HL_BASE_OUT_PATH:-/root/logs}
n_nodes=${HL_NUM_NODES:-1}
n_devices_per_node=${HL_DEVICES_PER_NODE:-8}
act_zero_stage=${HL_ACTOR_ZERO_STAGE:-1}
ckp_act=${HL_ACTOR_CP_ACT:-0}
seed=${HL_SEED:-10}
mbs=${HL_MBS:-8}
tensorboard_path=${HL_TENSORBOARD_PATH:-}
log_file=${HL_LOG_FILE:-}
master_port=${HL_MASTER_PORT:-29500}
model_name_or_path=${HL_ACTOR_MODEL:-${DATA_DIR_ROOT}/data/pytorch/deepspeed-chat/models/bloom1b1}
dataset_path=${HL_DATASET_PATH}


if [ ! -d "$model_name_or_path" ]; then
  echo "fallback to HF as not a folder"
  echo $model_name_or_path
  model_name_or_path="bigscience/bloom-1b1"
fi

# fixed training parameters
LR=2e-5
DROPOUT=0.0
WD=0.0
GB=256
EPOCHS=2

# Calculate GAS given global batch, n_nodes, n_devices_per_node
total_devices=$(($n_nodes*$n_devices_per_node))
per_device_batch=$(($GB/$total_devices))
gas=$(($per_device_batch/$mbs))

# set gradient checkpointing arguments
ckp_act_args=""
if [ "$ckp_act" -eq "1" ]; then
  ckp_act_args="--gradient_checkpointing "
fi

# setup checkpoint, tensorboard and log path
prefix_name=${tag}/bloom/step1/1.1b
run_name=gb_${GB}_mbs_${mbs}_lr_${LR}_do_${DROPOUT}_wd_${WD}_ep_${EPOCHS}
checkpoint_path=${base_out_path}/checkpoints/${prefix_name}/${run_name}

if [ -z "$tensorboard_path" ]; then
  tensorboard_path=${base_out_path}/tensorboard/${prefix_name}
fi

if [ -z "$log_file" ]; then
  log_file=${base_out_path}/logs/${prefix_name}/${run_name}.txt
fi

# if using a single device, set both n_nodes and n_devices_per_node to default
# otherwise, deepspeed ignores CUDA_VISIBLE_DEVICES configuration
if [[ "$n_nodes" -eq 1 ]] && [[ "$n_devices_per_node" -eq 1 ]]; then
  n_nodes=-1
  n_devices_per_node=-1
fi

# create required paths
# if log-file/tb-path provided, caller should make sure directories exist
mkdir -p ${base_out_path}/logs/${prefix_name}

# RUN
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
training_dir=$( realpath $script_dir/../training)
cd ${training_dir}

python -m deepspeed.launcher.runner --num_nodes ${n_nodes} --num_gpus ${n_devices_per_node} --master_port ${master_port} \
    step1_supervised_finetuning/main.py \
        --model_name_or_path ${model_name_or_path} \
        --data_path ${dataset_path} \
        --data_cached_path $DATA_DIR_ROOT/data/pytorch/deepspeed-chat/datasets/stage1 \
        --bf16 \
        --learning_rate ${LR} \
        --dropout ${DROPOUT} \
        --weight_decay ${WD} \
        --per_device_train_batch_size ${mbs} \
        --gradient_accumulation_steps ${gas} \
        --num_train_epochs ${EPOCHS} \
        --num_warmup_steps 20 \
        --zero_stage ${act_zero_stage} \
        ${ckp_act_args} \
        --per_device_eval_batch_size 8 \
        --seed ${seed} \
        --deepspeed \
        --output_dir ${checkpoint_path} \
        --tb_output_dir ${tensorboard_path} \
        --tb_job_name ${run_name} \
        --no_fused_kernels \
    |& tee ${log_file}
exit $PIPESTATUS
