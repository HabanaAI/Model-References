#!/bin/bash
# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company

# -----------------------------------------------------------------------
# RLHF step3 reference training script for Bloom-1.1B + Bloom-560m models
#
# The script contains FIXED training parameters.
#
# Arguments:
#  tag - tag name added to the artifacts of this run
#  base_out_path - base path for artifacts
#  act_model_path - path for actor 1.1B model checkpoint from step1
#  cri_model_path - path for critic 560m model checkpoint from step2
#  dataset_path - HF path or list of paths to the dataset
#  n_nodes - number of nodes
#  n_devices_per_node - number of devices at each node
#  act_ckp_act - actor checkpoint activations. 0 for disabled, 1 for enabled
#  cri_ckp_act - critic checkpoint activations. 0 for disabled, 1 for enabled
#  seed - seed value
#  mbs - micro batch size
#  gas - gradient accumulation steps. 0 for auto calculated
#  tensorboard_path - tensorboard path. empty string for default
#  log_file - log full filename. empty string for default
#  master_port - deepspeed runner master_port
# -----------------------------------------------------------------------

set -ex

tag=${HL_TAG-default_tag}
base_out_path=${HL_BASE_OUT_PATH:-/root/logs}
act_model_path=${HL_ACTOR_MODEL_PATH}
cri_model_path=${HL_CRITIC_MODEL_PATH}
dataset_path=${HL_DATASET_PATH}
n_nodes=${HL_NUM_NODES:-1}
n_devices_per_node=${HL_DEVICES_PER_NODE:-8}
act_ckp_act=${HL_ACTOR_CP_ACT:-0}
cri_ckp_act=${HL_CRITIC_CP_ACT:-0}
act_zero_stage=${HL_ACTOR_ZERO_STAGE:-1}
cri_zero_stage=${HL_CRITIC_CP_ACT:-1}
seed=${HL_SEED:-10}
mbs=${HL_MBS:-2}
tensorboard_path=${HL_TENSORBOARD_PATH:-}
log_file=${HL_LOG_FILE:-}
master_port=${HL_MASTER_PORT:-29500}

# fixed training parameters
ACT_LR=1e-5
ACT_DROPOUT=0.0
ACT_WD=0.1
CRI_LR=2e-5
CRI_DROPOUT=0.0
CRI_WD=0.0
GB=32
EPOCHS=1
LORA_DIM=0

# Calculate GAS given global batch, n_nodes, n_devices_per_node
total_devices=$(($n_nodes*$n_devices_per_node))
per_device_batch=$(($GB/$total_devices))
gas=$(($per_device_batch/$mbs))

# set gradient checkpointing arguments
ckp_act_args=""
if [ "$act_ckp_act" -eq "1" ]; then
  ckp_act_args="--actor_gradient_checkpointing "
fi
if [ "$cri_ckp_act" -eq "1" ]; then
  ckp_act_args="$ckp_act_args --critic_gradient_checkpointing "
fi

# setup checkpoint, tensorboard and log path
prefix_name=${tag}/bloom/step3/1.1b_560m
run_name=gb_${GB}_mbs_${mbs}_ep_${EPOCHS}_act_lr_${ACT_LR}_do_${ACT_DROPOUT}_wd_${ACT_WD}_cri_lr_${CRI_LR}_do_${CRI_DROPOUT}_wd_${CRI_WD}_lora_${LORA_DIM}
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
    step3_rlhf_finetuning/main.py \
      --bf16 \
      --actor_model_name_or_path ${act_model_path} \
      --critic_model_name_or_path ${cri_model_path} \
      --data_path ${dataset_path} \
      --actor_zero_stage ${act_zero_stage} \
      --critic_zero_stage ${cri_zero_stage} \
      --num_padding_at_beginning 0 \
      --per_device_train_batch_size ${mbs} \
      --per_device_mini_train_batch_size ${mbs} \
      --gradient_accumulation_steps ${gas} \
      --actor_learning_rate ${ACT_LR} \
      --critic_learning_rate ${CRI_LR} \
      --actor_weight_decay ${ACT_WD} \
      --critic_weight_decay ${CRI_WD} \
      --actor_dropout ${ACT_DROPOUT} \
      --critic_dropout ${CRI_DROPOUT} \
      ${ckp_act_args} \
      --seed ${seed} \
      --deepspeed \
      --output_dir ${checkpoint_path} \
      --tb_output_dir ${tensorboard_path} \
      --tb_job_name_actor act_${run_name} \
      --tb_job_name_critic cri_${run_name} \
      --no_fused_kernels \
      --enable_hpu_graphs \
    |& tee ${log_file}
exit $PIPESTATUS
