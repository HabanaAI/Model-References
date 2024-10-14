###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################
#!/bin/bash
export PATH=$PATH:~/.local/bin
#export WANDB_API_KEY=
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export INSTANCE_DIR='data/bo'
export OUTPUT_DIR='results/bo'
#export MASK_DIR=$SM_CHANNEL_MASK
export TRAIN_STEPS_TUNING=1000
export SM_MODEL_DIR=${OUTPUT_DIR}
mkdir -p $OUTPUT_DIR

PT_HPU_LAZY_MODE=0 \
lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --use_face_segmentation_condition \
  --resolution=512 \
  --train_batch_size=7 \
  --gradient_accumulation_steps=1 \
  --learning_rate_unet=5e-5 \
  --learning_rate_ti=2e-3 \
  --color_jitter \
  --lr_scheduler="linear" --lr_scheduler_lora="linear"\
  --lr_warmup_steps=0 \
  --placeholder_tokens="<s1>|<s2>" \
  --use_template="object"\
  --save_steps=50 \
  --max_train_steps_ti=500 \
  --max_train_steps_tuning=$TRAIN_STEPS_TUNING \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-3 \
  --device="hpu" \
  --lora_rank=16 \
  --use_fused_adamw=True \
  --use_lazy_mode=False \
  --print_freq=1 \
  --use_fused_clip_norm=True \
  --use_torch_compile=True \
  --lora_clip_target_modules="{'CLIPSdpaAttention'}" \
  2>&1 |tee log_1x_ft_hpu_compile.txt



cp $OUTPUT_DIR/step_$TRAIN_STEPS_TUNING.safetensors $SM_MODEL_DIR/lora.safetensors
