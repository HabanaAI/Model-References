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
#export HABANA_PROFILE=1

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
  --use_lazy_mode=True \
  --use_fused_adamw=True \
  --print_freq=50 \
  --use_fused_clip_norm=True \
  2>&1 |tee log_1x_ft.txt


  #--log_tb=True \
  #--use_synapse_profiler=True \
  #--use_pytorch_profiler=True \
  #--profiler_step=7 \
  #--profile_ti=False \
  #--profile_tuning=True \
  #--enable_xformers_memory_efficient_attention \
  #--log_wandb=True \
  #--mixed_precision="fp16"
  #--ema_decay=0.9 \
  #--train_timesteps_percentage=0.8 \
  #--cache_instance_latent \
  #--lr_scheduler_span=.5 \
  #--preprocessed_mask_dir=$MASK_DIR \
  #--use_preprocessed_mask \

cp $OUTPUT_DIR/step_$TRAIN_STEPS_TUNING.safetensors $SM_MODEL_DIR/lora.safetensors
