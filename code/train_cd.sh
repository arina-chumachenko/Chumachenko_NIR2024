#!/bin/bash

wandb login 04971049ca813b25cd6db3d781313e4ea63ffd0f

concept=${1} # dog6
modifier_token=${2} # <new1>
initializer_token=${3} # dog
superclass=${4} # dog
exp_num=${5}

export MODEL_NAME="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/stable-diffusion-2-base"
export INSTANCE_DIR="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/datasets/dataset/${concept}_one"
export OUTPUT_DIR="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/examples/core/res_CR/00${exp_num}-res-${concept}_CR"
export CLASS_DIR="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/examples/core/reg_stable_diff_2/${superclass}_class_dir"


accelerate launch train_cd.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --modifier_token="${modifier_token}" \
  --class_prompt="${superclass}" \
  --instance_prompt="a photo of a ${modifier_token} ${superclass}" \
  --initializer_token="${initializer_token}" \
  --exp_name="00${exp_num}-res-${concept}_CR" \
  --report_to="wandb" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --checkpointing_steps=50 \
  --validation_steps=50 \
  --validation_prompt="a ${modifier_token} ${superclass} in the desert" \
  --num_class_images=100 \
  --scale_lr \
  --hflip \
