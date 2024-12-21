#!/bin/bash

wandb login 04971049ca813b25cd6db3d781313e4ea63ffd0f

concept=${1} # dog6
placeholder_token=${2} # <dog6>
initializer_token=${3} # dog
superclass=${4} # dog
# conc_property=${5} # object / live
exp_num=${5}

export MODEL_NAME="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/stable-diffusion-2-base"
export INSTANCE_DIR="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/datasets/dataset/${concept}_one"
export OUTPUT_DIR="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/examples/core/res_CR/00${exp_num}-res-${concept}_CR"
export CLASS_DIR="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/examples/core/reg_stable_diff_2/${superclass}_class_dir"


accelerate launch train_db.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_data_dir=$INSTANCE_DIR \
  --test_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --placeholder_token="${placeholder_token}" \
  --class_name="${superclass}" \
  --instance_prompt="a photo of a ${placeholder_token}" \
  --class_prompt="a photo of a ${superclass}" \
  --initializer_token="${initializer_token}" \
  --exp_name="00${exp_num}-res-${concept}_CR" \
  --report_to="wandb" \
  --mixed_precision='no' \
  --resolution=512 \
  --seed=0 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=4000 \
  --checkpointing_steps=500 \
  --validation_steps=500 \
  --num_class_images=100 \
