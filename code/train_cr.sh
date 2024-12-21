#!/bin/bash

wandb login 04971049ca813b25cd6db3d781313e4ea63ffd0f

concept=${1} # dog6
placeholder_token=${2} # (<${concept}>), <dog6> 
initializer_token=${3} # dog
superclass=${4} # dog
concept_property=${5} # object / live
emb_reg=${6} # if use, set 1 else 0
attn_reg=${7} # if use, set 1 else 0
pairwise_reg=${8} # if use, set 1 else 0
exp_num=${9}

export MODEL_NAME="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/stable-diffusion-2-base"
export DATA_DIR="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/datasets/dataset/${concept}_one"
export OUTPUT_DIR="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/examples/core/res_CR/00${exp_num}-res-${concept}_CR"

accelerate launch train_cr.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --test_data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --placeholder_token="${placeholder_token}" \
  --initializer_token="${initializer_token}" \
  --class_name="${superclass}" \
  --concept_property="${concept_property}" \
  --exp_name="00${exp_num}-res-${concept}_CR" \
  --resolution=512 \
  --train_batch_size=3 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=600 \
  --learning_rate=2.5e-3 \
  --scale_lr \
  --seed=0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="wandb" \
  --num_validation_images=3 \
  --validation_steps=200 \
  --checkpointing_steps=200 \
  --lower_hinge_loss_bound=0.7 \
  --upper_hinge_loss_bound=0.8 \
  --add_emb_reg=$( [ "$emb_reg" -eq 1 ] && echo "True" || echo "False" ) \
  --add_attn_reg=$( [ "$attn_reg" -eq 1 ] && echo "True" || echo "False" ) \
  --add_pairwise_reg=$( [ "$attn_reg" -eq 1 ] && echo "True" || echo "False" ) \
  --add_hinge_reg=$( [ "$attn_reg" -eq 1 ] && echo "True" || echo "False" ) \
