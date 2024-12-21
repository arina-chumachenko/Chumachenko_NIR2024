#!/bin/bash

wandb login 04971049ca813b25cd6db3d781313e4ea63ffd0f

OUTPUT_DIR="/home/jovyan/shares/SR006.nfs2/chumachenko/diffusers/examples/core/res_CR"
exp_name=${1} # 0042-res-dog6_CR
checkpoint_idx=${2}

python inference.py \
  --pretrained_model_name="${OUTPUT_DIR}/${exp_name}" \
  --config_path "${OUTPUT_DIR}/${exp_name}/logs/hparams.yml" \
  --checkpoint_idx "${checkpoint_idx}" 