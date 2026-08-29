#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repository_root}"

python_bin="${repository_root}/myenv/bin/python"
sample_root="${repository_root}/data/salaad_vision/smoke/imagenet_val64"
manifest_path="${sample_root}/manifest.jsonl"
output_dir="${repository_root}/data/figures/salaad_vision/self_attention/mixed_rho_all_20260828_154729"
mixed_rho_dir="${repository_root}/data/salaad_vision/vit_b8_all_qkv_rho5e6_fc_rho5e8/20260828_154729"

image_args=(
  --image
  "${sample_root}/class_0091/val_00000_label_0091.jpg"
  "${sample_root}/class_0171/val_00001_label_0171.jpg"
  "${sample_root}/class_0980/val_00002_label_0980.jpg"
  "${sample_root}/class_0505/val_00013_label_0505.jpg"
  "${sample_root}/class_0817/val_00019_label_0817.jpg"
  "${sample_root}/class_0046/val_00052_label_0046.jpg"
)

common_args=(
  "${image_args[@]}"
  --manifest "${manifest_path}"
  --output-dir "${output_dir}"
  --attention-mass 0.60
  --batch-size 6
  --device cuda
)

"${python_bin}" -u scripts/visualize_dino_attention.py \
  "${common_args[@]}" \
  --checkpoint "${repository_root}/data/salaad_vision/pretrained/dino_vitbase8_pretrain.pth" \
  --checkpoint-kind teacher_backbone \
  --model-label teacher \
  --title-label Teacher

"${python_bin}" -u scripts/visualize_dino_attention.py \
  "${common_args[@]}" \
  --checkpoint "${repository_root}/data/salaad_vision/vit_b8_vanilla/20260803_101747/model.pth" \
  --checkpoint-kind student_model \
  --model-label vanilla \
  --title-label Vanilla

"${python_bin}" -u scripts/visualize_dino_attention.py \
  "${common_args[@]}" \
  --checkpoint "${mixed_rho_dir}/model.pth" \
  --checkpoint-kind student_model \
  --model-label salaad_x \
  --title-label Mixed-rho-X

"${python_bin}" -u scripts/visualize_dino_attention.py \
  "${common_args[@]}" \
  --checkpoint "${mixed_rho_dir}/model.pth" \
  --checkpoint-kind student_model \
  --matrix-dir "${mixed_rho_dir}" \
  --matrix-component l_plus_s \
  --matrix-layer-group all \
  --model-label salaad_l_plus_s \
  --title-label Mixed-rho-L+S

"${python_bin}" -u scripts/visualize_dino_attention.py \
  "${common_args[@]}" \
  --checkpoint "${mixed_rho_dir}/model.pth" \
  --checkpoint-kind student_model \
  --matrix-dir "${mixed_rho_dir}" \
  --matrix-component l_only \
  --matrix-layer-group all \
  --model-label salaad_l_only \
  --title-label Mixed-rho-L-only

"${python_bin}" -u scripts/visualize_dino_attention.py \
  "${common_args[@]}" \
  --checkpoint "${mixed_rho_dir}/model.pth" \
  --checkpoint-kind student_model \
  --matrix-dir "${mixed_rho_dir}" \
  --matrix-component s_only \
  --matrix-layer-group all \
  --model-label salaad_s_only \
  --title-label Mixed-rho-S-only

