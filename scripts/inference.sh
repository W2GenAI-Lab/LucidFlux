#!/usr/bin/env bash
set -e

# Minimal, readable, zero-logic wrapper. All paths are relative.
# Ensure you've prepared weights beforehand (e.g., source weights/env.sh for FLUX base).

source weights/env.sh

# # test
# python scripts/inference.py \
#   --checkpoint weights/lucidflux/lucidflux.pth \
#   --control_image assets/3.png \
#   --output_dir outputs \
#   --width 1024 \
#   --height 1024 \
#   --num_steps 20 \
#   --swinir_pretrained weights/swinir.pth \
#   --siglip_ckpt weights/siglip \
#   --offload

# # 2k output
# python scripts/inference-2k.py \
#   --checkpoint weights/lucidflux/lucidflux.pth \
#   --control_image assets/3.png \
#   --output_dir outputs-2k \
#   --width 2048 \
#   --height 2048 \
#   --num_steps 20 \
#   --swinir_pretrained weights/swinir.pth \
#   --siglip_ckpt weights/siglip \
#   --offload


# # evaluation checkpoint
# python scripts/inference.py \
#   --checkpoint lucidflux_checkpoint/checkpoint-10/lucidflux.pth \
#   --control_image assets/3.png \
#   --output_dir evaluation/lucidflux_checkpoint/checkpoint-10 \
#   --width 1024 \
#   --height 1024 \
#   --num_steps 20 \
#   --swinir_pretrained weights/swinir.pth \
#   --siglip_ckpt weights/siglip \
#   --offload

# PiD decode for 4k generation, width and height here are only for LucidFlux output
python scripts/inference_pid.py \
  --checkpoint weights/lucidflux/lucidflux.pth \
  --control_image assets \
  --output_dir outputs-4k \
  --width 1024 \
  --height 1024 \
  --num_steps 20 \
  --swinir_pretrained weights/swinir.pth \
  --siglip_ckpt weights/siglip \
  --pid_ckpt_type 2kto4k \
  --pid_cfg_scale 1 \
  --pid_inference_steps 4 \
  --offload
