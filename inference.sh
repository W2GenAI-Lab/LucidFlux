#!/usr/bin/env bash
set -e

# Minimal, readable, zero-logic wrapper. All paths are relative.
# Ensure you've prepared weights beforehand (e.g., source weights/env.sh for FLUX base).

python inference.py \
  --checkpoint weights/lucidflux/lucidflux.pth \
  --control_image assets/3.png \
  --prompt "restore this image into high-quality, clean, high-resolution result" \
  --output_dir outputs \
  --width 1024 \
  --height 1024 \
  --num_steps 20 \
  --swinir_pretrained weights/swinir.pth \
  --siglip_ckpt weights/siglip
