#!/usr/bin/env bash

source weights/env.sh

log_dir="train_logs"
timestamp="$(date +%Y%m%d%H%M)"
log_file="${log_dir}/train_log_${timestamp}.log"

mkdir -p "${log_dir}"

accelerate launch --config_file "train_configs/ds_zero2.yaml" scripts/train.py \
                --config "train_configs/train_LucidFlux.yaml" \
                2>&1 | tee "${log_file}"
