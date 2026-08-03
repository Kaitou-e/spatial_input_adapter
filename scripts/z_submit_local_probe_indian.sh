#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPERSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${HYPERSL_DIR}"

DATASET="${DATASET:-IndianPine}"
SEED="${SEED:-0}"
DATA_PATH="${DATA_PATH:-/home/lxdcis/hypersl_training/data/${DATASET}.mat}"
CHECKPOINT="${CHECKPOINT:-/home/lxdcis/hypersl_training/hypersl/modelarchive/indianpine_mae_aug/embed128_enc8_dec8_heads8_mask80_epoch200.pt}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing data file: ${DATA_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

python hypersl_linear_probe.py \
  --checkpoint "${CHECKPOINT}" \
  --data-path "${DATA_PATH}" \
  --model-size small \
  --embedding-dim 128 \
  --encoder-depth 8 \
  --decoder-depth 8 \
  --num-heads 8 \
  --head-type local_attention \
  --freeze-encoder \
  --patch-size 7 \
  --spatial-depth 2 \
  --spatial-heads 4 \
  --spatial-mlp-ratio 4 \
  --spatial-dropout 0.1 \
  --epochs 200 \
  --batch-size 4 \
  --test-batch-size 16 \
  --grad-accum-steps 8 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --split-seed "${SEED}" \
  --eval-every 10 \
  --wandb \
  --wandb-project "${WANDB_PROJECT:-HyperSL-LocalProbe}" \
  --wandb-run-name "${WANDB_RUN_NAME:-hypersl_indian_local_attention_frozen_seed${SEED}}" \
  --wandb-mode "${WANDB_MODE:-online}"
