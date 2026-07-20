#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET="${DATASET:-Indian}"
SEED="${SEED:-0}"
SPLIT_DIR="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/splits}"
DATA_PATH="${DATA_PATH:-${SPLIT_DIR}/${DATASET}_stratified_80_20_seed${SEED}.mat}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/home/lxdcis/hypersl_training/hypersl/modelarchive/${DATASET,,}_mae_aug}"
CHECKPOINT="${CHECKPOINT:-${PRETRAIN_DIR}/embed128_enc8_dec8_heads8_mask75_epoch300.pt}"

WANDB_PROJECT="${WANDB_PROJECT:-HyperSL-LinearProbeAug}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-hypersl_${DATASET,,}_linear_probe_aug_seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY_ARGS=()

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ENTITY_ARGS=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing linear-probe data file: ${DATA_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing pretrained checkpoint: ${CHECKPOINT}" >&2
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
  --patch-size 1 \
  --epochs 400 \
  --batch-size 64 \
  --test-batch-size 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --split-seed "${SEED}" \
  --eval-every 10 \
  --linear-probe \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-mode "${WANDB_MODE}" \
  "${WANDB_ENTITY_ARGS[@]}"
