#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPERSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${HYPERSL_DIR}"

DATASET="${DATASET:-Houston}"
SEED="${SEED:-0}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
ENCODER_DEPTH="${ENCODER_DEPTH:-8}"
DECODER_DEPTH="${DECODER_DEPTH:-8}"
NUM_HEADS="${NUM_HEADS:-8}"
MASK_RATIO="${MASK_RATIO:-0.75}"
SPLIT_DIR="${SPLIT_DIR:-/home/dk6012/Desktop/darpa_ai/data/splits}"
DATA_PATH="${DATA_PATH:-${SPLIT_DIR}/augmented/${DATASET}_stratified_80_20_seed${SEED}_pretrain_C_plus_D.mat}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/dk6012/Desktop/darpa_ai/hypersl/modelarchive/${DATASET,,}_mae_aug}"
WANDB_PROJECT="${WANDB_PROJECT:-HyperSL-Aug}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-hypersl_${DATASET,,}_pretrain_aug_seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ENTITY_ARGS=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing pretraining data file: ${DATA_PATH}" >&2
  exit 1
fi

python hypersl_trainer.py \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --split all \
  --epochs "${PRETRAIN_EPOCHS}" \
  --batch-size 128 \
  --lr 5e-4 \
  --weight-decay 5e-5 \
  --mask-ratio "${MASK_RATIO}" \
  --embedding-dim "${EMBEDDING_DIM}" \
  --encoder-depth "${ENCODER_DEPTH}" \
  --decoder-depth "${DECODER_DEPTH}" \
  --num-heads "${NUM_HEADS}" \
  --num-workers 0 \
  --seed "${SEED}" \
  --disable-tensorboard \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-mode "${WANDB_MODE}" \
  "${WANDB_ENTITY_ARGS[@]}"
