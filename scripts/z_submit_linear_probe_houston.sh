#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPERSL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${HYPERSL_DIR}"

DATASET="${DATASET:-Houston}"
SEED="${SEED:-0}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
PROBE_EPOCHS="${PROBE_EPOCHS:-400}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
ENCODER_DEPTH="${ENCODER_DEPTH:-8}"
DECODER_DEPTH="${DECODER_DEPTH:-8}"
NUM_HEADS="${NUM_HEADS:-8}"
MASK_TAG="${MASK_TAG:-75}"
SPLIT_DIR="${SPLIT_DIR:-/home/dk6012/Desktop/darpa_ai/data/splits}"
DATA_PATH="${DATA_PATH:-${SPLIT_DIR}/${DATASET}_stratified_80_20_seed${SEED}.mat}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/home/dk6012/Desktop/darpa_ai/hypersl/modelarchive/${DATASET,,}_mae_aug}"
CHECKPOINT="${CHECKPOINT:-${PRETRAIN_DIR}/embed${EMBEDDING_DIM}_enc${ENCODER_DEPTH}_dec${DECODER_DEPTH}_heads${NUM_HEADS}_mask${MASK_TAG}_epoch${PRETRAIN_EPOCHS}.pt}"
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
  --embedding-dim "${EMBEDDING_DIM}" \
  --encoder-depth "${ENCODER_DEPTH}" \
  --decoder-depth "${DECODER_DEPTH}" \
  --num-heads "${NUM_HEADS}" \
  --patch-size 1 \
  --epochs "${PROBE_EPOCHS}" \
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
