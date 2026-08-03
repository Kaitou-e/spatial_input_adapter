#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATASET="${DATASET:-IndianPine}"
SEED="${SEED:-0}"
# SPLIT_DIR="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/splits}"
DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/${DATASET}.mat}"
# DATA_PATH="${DATA_PATH:-${SPLIT_DIR}/${DATASET}_stratified_80_20_seed${SEED}.mat}"
# PRETRAIN_DIR="${PRETRAIN_DIR:-/home/lxdcis/hypersl_training/hypersl/modelarchive/${DATASET,,}_mae_aug}"
# CHECKPOINT="${CHECKPOINT:-${PRETRAIN_DIR}/embed128_enc8_dec8_heads8_mask80_epoch200.pt}"
CHECKPOINT="/home/lxdcis/hypersl_training/hypersl/modelarchive/10_base_mask95_checkpoint.pt"
WAVELENGTHS_PATH="${WAVELENGTHS_PATH:-/home/lxdcis/hypersl_training/data/indian_pines_wavelengths_220.csv}"

WANDB_PROJECT="${WANDB_PROJECT:-HyperSL-LinearProbeLoc}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-hypersl_${DATASET,,}_loc_smaller_seed${SEED}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY_ARGS=()

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ENTITY_ARGS=(--wandb-entity "${WANDB_ENTITY}")
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing linear-probe data file: ${DATA_PATH}" >&2
  exit 1
fi

if [[ ! -f "${WAVELENGTHS_PATH}" ]]; then
  echo "Missing wavelength file: ${WAVELENGTHS_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing pretrained checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

# OLD
# python hypersl_linear_probe.py \
#   --checkpoint "${CHECKPOINT}" \
#   --data-path "${DATA_PATH}" \
#   # --model-size small \
#   # --embedding-dim 128 \
#   # --encoder-depth 8 \
#   # --decoder-depth 8 \
#   # --num-heads 8 \
#   --patch-size 1 \
#   --epochs 400 \
#   --batch-size 64 \
#   --test-batch-size 256 \
#   --lr 1e-3 \
#   --weight-decay 1e-4 \
#   --split-seed "${SEED}" \
#   # --train-ratio 0.8 \
#   --eval-every 10 \
#   --linear-probe \
#   --wandb \
#   --wandb-project "${WANDB_PROJECT}" \
#   --wandb-run-name "${WANDB_RUN_NAME}" \
#   --wandb-mode "${WANDB_MODE}" \
#   "${WANDB_ENTITY_ARGS[@]}"

# ---------- TESTING ONE EPOCH
# python hypersl_linear_probe.py \
#   --checkpoint "${CHECKPOINT}" \
#   --data-path "${DATA_PATH}" \
#   --wavelengths-path "${WAVELENGTHS_PATH}" \
#   --model-size small \
#   --embedding-dim 256 \
#   --encoder-depth 8 \
#   --decoder-depth 4 \
#   --num-heads 8 \
#   --head-type local_attention \
#   --freeze-encoder \
#   --patch-size 7 \
#   --spatial-depth 2 \
#   --spatial-heads 4 \
#   --spatial-mlp-ratio 4 \
#   --spatial-dropout 0.1 \
#   --epochs 200 \
#   --batch-size 4 \
#   --test-batch-size 16 \
#   --grad-accum-steps 8 \
#   --lr 3e-4 \
#   --weight-decay 1e-4 \
#   --split-seed "${SEED}" \
#   --eval-every 10

# LOCAL ATTENTION HEAD
# python hypersl_linear_probe.py \
#   --checkpoint "$CHECKPOINT" \
#   --data-path "$DATA_PATH" \
#   --wavelengths-path "$WAVELENGTHS_PATH" \
#   --model-size small \
#   --embedding-dim 256 \
#   --encoder-depth 8 \
#   --decoder-depth 4 \
#   --num-heads 8 \
#   --head-type local_attention \
#   --freeze-encoder \
#   --patch-size 7 \
#   --spatial-depth 1 \
#   --spatial-heads 4 \
#   --spatial-mlp-ratio 2 \
#   --spatial-dropout 0.1 \
#   --epochs 300 \
#   --eval-every 10 \
#   --batch-size 4 \
#   --test-batch-size 16 \
#   --grad-accum-steps 8 \
#   --lr 3e-4 \
#   --weight-decay 1e-4 \
#   --wandb \
#   --wandb-project "${WANDB_PROJECT}" \
#   --wandb-run-name "${WANDB_RUN_NAME}" \
#   --wandb-mode "${WANDB_MODE}" \
#   "${WANDB_ENTITY_ARGS[@]}"

# LINEAR CLASSIFIER
python hypersl_linear_probe.py \
  --checkpoint "$CHECKPOINT" \
  --data-path "$DATA_PATH" \
  --wavelengths-path "$WAVELENGTHS_PATH" \
  --model-size small \
  --embedding-dim 256 \
  --encoder-depth 8 \
  --decoder-depth 4 \
  --num-heads 8 \
  --head-type linear_center_mean \
  --freeze-encoder \
  --patch-size 1 \
  --epochs 200 \
  --eval-every 10 \
  --batch-size 4 \
  --test-batch-size 16 \
  --grad-accum-steps 8 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-mode "${WANDB_MODE}" \
  "${WANDB_ENTITY_ARGS[@]}"




# Original CNN 
# python hypersl_linear_probe.py \
#   --checkpoint "$CHECKPOINT" \
#   --data-path "$DATA_PATH" \
#   --wavelengths-path "$WAVELENGTHS_PATH" \
#   --model-size small \
#   --embedding-dim 256 \
#   --encoder-depth 8 \
#   --decoder-depth 4 \
#   --num-heads 8 \
#   --head-type cnn \
#   --freeze-encoder \
#   --patch-size 7 \
#   --epochs 200 \
#   --eval-every 10 \
#   --batch-size 4 \
#   --test-batch-size 16 \
#   --grad-accum-steps 8 \
#   --lr 3e-4 \
#   --weight-decay 1e-4 \
#   --wandb \
#   --wandb-project "${WANDB_PROJECT}" \
#   --wandb-run-name "${WANDB_RUN_NAME}" \
#   --wandb-mode "${WANDB_MODE}" \
#   "${WANDB_ENTITY_ARGS[@]}"