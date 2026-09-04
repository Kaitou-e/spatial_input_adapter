#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
wb_key=$(<wandb/wandb_key.txt)
export WANDB_API_KEY="$wb_key" 

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# DATASET="${DATASET:-IndianPine}"
# DATASET="${DATASET:-Pavia}"
# DATASET="${DATASET:-Houston}"
DATASET="${DATASET:-Chikusei}"
SEED="${SEED:-0}"
# SPLIT_DIR="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/splits}"
# DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/${DATASET}.mat}"
# DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/Pavia_spatial_blocks_seed0.mat}"
# DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/Houston2018_lesstrain_blocks40_seed0.mat}"
# DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/Houston2013_blocks30_seed0.mat}"
# DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/WHU_Hi_HongHu_patch7_seed0.mat}"
# DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/WashingtonDC_patch7_seed0_train10.mat}"
# DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/Botswana_block30_train80.mat}"
# DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/Salinas_block20_train80_seed0.mat}"
DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/Chikusei_block30_train80_seed0.mat}"
# DATA_PATH="${SPLIT_DIR:-/home/lxdcis/hypersl_training/data/IndianPines_patch5_double_split_8020_seed0.mat}"
CHECKPOINT="/home/lxdcis/hypersl_training/hypersl/modelarchive/10_base_mask95_checkpoint.pt"
# WAVELENGTHS_PATH="${WAVELENGTHS_PATH:-/home/lxdcis/hypersl_training/data/indian_pines_wavelengths_220.csv}"
# WAVELENGTHS_PATH="${WAVELENGTHS_PATH:-/home/lxdcis/hypersl_training/data/pavia_wavelengths_approx.csv}"
# WAVELENGTHS_PATH="/home/lxdcis/hypersl_training/data/houston2018_wavelengths_approx.csv"
# WAVELENGTHS_PATH="/home/lxdcis/hypersl_training/data/WHU_Hi_HongHu_wavelengths_nm.csv"
# WAVELENGTHS_PATH="/home/lxdcis/hypersl_training/data/wavelengths_191_bands.csv"
# WAVELENGTHS_PATH="/home/lxdcis/hypersl_training/data/Botswana/botswana_145_wavelengths.csv"
# WAVELENGTHS_PATH="/home/lxdcis/hypersl_training/data/Salinas/salinas_204_wavelengths.csv"
WAVELENGTHS_PATH="/home/lxdcis/hypersl_training/data/Chikusei/chikusei_128_wavelengths_exact.csv"
OUTPUT_PATH="outputs/chikusei_sia_seed0"

# echo "${WAVELENGTHS_PATH}" /home/lxdcis/hypersl_training/data

WANDB_PROJECT="${WANDB_PROJECT:-HyperSL-MixedNeighborsS-Chikusei}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-hypersl_${DATASET,,}_sia_seed${SEED}}"
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

# NEW SIA with proper eval/test
python hypersl_linear_probe.py \
  --dataset houston \
  --double-split \
  --checkpoint "$CHECKPOINT" \
  --data-path "$DATA_PATH" \
  --wavelengths-path "$WAVELENGTHS_PATH" \
  --model-size small \
  --embedding-dim 256 \
  --encoder-depth 8 \
  --decoder-depth 4 \
  --num-heads 8 \
  --head-type input_adapter_linear \
  --initial-mix 0.7 \
  --freeze-encoder \
  --patch-size 7 \
  --epochs 150 \
  --eval-every 10 \
  --selection-metric aa \
  --output-dir "$OUTPUT_PATH" \
  --batch-size 32 \
  --val-batch-size 128 \
  --test-batch-size 128 \
  --grad-accum-steps 1 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-mode "${WANDB_MODE}" \
  "${WANDB_ENTITY_ARGS[@]}"