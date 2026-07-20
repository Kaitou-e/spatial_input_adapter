#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-Houston}"
SEED="${SEED:-0}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
ENCODER_DEPTH="${ENCODER_DEPTH:-8}"
DECODER_DEPTH="${DECODER_DEPTH:-8}"
NUM_HEADS="${NUM_HEADS:-8}"
MASK_PCT="${MASK_PCT:-75}"
EPOCH="${EPOCH:-300}"
DEVICE="${DEVICE:-auto}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/data/splits}"
DATA_PATH="${DATA_PATH:-${SPLIT_DIR}/${DATASET}_stratified_80_20_seed${SEED}.mat}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/hypersl/modelarchive/${DATASET,,}_mae_aug/embed${EMBEDDING_DIM}_enc${ENCODER_DEPTH}_dec${DECODER_DEPTH}_heads${NUM_HEADS}_mask${MASK_PCT}_epoch${EPOCH}.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/reports/hypersl/${DATASET,,}_embed${EMBEDDING_DIM}_epoch${EPOCH}_seed${SEED}}"

EMBED_SAMPLE_SIZE="${EMBED_SAMPLE_SIZE:-3000}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-512}"
RECON_SAMPLE_SIZE="${RECON_SAMPLE_SIZE:-6}"
RECON_BATCH_SIZE="${RECON_BATCH_SIZE:-256}"
SCENE_BATCH_SIZE="${SCENE_BATCH_SIZE:-256}"

WAVELENGTH_ARGS=()
if [[ -n "${WAVELENGTHS_PATH:-}" ]]; then
  WAVELENGTH_ARGS=(--wavelengths-path "${WAVELENGTHS_PATH}")
fi

TSNE_ARGS=()
if [[ "${RUN_TSNE:-1}" == "1" ]]; then
  TSNE_ARGS=(--run-tsne)
fi

SCENE_ARGS=()
if [[ "${SKIP_SCENE:-0}" == "1" ]]; then
  SCENE_ARGS=(--skip-scene)
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Missing visualization data file: ${DATA_PATH}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing HyperSL checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

python -m hypersl.viz \
  --dataset "${DATASET}" \
  --data-path "${DATA_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --embed-sample-size "${EMBED_SAMPLE_SIZE}" \
  --embed-batch-size "${EMBED_BATCH_SIZE}" \
  --recon-sample-size "${RECON_SAMPLE_SIZE}" \
  --recon-batch-size "${RECON_BATCH_SIZE}" \
  --scene-batch-size "${SCENE_BATCH_SIZE}" \
  "${WAVELENGTH_ARGS[@]}" \
  "${TSNE_ARGS[@]}" \
  "${SCENE_ARGS[@]}"

