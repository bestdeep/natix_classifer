#!/usr/bin/env bash
# run_train.sh
# Example: ./run_train.sh convnext_small,resnet18 /data/train /data/val checkpts tb_logs 0
# MODELS comma-separated, TRAIN_DIRS and VAL_DIRS can be colon-separated lists (will be split into multiple args)

set -euo pipefail

# -----------------------
# Config (edit or pass env overrides)
# -----------------------
MODELS="${1:-}"            # comma-separated backbone names (required)
TRAIN_DIRS="${2:-}"        # colon-separated paths (required)
VAL_SPLIT="${3:-0.2}"      # fraction of data used for validation (0–1)
OUTPUT_DIR="${4:-checkpoints}"
TB_LOGDIR="${5:-tb_logs}"
CUDA_DEVICES="${6:-0}"     # e.g. "0" or "0,1"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
PRETRAINED_FLAG="${PRETRAINED_FLAG:-}"   # set to "--pretrained" to enable
AUGMENT_FLAG="${AUGMENT_FLAG:-}"         # set to "--augment" to enable
MIXUP_ALPHA="${MIXUP_ALPHA:-0.2}"
MIXUP_PROB="${MIXUP_PROB:-0.5}"
CUTMIX_ALPHA="${CUTMIX_ALPHA:-1.0}"
CUTMIX_PROB="${CUTMIX_PROB:-0.5}"
REAL_ONLY_VAL_FLAG="${REAL_ONLY_VAL_FLAG:-}"  # set to "--real-only-val" to enable
IS_SYNTHETIC_KEY="${IS_SYNTHETIC_KEY:-is_synthetic}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
LOG_IMAGE_COUNT="${LOG_IMAGE_COUNT:-16}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"

if [ -z "$MODELS" ] || [ -z "$TRAIN_DIRS" ] || [ -z "$VAL_DIRS" ]; then
  echo "Usage: $0 <models(comma-separated)> <train_dirs(colon-separated)> <val_dirs(colon-separated)> [output_dir] [tb_logdir] [cuda_devices]"
  echo "Example: $0 convnext_small,resnet18 /data/train1:/data/train2 /data/val checkpts tb_logs 0,1"
  exit 2
fi

# Build args
IFS=',' read -ra MODEL_ARR <<< "$MODELS"
MODELS_ARGS=""
for m in "${MODEL_ARR[@]}"; do
  MODELS_ARGS+=" $m"
done

# build train/val dirs as multiple args
IFS=':' read -ra TDIRS <<< "$TRAIN_DIRS"
TRAIN_ARGS=""
for d in "${TDIRS[@]}"; do
  TRAIN_ARGS+=" --train-dirs $d"
done

IFS=':' read -ra VDIRS <<< "$VAL_DIRS"
VAL_ARGS=""
for d in "${VDIRS[@]}"; do
  VAL_ARGS+=" --val-dirs $d"
done

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Starting training for models:${MODEL_ARR[*]}"
echo "Output dir: ${OUTPUT_DIR}"
echo "TensorBoard dir: ${TB_LOGDIR}"

python scripts/train.py \
  --models ${MODELS_ARGS} \
  ${TRAIN_ARGS} \
  --val-split "${VAL_SPLIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --image-size "${IMAGE_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  ${PRETRAINED_FLAG} \
  ${AUGMENT_FLAG} \
  --mixup-alpha "${MIXUP_ALPHA}" \
  --mixup-prob "${MIXUP_PROB}" \
  --cutmix-alpha "${CUTMIX_ALPHA}" \
  --cutmix-prob "${CUTMIX_PROB}" \
  --tb-logdir "${TB_LOGDIR}" \
  --log-every-steps "${LOG_EVERY_STEPS}" \
  --log-image-count "${LOG_IMAGE_COUNT}" \
  --warmup-epochs "${WARMUP_EPOCHS}" \
  ${REAL_ONLY_VAL_FLAG} \
  --is-synthetic-key "${IS_SYNTHETIC_KEY}"

echo "Training finished. Checkpoints in ${OUTPUT_DIR}. TensorBoard logs in ${TB_LOGDIR}"
