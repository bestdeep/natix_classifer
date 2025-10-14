#!/usr/bin/env bash
# run_test.sh
# Example:
# ./run_test.sh "checkpoints/convnext_small_best.pth checkpts/resnet18_best.pth" convnext_small /data/test tb_test 0

set -euo pipefail

CKPT_LIST="${1:-}"        # space-separated quoted string of checkpoint paths (required)
MODEL_NAME="${2:-}"       # the backbone builder name used when training (required)
TEST_DIRS="${3:-}"        # colon-separated test dirs (required)
TB_LOGDIR="${4:-tb_test}"
CUDA_DEVICES="${5:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
CPU_FLAG="${CPU_FLAG:-}"  # set to "--cpu" to force CPU
REAL_ONLY_FLAG="${REAL_ONLY_FLAG:-}" # set to "--real-only" to filter synthetic samples
IS_SYNTH_KEY="${IS_SYNTH_KEY:-is_synthetic}"

if [ -z "$CKPT_LIST" ] || [ -z "$MODEL_NAME" ] || [ -z "$TEST_DIRS" ]; then
  echo "Usage: $0 \"<ckpt1> <ckpt2> ...\" <model_name> <test_dirs(colon-separated)> [tb_logdir] [cuda_devices]"
  exit 2
fi

# convert ckpt list into arguments
read -ra CKPTS <<< "$CKPT_LIST"
CKPT_ARGS=""
for c in "${CKPTS[@]}"; do
  CKPT_ARGS+=" $c"
done

# test dirs -> multiple args
IFS=':' read -ra TDIRS <<< "$TEST_DIRS"
TEST_ARGS=""
for d in "${TDIRS[@]}"; do
  TEST_ARGS+=" --data-dirs $d"
done

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Running ensemble test with checkpoints: ${CKPTS[*]}"

python scripts/test.py \
  --model-checkpoints ${CKPT_ARGS} \
  --model-name "${MODEL_NAME}" \
  ${TEST_ARGS} \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --image-size "${IMAGE_SIZE}" \
  ${CPU_FLAG} \
  --tb-logdir "${TB_LOGDIR}" \
  ${REAL_ONLY_FLAG} \
  --is-synthetic-key "${IS_SYNTH_KEY}"

echo "Test complete. TB logs (if any) in ${TB_LOGDIR}"
