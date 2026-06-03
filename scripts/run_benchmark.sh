#!/bin/bash
# Run a full benchmark (all 4 methods) for one model × task inside a RunAI container.
#
# Required env var (set via runai submit --env):
#   MODEL_CONFIG  — relative path to yaml, e.g. configs/qwen2_5_0_5b_countdown.yaml
#   OUTPUT_DIR    — absolute output path, e.g. /storage/noy/Randopt/results/qwen2_5_0_5b_countdown
#
# Optional:
#   WANDB_PROJECT — defaults to randopt-benchmark

set -euo pipefail

WORKDIR=/storage/noy/Randopt
WANDB_PROJECT=${WANDB_PROJECT:-randopt-benchmark}

# ── WandB auth ─────────────────────────────────────────────────────────────
if [ -f /storage/noy/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)
    echo "WandB: key loaded"
else
    echo "WandB: key not found at /storage/noy/.wandb_api_key — will run without logging"
fi

# ── Info ────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  Randopt Benchmark"
echo "========================================================"
echo "  Host   : $(hostname)"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
echo "  VRAM   : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
echo "  Config : ${MODEL_CONFIG}"
echo "  Output : ${OUTPUT_DIR}"
echo "  Code   : $(git -C $WORKDIR log --oneline -1)"
echo "========================================================"

cd "$WORKDIR"

python3 -m scripts.run \
    --config "${MODEL_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_project "${WANDB_PROJECT}"

echo ""
echo "========================================================"
echo "  Done. Results: ${OUTPUT_DIR}"
echo "========================================================"
