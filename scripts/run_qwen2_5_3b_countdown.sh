#!/bin/bash
set -euo pipefail
WORKDIR=/storage/noy/Randopt

[ -f /storage/noy/.wandb_api_key ] && export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key) && echo "WandB: key loaded" || echo "WandB: key not found"

echo "========================================================"
echo "  qwen2.5.3b x Countdown  |  N=3000 K=50"
echo "  Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
echo "  Code: $(git -C $WORKDIR log --oneline -1)"
echo "========================================================"

cd "$WORKDIR"
python3 -m scripts.run \
    --config configs/qwen2_5_3b_countdown.yaml \
    --output_dir $WORKDIR/results/qwen2_5_3b_countdown \
    --wandb_project randopt-benchmark

echo "Done: $WORKDIR/results/qwen2_5_3b_countdown"
