#!/bin/bash
set -euo pipefail
WORKDIR=/storage/noy/Randopt
[ -f /storage/noy/.wandb_api_key ] && export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key) && echo "WandB: key loaded" || echo "WandB: key not found"
echo "qwen2.5.1.5b-v2 x Countdown (real HF data) | N=3000 K=50"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Code: $(git -C $WORKDIR log --oneline -1)"
cd "$WORKDIR"
python3 -m scripts.run \
    --config configs/qwen2_5_1_5b_countdown.yaml \
    --output_dir $WORKDIR/results/qwen2_5_1_5b_countdown_v2 \
    --wandb_project randopt-benchmark
echo "Done: $WORKDIR/results/qwen2_5_1_5b_countdown_v2"
