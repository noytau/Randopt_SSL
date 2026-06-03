#!/bin/bash
set -euo pipefail
WORKDIR=/storage/noy/Randopt
[ -f /storage/noy/.wandb_api_key ] && export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)
echo "qwen2.5.1.5b x Countdown | N=1000 K=50 | real HF data"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Code: $(git -C $WORKDIR log --oneline -1)"
cd "$WORKDIR"
python3 -m scripts.run \
    --config configs/qwen2_5_1_5b_countdown.yaml \
    --n_candidates 1000 --top_k 50 \
    --output_dir $WORKDIR/results/qwen2_5_1_5b_countdown_n1000 \
    --wandb_project randopt-benchmark
echo "Done: $WORKDIR/results/qwen2_5_1_5b_countdown_n1000"
