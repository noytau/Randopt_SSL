#!/bin/bash
# Submit a single full benchmark run (all methods) for one model × task.
#
# Usage (from Geoffrey):
#   bash /mnt5/noy/Randopt/scripts/submit_benchmark.sh <config> [gpu_count]
#
# Examples:
#   bash /mnt5/noy/Randopt/scripts/submit_benchmark.sh configs/qwen2_5_0_5b_countdown.yaml
#   bash /mnt5/noy/Randopt/scripts/submit_benchmark.sh configs/qwen2_5_1_5b_countdown.yaml
#   bash /mnt5/noy/Randopt/scripts/submit_benchmark.sh configs/qwen2_5_3b_countdown.yaml
#   bash /mnt5/noy/Randopt/scripts/submit_benchmark.sh configs/qwen2_5_3b_gsm8k.yaml
#
# WandB: reads API key from /storage/noy/.wandb_api_key (set once: echo KEY > /storage/noy/.wandb_api_key)
# Results: /storage/noy/Randopt/results/<config_basename>/

set -euo pipefail

CONFIG=${1:?"Usage: submit_benchmark.sh <config_path> [gpu_count]"}
GPU=${2:-1}
WORKDIR=/storage/noy/Randopt
BASENAME=$(basename "$CONFIG" .yaml)
JOB_NAME="randopt-$(echo "$BASENAME" | tr '_' '-')"
RESULTS="$WORKDIR/results/$BASENAME"

# Remove previous job with same name if exists
runai delete job "$JOB_NAME" -p raja 2>/dev/null && sleep 2 || true

echo "========================================================"
echo "  Submitting benchmark job"
echo "  Job     : $JOB_NAME"
echo "  Config  : $CONFIG"
echo "  GPU     : $GPU × RTX A5000"
echo "  Results : $RESULTS"
echo "========================================================"

runai submit "$JOB_NAME" \
    --project raja \
    --image noyhassid/randopt:v1 \
    --gpu "$GPU" \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir "$WORKDIR" \
    --node-pools faculty,raja \
    --command -- bash -c "
        set -euo pipefail
        echo 'Host: '\$(hostname)
        echo 'GPU:  '\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)

        # WandB auth from PVC key file
        if [ -f /storage/noy/.wandb_api_key ]; then
            export WANDB_API_KEY=\$(cat /storage/noy/.wandb_api_key)
            echo 'WandB: key loaded from PVC'
        else
            echo 'WandB: /storage/noy/.wandb_api_key not found — logging disabled'
        fi

        cd $WORKDIR
        echo 'Code HEAD: '\$(git log --oneline -1)

        python3 -m scripts.run \
            --config $CONFIG \
            --output_dir $RESULTS \
            --wandb_project randopt-benchmark
    "

echo ""
echo "Submitted. Commands:"
echo "  Monitor : runai job logs $JOB_NAME -f"
echo "  Results : $RESULTS/results.json  (on Lustre PVC)"
echo "  WandB   : https://wandb.ai/noyhassid/randopt-benchmark"
