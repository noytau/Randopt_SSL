#!/bin/bash
# Submit a single RandOpt experiment to the RunAI cluster.
#
# Prerequisites:
#   Image noyhassid/randopt:v1 must be built and pushed:
#     cd ~/PycharmProjects/Randopt && docker build -t noyhassid/randopt:v1 . && docker push noyhassid/randopt:v1
#
# Usage:
#   bash scripts/submit_runai.sh configs/qwen2_5_0_5b_countdown.yaml
#   bash scripts/submit_runai.sh configs/qwen2_5_3b_gsm8k.yaml --gpu 2
#
# Code:    baked into image at /opt/randopt
# Results: /storage/noy/Randopt/results/  (Lustre PVC, persists)
# Image:   noyhassid/randopt:v1
# Project: raja

set -e

CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/submit_runai.sh <config.yaml> [--gpu N] [--no_wandb]"
    exit 1
fi

GPU=1
NO_WANDB=""
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2;;
        --no_wandb) NO_WANDB="--no_wandb"; shift;;
        *) shift;;
    esac
done

JOB_NAME="randopt-$(basename $CONFIG .yaml | tr '_' '-')"
CODE_DIR="/opt/randopt"

echo "Submitting: $JOB_NAME"
echo "  Config : $CONFIG"
echo "  GPUs   : $GPU"
echo "  WandB  : ${NO_WANDB:-enabled}"

runai submit "$JOB_NAME" \
    --project raja \
    --image noyhassid/randopt:v1 \
    --gpu "$GPU" \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir "$CODE_DIR" \
    --node-pools faculty,raja \
    --command -- bash -c "
        [ -f /storage/noy/.wandb_api_key ] && export WANDB_API_KEY=\$(cat /storage/noy/.wandb_api_key)
        cd ${CODE_DIR}
        python3 -m scripts.run --config ${CONFIG} ${NO_WANDB}
    "

echo "Submitted: $JOB_NAME"
echo "Monitor: runai logs $JOB_NAME -f"
echo "Status:  runai describe job $JOB_NAME"
