#!/bin/bash
# Submit a single RandOpt experiment to the RunAI cluster.
#
# Usage:
#   bash scripts/submit_runai.sh configs/bert_rte.yaml
#   bash scripts/submit_runai.sh configs/qwen2_5_3b_countdown.yaml --gpu 2
#
# Storage:  on Geoffrey = /mnt5/noy/   inside container = /storage/noy/
# Image:    noyhassid/spectralfm-lean:v6
# Project:  raja

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
CONTAINER_WORKDIR="/storage/noy/Randopt"

echo "Submitting: $JOB_NAME"
echo "  Config : $CONFIG"
echo "  GPUs   : $GPU"
echo "  WandB  : ${NO_WANDB:-enabled}"

runai submit "$JOB_NAME" \
    --project raja \
    --image noyhassid/randopt:v1 \
    --gpu "$GPU" \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir "$CONTAINER_WORKDIR" \
    --node-pools faculty,raja \
    --command -- bash -c "
        [ -f /storage/noy/.wandb_api_key ] && export WANDB_API_KEY=\$(cat /storage/noy/.wandb_api_key)
        [ -f /storage/noy/miniconda3/envs/spectralfm/bin/python3 ] && export PATH=/storage/noy/miniconda3/envs/spectralfm/bin:/storage/noy/miniconda3/bin:\$PATH
        cd ${CONTAINER_WORKDIR}
        python3 -m scripts.run --config ${CONFIG} ${NO_WANDB}
    "

echo "Submitted: $JOB_NAME"
echo "Monitor: runai logs $JOB_NAME -f"
echo "Status:  runai describe job $JOB_NAME"
