#!/bin/bash
# Submit a single RandOpt experiment to the RunAI cluster.
#
# Usage:
#   bash scripts/submit_runai.sh configs/bert_rte.yaml
#   bash scripts/submit_runai.sh configs/qwen2_5_3b_countdown.yaml --gpu 2
#
# Requirements:
#   - runai CLI configured (runai login)
#   - PVC "noy-storage" mounted at /mnt5/noy on cluster
#   - conda env "spectralfm" with all dependencies installed at /mnt5/noy

set -e

CONFIG=$1
if [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/submit_runai.sh <config.yaml> [--gpu N] [--no_wandb]"
    exit 1
fi

# Parse optional args
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

# Derive job name from config file (e.g. bert_rte.yaml -> randopt-bert-rte)
JOB_NAME="randopt-$(basename $CONFIG .yaml | tr '_' '-')"
WORKDIR="/mnt5/noy/Randopt"
CONFIG_PATH="${WORKDIR}/${CONFIG}"

echo "Submitting: $JOB_NAME"
echo "  Config : $CONFIG_PATH"
echo "  GPUs   : $GPU"
echo "  WandB  : ${NO_WANDB:-enabled}"

runai submit "$JOB_NAME" \
    --gpu "$GPU" \
    --pvc noy-storage:/mnt5/noy \
    --working-dir "$WORKDIR" \
    --image nvcr.io/nvidia/pytorch:23.10-py3 \
    --command -- bash -c "
        export PATH=/mnt5/noy/miniconda3/bin:\$PATH
        source activate spectralfm
        cd $WORKDIR
        python3 -m scripts.run --config $CONFIG $NO_WANDB
    "

echo "Submitted: $JOB_NAME"
echo "Monitor: runai logs $JOB_NAME -f"
echo "Status:  runai describe job $JOB_NAME"
