#!/usr/bin/env bash
# ─── RandOpt launcher for Geoffrey cluster ───────────────────────────────────
# Usage: ./run_server.sh [--config configs/dev_quick.yaml] [extra args...]
# Sets PYTHONPATH and CUDA env, then delegates to scripts/run.py
#
# Run in background with tmux:
#   tmux new -s randopt
#   ./run_server.sh --config configs/dev_quick_bert.yaml

set -e

PYTHON="/mnt5/noy/miniconda3/envs/spectralfm/bin/python"
EXTRA_PKGS="/mnt/noy/site-packages"
REPO_DIR="/mnt/noy/Randopt"

export PYTHONPATH="${REPO_DIR}:${EXTRA_PKGS}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_DATASETS_CACHE="/mnt/noy/.cache/hf_datasets"
export TRANSFORMERS_CACHE="/mnt/noy/.cache/transformers"
export HF_HOME="/mnt/noy/.cache/hf"

mkdir -p /mnt/noy/.cache/hf_datasets /mnt/noy/.cache/transformers /mnt/noy/.cache/hf
mkdir -p /mnt/noy/Randopt/results

cd "${REPO_DIR}"
echo "─── RandOpt (Geoffrey) ───────────────────────────────"
echo "  Python : ${PYTHON}"
echo "  GPU    : ${CUDA_VISIBLE_DEVICES}"
echo "  Args   : $@"
echo "──────────────────────────────────────────────────────"
exec "${PYTHON}" -m scripts.run "$@"
