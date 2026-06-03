#!/bin/bash
# Overnight sequential run inside the RunAI container (noyhassid/randopt:v1).
#
# Stage 1: smoke (N=20, ~15 min)  →  Stage 2: dev (N=100, ~45 min)
# Stage 2 only runs if Stage 1 exits cleanly.
#
# Paths (all on Lustre PVC):
#   Code    : /storage/noy/Randopt/          (git-managed, update via update_cluster_code.sh)
#   Results : /storage/noy/Randopt/results/
#   HF cache: /storage/noy/.cache/huggingface  (set via HF_HOME in image)

set -euo pipefail

WORKDIR=/storage/noy/Randopt
RESULTS=$WORKDIR/results
LOG=$RESULTS/overnight.log

mkdir -p "$RESULTS"

if [ -f /storage/noy/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)
fi

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOG"; }

log "========================================================"
log "  Randopt overnight run"
log "========================================================"
log "  Host   : $(hostname)"
log "  Python : $(python3 --version 2>&1)"
log "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none)"
log "  VRAM   : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
log "  Workdir: $WORKDIR"
log "========================================================"

cd "$WORKDIR"

log ""
log "Stage 1 | Smoke | Qwen2.5-0.5B × Countdown | N=20, K=3"
log "--------------------------------------------------------"
python3 -m scripts.run \
    --config configs/qwen2_5_0_5b_countdown.yaml \
    --n_candidates 20 --top_k 3 --no_wandb \
    --output_dir "$RESULTS/stage1_smoke" \
    2>&1 | tee -a "$LOG"
log "Stage 1 PASSED ✓"

log ""
log "Stage 2 | Dev | Qwen2.5-0.5B × Countdown | N=100, K=10"
log "--------------------------------------------------------"
python3 -m scripts.run \
    --config configs/qwen2_5_0_5b_countdown.yaml \
    --n_candidates 100 --top_k 10 --no_wandb \
    --output_dir "$RESULTS/stage2_dev" \
    2>&1 | tee -a "$LOG"
log "Stage 2 PASSED ✓"

log ""
log "========================================================"
log "  Done. Results: $RESULTS"
log "========================================================"
python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, pathlib
results = pathlib.Path('/storage/noy/Randopt/results')
for stage, sub in [('Stage 1 (N=20)', 'stage1_smoke'), ('Stage 2 (N=100)', 'stage2_dev')]:
    p = results / sub / 'results.json'
    if not p.exists():
        print(f'{stage}: no results.json'); continue
    data = json.load(open(p))
    print(f'\n{stage}:')
    for r in data:
        print(f"  {r['method_name']:20s}  {r['primary_metric_name']}={r['primary_metric_value']:.4f}")
PYEOF
