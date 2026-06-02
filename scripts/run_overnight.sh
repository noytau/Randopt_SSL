#!/bin/bash
# Overnight sequential run inside the RunAI container (noyhassid/randopt:v1).
#
# Runs Stage 1 (smoke, N=20, ~15 min) then Stage 2 (dev, N=100, ~45 min).
# Stage 2 only starts if Stage 1 exits cleanly.
#
# Filesystem note:
#   Code      : /opt/randopt              (baked into image — no sync with Geoffrey)
#   Results   : /storage/noy/Randopt/results/  (Lustre PVC, persists across jobs)
#   HF cache  : /storage/noy/.cache/huggingface  (set via HF_HOME in image)
#   WandB key : /storage/noy/.wandb_api_key      (must be set up on PVC once)

set -euo pipefail

CODE_DIR=/opt/randopt
RESULTS_DIR=/storage/noy/Randopt/results
LOG=$RESULTS_DIR/overnight.log

mkdir -p "$RESULTS_DIR"

# ── WandB auth from PVC key file ──────────────────────────────────────────────
if [ -f /storage/noy/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)
fi

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOG"
}

log "========================================================"
log "  Randopt overnight run"
log "========================================================"
log "  Host   : $(hostname)"
log "  Python : $(which python3) [$(python3 --version 2>&1)]"
log "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
log "  VRAM   : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 'n/a')"
log "  Code   : $CODE_DIR"
log "  Results: $RESULTS_DIR"
log "========================================================"

cd "$CODE_DIR"

# ── Stage 1: Smoke test (N=20, no WandB — pipeline check only, ~15 min) ─────
log ""
log "Stage 1 | Smoke | Qwen2.5-0.5B × Countdown | N=20, K=3"
log "--------------------------------------------------------"
python3 -m scripts.run \
    --config configs/qwen2_5_0_5b_countdown.yaml \
    --n_candidates 20 --top_k 3 --no_wandb \
    --output_dir "$RESULTS_DIR/stage1_smoke" \
    2>&1 | tee -a "$LOG"

log "Stage 1 PASSED ✓"

# ── Stage 2: Dev run (N=100, no WandB — first accuracy signal, ~45 min) ──────
log ""
log "Stage 2 | Dev | Qwen2.5-0.5B × Countdown | N=100, K=10"
log "--------------------------------------------------------"
python3 -m scripts.run \
    --config configs/qwen2_5_0_5b_countdown.yaml \
    --n_candidates 100 --top_k 10 --no_wandb \
    --output_dir "$RESULTS_DIR/stage2_dev" \
    2>&1 | tee -a "$LOG"

log "Stage 2 PASSED ✓"

# ── Summary ──────────────────────────────────────────────────────────────────
log ""
log "========================================================"
log "  All done. Results in $RESULTS_DIR"
log "========================================================"
log ""
log "Quick view:"
python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, pathlib
results_dir = pathlib.Path('/storage/noy/Randopt/results')
for stage, subdir in [('Stage 1 (N=20)', 'stage1_smoke'), ('Stage 2 (N=100)', 'stage2_dev')]:
    p = results_dir / subdir / 'results.json'
    if not p.exists():
        print(f'{stage}: no results.json found')
        continue
    data = json.load(open(p))
    print(f'\n{stage}:')
    for r in data:
        print(f"  {r['method_name']:20s}  {r['primary_metric_name']}={r['primary_metric_value']:.4f}")
PYEOF
