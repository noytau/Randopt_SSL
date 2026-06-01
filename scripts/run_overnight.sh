#!/bin/bash
# Overnight sequential run inside the RunAI container.
#
# Runs Stage 1 (smoke, N=20, ~15 min) then Stage 2 (dev, N=100, ~45 min).
# Stage 2 only starts if Stage 1 exits cleanly.
# Everything is logged to $WORKDIR/results/overnight.log.
#
# Works with both noyhassid/randopt:v1 (system Python)
# and noyhassid/spectralfm-lean:v6 (conda Python) — auto-detected.

set -euo pipefail

# ── Resolve Python ────────────────────────────────────────────────────────────
# Priority: (1) shared conda on Lustre PVC  (2) system python
CONDA_PYTHON=/storage/miniconda3/envs/spectralfm/bin/python3
if [ -f "$CONDA_PYTHON" ]; then
    export PATH=/storage/miniconda3/envs/spectralfm/bin:/storage/miniconda3/bin:$PATH
    echo "[setup] Using conda python: $CONDA_PYTHON"
else
    echo "[setup] Using system python: $(which python3 2>/dev/null || echo 'NOT FOUND')"
fi

# ── Ensure required packages are installed ────────────────────────────────────
python3 -m pip install --upgrade --quiet 'transformers>=4.44' datasets accelerate wandb scikit-learn 2>&1 | tail -3

# ── WandB auth from PVC key file ──────────────────────────────────────────────
WANDB_KEY_FILE=/storage/noy/.wandb_api_key
if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
fi

WORKDIR=/storage/noy/Randopt
LOG=$WORKDIR/results/overnight.log
mkdir -p "$WORKDIR/results"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $*" | tee -a "$LOG"
}

log "========================================================"
log "  Randopt overnight run"
log "========================================================"
log "  Host   : $(hostname)"
log "  Python : $(which python3)"
log "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'none')"
log "  VRAM   : $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo 'n/a')"
log "========================================================"

cd "$WORKDIR"

# ── Stage 1: Smoke test (N=20, no WandB — pipeline check only, ~15 min) ─────
log ""
log "Stage 1 | Smoke | Qwen2.5-0.5B × Countdown | N=20, K=3"
log "--------------------------------------------------------"
python3 -m scripts.run \
    --config configs/qwen2_5_0_5b_countdown.yaml \
    --n_candidates 20 --top_k 3 --no_wandb \
    --output_dir "$WORKDIR/results/stage1_smoke" \
    2>&1 | tee -a "$LOG"

log "Stage 1 PASSED ✓"

# ── Stage 2: Dev run (N=100, no WandB — first accuracy signal, ~45 min) ──────
log ""
log "Stage 2 | Dev | Qwen2.5-0.5B × Countdown | N=100, K=10"
log "--------------------------------------------------------"
python3 -m scripts.run \
    --config configs/qwen2_5_0_5b_countdown.yaml \
    --n_candidates 100 --top_k 10 --no_wandb \
    --output_dir "$WORKDIR/results/stage2_dev" \
    2>&1 | tee -a "$LOG"

log "Stage 2 PASSED ✓"

# ── Summary ──────────────────────────────────────────────────────────────────
log ""
log "========================================================"
log "  All done. Results:"
log "    Stage 1 : $WORKDIR/results/stage1_smoke/results.json"
log "    Stage 2 : $WORKDIR/results/stage2_dev/results.json"
log "========================================================"
log ""
log "Quick view:"
python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, pathlib
for stage, path in [('Stage 1 (N=20)', 'results/stage1_smoke'), ('Stage 2 (N=100)', 'results/stage2_dev')]:
    p = pathlib.Path(path) / 'results.json'
    if not p.exists():
        print(f'{stage}: no results.json found')
        continue
    data = json.load(open(p))
    print(f'\n{stage}:')
    for r in data:
        print(f"  {r['method_name']:20s}  {r['primary_metric_name']}={r['primary_metric_value']:.4f}")
PYEOF
