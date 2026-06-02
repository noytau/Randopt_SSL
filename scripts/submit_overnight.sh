#!/bin/bash
# Submit the overnight Stage 1 + Stage 2 job to RunAI.
#
# Prerequisites:
#   1. Code must be on Lustre PVC at /storage/noy/Randopt/
#      Push to GitHub then update cluster:
#        bash /mnt5/noy/Randopt/scripts/update_cluster_code.sh
#
#   2. Docker image (Python packages only — no code):
#        cd ~/PycharmProjects/Randopt
#        docker build -t noyhassid/randopt:v1 . && docker push noyhassid/randopt:v1
#      Only rebuild when dependencies change (not on code changes).
#
#   3. (Optional) Store WandB key on Lustre PVC:
#        echo "YOUR_WANDB_KEY" > /storage/noy/.wandb_api_key && chmod 600 /storage/noy/.wandb_api_key
#
# Run from Geoffrey:
#   bash /mnt5/noy/Randopt/scripts/submit_overnight.sh
#
# Monitor:   runai logs randopt-overnight -f
# Results:   /storage/noy/Randopt/results/overnight.log  (on Lustre PVC)

set -e

JOB_NAME="randopt-overnight"
IMAGE="noyhassid/randopt:v1"
WORKDIR="/storage/noy/Randopt"

# Delete previous run if it exists
runai delete job "$JOB_NAME" -p raja 2>/dev/null && sleep 3 || true

echo "Submitting: $JOB_NAME"
echo "  Image  : $IMAGE  (packages only)"
echo "  Code   : $WORKDIR  (Lustre PVC)"
echo "  Results: $WORKDIR/results/"
echo "  Stages : 1 (smoke N=20, ~15 min) → 2 (dev N=100, ~45 min)"

runai submit "$JOB_NAME" \
    --project raja \
    --image "$IMAGE" \
    --gpu 1 \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir "$WORKDIR" \
    --node-pools faculty,raja \
    --command -- bash scripts/run_overnight.sh

echo ""
echo "Submitted. Watch with:"
echo "  runai logs $JOB_NAME -f"
