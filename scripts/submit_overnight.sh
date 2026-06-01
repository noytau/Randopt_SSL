#!/bin/bash
# Submit the overnight Stage 1 + Stage 2 job to RunAI.
#
# Prerequisites (one-time setup):
#   echo "YOUR_WANDB_KEY" > /mnt5/noy/.wandb_api_key && chmod 600 /mnt5/noy/.wandb_api_key
#
# Run from Geoffrey:
#   bash scripts/submit_overnight.sh
#
# Monitor:
#   runai logs randopt-overnight -f
#
# Results in the morning:
#   tail -50 /mnt5/noy/Randopt/results/overnight.log

set -e

JOB_NAME="randopt-overnight"
IMAGE="noyhassid/spectralfm-lean:v6"   # use existing image until randopt:v1 is pushed
WORKDIR="/storage/noy/Randopt"

# Delete previous run if it exists
runai delete job "$JOB_NAME" -p raja 2>/dev/null && sleep 3 || true

echo "Submitting: $JOB_NAME"
echo "  Image : $IMAGE"
echo "  Stages: 1 (smoke N=20, ~15 min) → 2 (dev N=100, ~45 min)"

runai submit "$JOB_NAME" \
    --project raja \
    --image "$IMAGE" \
    --gpu 1 \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir "$WORKDIR" \
    --node-pools faculty,raja \
    --command -- bash "$WORKDIR/scripts/run_overnight.sh"

echo ""
echo "Submitted. Watch with:"
echo "  runai logs $JOB_NAME -f"
echo ""
echo "In the morning:"
echo "  tail -50 /mnt5/noy/Randopt/results/overnight.log"
