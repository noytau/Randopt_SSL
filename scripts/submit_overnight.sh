#!/bin/bash
# Submit the overnight Stage 1 + Stage 2 job to RunAI.
#
# Prerequisites:
#   1. Build and push the image (ONCE, from your MacBook):
#        cd ~/PycharmProjects/Randopt
#        docker build -t noyhassid/randopt:v1 .
#        docker push noyhassid/randopt:v1
#
#   2. (Optional) Store WandB key on the Lustre PVC (needed for Stage 3+):
#        From an interactive container:
#        echo "YOUR_WANDB_KEY" > /storage/noy/.wandb_api_key && chmod 600 /storage/noy/.wandb_api_key
#
# Run from Geoffrey:
#   bash /mnt5/noy/Randopt/scripts/submit_overnight.sh
#
# Monitor:
#   runai logs randopt-overnight -f
#
# Results go to:  /storage/noy/Randopt/results/overnight.log  (on Lustre PVC)
# Note: Geoffrey cannot read Lustre directly. Read via container:
#   runai submit randopt-showlog ... --command -- cat /storage/noy/Randopt/results/overnight.log

set -e

JOB_NAME="randopt-overnight"
IMAGE="noyhassid/randopt:v1"
CODE_DIR="/opt/randopt"   # baked into the image

# Delete previous run if it exists
runai delete job "$JOB_NAME" -p raja 2>/dev/null && sleep 3 || true

echo "Submitting: $JOB_NAME"
echo "  Image : $IMAGE"
echo "  Code  : $CODE_DIR (baked into image)"
echo "  Results → /storage/noy/Randopt/results/ (Lustre PVC)"
echo "  Stages: 1 (smoke N=20, ~15 min) → 2 (dev N=100, ~45 min)"

runai submit "$JOB_NAME" \
    --project raja \
    --image "$IMAGE" \
    --gpu 1 \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir "$CODE_DIR" \
    --node-pools faculty,raja \
    --command -- bash "$CODE_DIR/scripts/run_overnight.sh"

echo ""
echo "Submitted. Watch with:"
echo "  runai logs $JOB_NAME -f"
echo ""
echo "Results will be at (accessible from inside a container):"
echo "  /storage/noy/Randopt/results/overnight.log"
echo ""
echo "Quick log viewer (run from Geoffrey):"
echo "  runai submit randopt-showlog -p raja --image noyhassid/randopt:v1 --existing-pvc claimname=storage,path=/storage --node-pools faculty,raja --command -- cat /storage/noy/Randopt/results/overnight.log"
