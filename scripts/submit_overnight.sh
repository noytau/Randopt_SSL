#!/bin/bash
# Submit the overnight Stage 1 + Stage 2 job to RunAI.
#
# Prerequisites:
#   1. Build and push the image (rebuild whenever code changes):
#        cd ~/PycharmProjects/Randopt
#        docker build -t noyhassid/randopt:v1 . && docker push noyhassid/randopt:v1
#
#   2. (Optional) Store WandB key on Lustre PVC for Stage 3+:
#        echo "YOUR_WANDB_KEY" > /storage/noy/.wandb_api_key && chmod 600 /storage/noy/.wandb_api_key
#        (Must be done from inside a container — Geoffrey cannot write to /storage/)
#
# NOTE: Geoffrey /mnt5/noy/ and container /storage/noy/ are SEPARATE filesystems.
#       Code changes on Geoffrey do NOT reach containers automatically.
#       Always rebuild the Docker image after code changes.
#
# Run from Geoffrey:
#   bash /mnt5/noy/Randopt/scripts/submit_overnight.sh
#
# Monitor:   runai logs randopt-overnight -f
# Results:   /storage/noy/Randopt/results/overnight.log  (on Lustre PVC)
#            Read via: runai logs randopt-overnight (after job ends)

set -e

JOB_NAME="randopt-overnight"
IMAGE="noyhassid/randopt:v1"
CODE_DIR="/opt/randopt"

# Delete previous run if it exists
runai delete job "$JOB_NAME" -p raja 2>/dev/null && sleep 3 || true

echo "Submitting: $JOB_NAME"
echo "  Image  : $IMAGE  (code baked in at $CODE_DIR)"
echo "  Results: /storage/noy/Randopt/results/ (Lustre PVC)"
echo "  Stages : 1 (smoke N=20, ~15 min) → 2 (dev N=100, ~45 min)"

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
