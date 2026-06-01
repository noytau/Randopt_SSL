#!/bin/bash
# Submit the overnight Stage 1 + Stage 2 job to RunAI.
#
# Run this from Geoffrey AFTER the Docker image is pushed:
#   bash scripts/submit_overnight.sh
#
# Check progress:
#   runai logs randopt-overnight -f
#
# See results in the morning:
#   cat /mnt5/noy/Randopt/results/overnight.log
#   cat /mnt5/noy/Randopt/results/stage2_dev/results.json

set -e

JOB_NAME="randopt-overnight"
IMAGE="noyhassid/randopt:v1"
WORKDIR="/storage/noy/Randopt"

echo "Submitting: $JOB_NAME"
echo "  Image : $IMAGE"
echo "  Stages: 1 (smoke N=20) → 2 (dev N=100)"
echo "  Est.  : ~1 hour total"
echo ""

runai submit "$JOB_NAME" \
    --project raja \
    --image "$IMAGE" \
    --gpu 1 \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir "$WORKDIR" \
    --node-pools faculty,raja \
    --command -- bash scripts/run_overnight.sh

echo ""
echo "Submitted. Monitor with:"
echo "  runai logs $JOB_NAME -f"
echo ""
echo "In the morning, check results:"
echo "  cat /mnt5/noy/Randopt/results/overnight.log | tail -40"
