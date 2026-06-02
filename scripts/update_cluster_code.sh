#!/bin/bash
# Update the Randopt code on Lustre PVC by running a git pull inside a container.
#
# Usage (from Geoffrey or local):
#   bash /mnt5/noy/Randopt/scripts/update_cluster_code.sh
#
# What it does:
#   1. Submits a tiny CPU RunAI job (randopt-update)
#   2. The job runs: cd /storage/noy/Randopt && git pull
#   3. Uses GitHub token stored at /storage/noy/.github_token for auth
#
# One-time setup (run from inside any container with /storage mounted):
#   git -C /storage/noy/Randopt remote set-url origin \
#     https://$(cat /storage/noy/.github_token)@github.com/noytau/Randopt_SSL.git
#
# After this script completes, submit your experiment:
#   bash /mnt5/noy/Randopt/scripts/submit_overnight.sh

set -e

JOB_NAME="randopt-update"
WORKDIR="/storage/noy/Randopt"

runai delete job "$JOB_NAME" -p raja 2>/dev/null || true

echo "Submitting git pull job: $JOB_NAME"

runai submit "$JOB_NAME" \
    --project raja \
    --image noyhassid/randopt:v1 \
    --gpu 0 \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir "$WORKDIR" \
    --node-pools faculty,raja \
    --command -- bash -c "
        echo 'Pulling latest code into /storage/noy/Randopt ...'
        git -C /storage/noy/Randopt pull --ff-only
        echo 'Done. Current HEAD:'
        git -C /storage/noy/Randopt log --oneline -3
    "

echo ""
echo "Watch with:  runai logs $JOB_NAME -f"
echo "Then submit: bash /mnt5/noy/Randopt/scripts/submit_overnight.sh"
