#!/bin/bash
# Submit the full experiment sweep in stages.
#
# Prerequisite: image noyhassid/randopt:v1 must be built and pushed.
#   cd ~/PycharmProjects/Randopt && docker build -t noyhassid/randopt:v1 . && docker push noyhassid/randopt:v1
#
# Stages:
#   1 = smoke tests   (Qwen2.5-0.5B × Countdown, N=20, no WandB)
#   2 = dev run       (Qwen2.5-0.5B × Countdown, N=100, WandB enabled)
#   3 = full paper    (Qwen2.5 0.5B/1.5B/3B × Countdown, N=3000, K=50)
#   4 = GSM8K         (Qwen2.5-3B × GSM8K, N=3000, K=50)
#
# Code: baked into image at /opt/randopt
# Results: /storage/noy/Randopt/results/ (Lustre PVC)
# Image:   noyhassid/randopt:v1
# Project: raja

STAGE=${1:-1}
CODE_DIR="/opt/randopt"
RESULTS_BASE="/storage/noy/Randopt/results"
WANDB_KEY_FILE="/storage/noy/.wandb_api_key"

submit() {
    local NAME=$1; shift
    local GPU=$1; shift
    local EXTRA=$1; shift
    local CONFIG=$1

    runai submit "$NAME" \
        --project raja \
        --image noyhassid/randopt:v1 \
        --gpu "$GPU" \
        --existing-pvc claimname=storage,path=/storage \
        --working-dir "$CODE_DIR" \
        --node-pools faculty,raja \
        --command -- bash -c "
            [ -f ${WANDB_KEY_FILE} ] && export WANDB_API_KEY=\$(cat ${WANDB_KEY_FILE})
            cd ${CODE_DIR}
            python3 -m scripts.run --config ${CONFIG} ${EXTRA}
        "
    echo "  Submitted: $NAME"
}

case $STAGE in
1)
    echo "=== Stage 1: Smoke test — Qwen2.5-0.5B × Countdown, N=20 ==="
    submit randopt-smoke-countdown 1 \
        "--n_candidates 20 --top_k 3 --no_wandb \
         --output_dir ${RESULTS_BASE}/smoke_countdown" \
        configs/qwen2_5_0_5b_countdown.yaml
    ;;

2)
    echo "=== Stage 2: Dev run — Qwen2.5-0.5B × Countdown, N=100 ==="
    submit randopt-dev-countdown 1 \
        "--n_candidates 100 --top_k 10 \
         --output_dir ${RESULTS_BASE}/dev_countdown" \
        configs/qwen2_5_0_5b_countdown.yaml
    ;;

3)
    echo "=== Stage 3: Full paper — Qwen2.5 × Countdown, N=3000, K=50 ==="
    for MODEL in qwen2_5_0_5b qwen2_5_1_5b qwen2_5_3b; do
        submit "randopt-countdown-${MODEL}" 2 \
            "--output_dir ${RESULTS_BASE}/${MODEL}_countdown_full" \
            "configs/${MODEL}_countdown.yaml"
    done
    ;;

4)
    echo "=== Stage 4: GSM8K — Qwen2.5-3B, N=3000, K=50 ==="
    submit "randopt-gsm8k-3b" 2 \
        "--output_dir ${RESULTS_BASE}/qwen2_5_3b_gsm8k_full" \
        configs/qwen2_5_3b_gsm8k.yaml
    ;;

*)
    echo "Unknown stage: $STAGE. Use 1, 2, 3, or 4."
    exit 1
    ;;
esac
