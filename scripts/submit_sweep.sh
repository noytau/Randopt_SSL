#!/bin/bash
# Submit the full experiment sweep in stages.
#
# Stages:
#   1 = smoke tests   (Qwen2.5-0.5B × Countdown, N=20, no WandB — paper pipeline check)
#   2 = LLM dev run   (Qwen2.5-0.5B × Countdown, N=100, WandB enabled)
#   3 = full paper    (Qwen2.5 0.5B/1.5B/3B × Countdown, N=3000, K=50)
#   4 = GSM8K         (Qwen2.5-3B × GSM8K, N=3000, K=50)
#
# Storage:  on Geoffrey = /mnt5/noy/   inside container = /storage/noy/
# Image:    noyhassid/randopt:v1
# Project:  raja

STAGE=${1:-1}
CONTAINER_WORKDIR="/storage/noy/Randopt"
# WandB API key is stored once on the PVC (see setup instructions below).
# Containers read it at startup — no manual login needed per-job.
# To set up: echo "your-key" > /mnt5/noy/.wandb_api_key && chmod 600 /mnt5/noy/.wandb_api_key
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
        --working-dir "$CONTAINER_WORKDIR" \
        --node-pools faculty,raja \
        --command -- bash -c "
            export WANDB_API_KEY=\$(cat ${WANDB_KEY_FILE} 2>/dev/null || echo '')
            cd ${CONTAINER_WORKDIR}
            python3 -m scripts.run --config ${CONFIG} ${EXTRA}
        "
    echo "  Submitted: $NAME"
}

case $STAGE in
1)
    # Smoke test: verify the full pipeline works end-to-end on the paper's actual model.
    # N=20 is fast (~5 min on GPU). Methods: passatone, majority_vote, sft, randopt.
    echo "=== Stage 1: Smoke test — Qwen2.5-0.5B × Countdown, N=20 ==="
    submit randopt-smoke-countdown 1 \
        "--n_candidates 20 --top_k 3 --no_wandb \
         --output_dir /storage/noy/Randopt/results/smoke_countdown" \
        configs/qwen2_5_0_5b_countdown.yaml
    ;;

2)
    # Dev run: confirm RandOpt beats passatone/majority_vote at N=100 before the big run.
    echo "=== Stage 2: LLM dev run — Qwen2.5-0.5B × Countdown, N=100 ==="
    submit randopt-dev-countdown 1 \
        "--n_candidates 100 --top_k 10 \
         --output_dir /storage/noy/Randopt/results/dev_countdown" \
        configs/qwen2_5_0_5b_countdown.yaml
    ;;

3)
    # Full paper replication: N=3000, K=50, all three Qwen2.5 sizes.
    echo "=== Stage 3: Full paper — Qwen2.5 × Countdown, N=3000, K=50 ==="
    for MODEL in qwen2_5_0_5b qwen2_5_1_5b qwen2_5_3b; do
        submit "randopt-countdown-${MODEL}" 2 \
            "--output_dir /storage/noy/Randopt/results/${MODEL}_countdown_full" \
            "configs/${MODEL}_countdown.yaml"
    done
    ;;

4)
    # GSM8K: Qwen2.5-3B on math word problems, N=3000, K=50.
    echo "=== Stage 4: GSM8K — Qwen2.5-3B, N=3000, K=50 ==="
    submit "randopt-gsm8k-3b" 2 \
        "--output_dir /storage/noy/Randopt/results/qwen2_5_3b_gsm8k_full" \
        configs/qwen2_5_3b_gsm8k.yaml
    ;;

*)
    echo "Unknown stage: $STAGE. Use 1, 2, 3, or 4."
    exit 1
    ;;
esac
