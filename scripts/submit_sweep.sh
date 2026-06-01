#!/bin/bash
# Submit the full experiment sweep in stages.
#
# Stages:
#   1 = smoke tests (no WandB, tiny N)
#   2 = encoder sweep  (BERT GLUE tasks, full N=100)
#   3 = LLM dev run    (Qwen 0.5B, N=50, no WandB)
#   4 = full paper replication (Qwen 0.5B/1.5B/3B, N=3000)
#
# Storage:  on Geoffrey = /mnt5/noy/   inside container = /storage/noy/
# Image:    noyhassid/spectralfm-lean:v6
# Project:  raja

STAGE=${1:-1}
CONTAINER_WORKDIR="/storage/noy/Randopt"
CONDA="/storage/noy/miniconda3/bin"

submit() {
    local NAME=$1; shift
    local GPU=$1; shift
    local EXTRA=$1; shift
    local CONFIG=$1

    runai submit "$NAME" \
        --project raja \
        --image noyhassid/spectralfm-lean:v6 \
        --gpu "$GPU" \
        --existing-pvc claimname=storage,path=/storage \
        --working-dir "$CONTAINER_WORKDIR" \
        --node-pools faculty,raja \
        --command -- bash -c "
            export PATH=${CONDA}:\$PATH
            source activate spectralfm
            cd ${CONTAINER_WORKDIR}
            python3 -m scripts.run --config ${CONFIG} ${EXTRA}
        "
    echo "  Submitted: $NAME"
}

case $STAGE in
1)
    echo "=== Stage 1: Smoke tests (no WandB, tiny N) ==="
    submit randopt-smoke-rte 1 \
        "--n_candidates 20 --top_k 3 --no_wandb --output_dir /storage/noy/Randopt/results/smoke_rte" \
        configs/bert_rte.yaml
    ;;

2)
    echo "=== Stage 2: Encoder sweep (BERT, full N=100, WandB enabled) ==="
    for TASK in rte cola mrpc stsb sst2; do
        submit "randopt-bert-${TASK}" 1 \
            "--output_dir /storage/noy/Randopt/results/bert_${TASK}_sigma_set" \
            "configs/bert_${TASK}.yaml"
    done
    ;;

3)
    echo "=== Stage 3: LLM dev run (Qwen 0.5B, N=50, no WandB) ==="
    submit randopt-llm-dev 1 \
        "--n_candidates 50 --top_k 5 --no_wandb --output_dir /storage/noy/Randopt/results/llm_dev" \
        configs/qwen2_5_0_5b_countdown.yaml
    ;;

4)
    echo "=== Stage 4: Full paper replication (N=3000, K=50, WandB enabled) ==="
    for MODEL in qwen2_5_0_5b qwen2_5_1_5b qwen2_5_3b; do
        submit "randopt-countdown-${MODEL}" 2 \
            "--output_dir /storage/noy/Randopt/results/${MODEL}_countdown_full" \
            "configs/${MODEL}_countdown.yaml"
    done
    submit "randopt-gsm8k-3b" 2 \
        "--output_dir /storage/noy/Randopt/results/qwen2_5_3b_gsm8k_full" \
        configs/qwen2_5_3b_gsm8k.yaml
    ;;

*)
    echo "Unknown stage: $STAGE. Use 1, 2, 3, or 4."
    exit 1
    ;;
esac
