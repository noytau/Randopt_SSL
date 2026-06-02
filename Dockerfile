# Randopt Benchmark — Docker image
#
# Provides Python + all packages for Qwen2.5 LLM experiments.
# Code lives on the shared filesystem at /storage/noy/Randopt (= Geoffrey /mnt5/noy/Randopt).
#
# Build & push:
#   docker build -t noyhassid/randopt:v1 .
#   docker push noyhassid/randopt:v1
#
# RunAI usage:
#   --image noyhassid/randopt:v1  --working-dir /storage/noy/Randopt
#   --existing-pvc claimname=storage,path=/storage
#   Results → /storage/noy/Randopt/results/   HF cache → /storage/noy/.cache/huggingface

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# HF models are cached on the Lustre PVC (persists across jobs)
ENV HF_HOME=/storage/noy/.cache/huggingface

# ── System packages ─────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl wget ca-certificates \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Upgrade pip ─────────────────────────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ── Core HuggingFace stack ───────────────────────────────────────────────────
# Pin transformers to a version that ships Qwen2.5 support (>=4.40)
RUN pip install --no-cache-dir \
    "transformers>=4.44,<5.0" \
    "tokenizers>=0.19" \
    "datasets>=2.20" \
    "accelerate>=0.33" \
    "huggingface-hub>=0.24"

# ── Optimisation / quantisation (optional but useful for 3B+ models) ────────
RUN pip install --no-cache-dir \
    "bitsandbytes>=0.43"

# ── Metrics & evaluation ─────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "scikit-learn>=1.3" \
    "scipy>=1.11" \
    "evaluate>=0.4"

# ── Experiment tracking ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "wandb>=0.17"

# ── Reporting & visualisation ────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "matplotlib>=3.7" \
    "seaborn>=0.12" \
    "pandas>=2.0" \
    "tabulate>=0.9" \
    "tqdm>=4.65"

# ── Misc utilities ───────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "pyyaml>=6.0" \
    "numpy>=1.24"

# ── No code baked in — code lives on Lustre PVC at /storage/noy/Randopt/ ────
# Update code on cluster: bash /mnt5/noy/Randopt/scripts/update_cluster_code.sh
# RunAI jobs use --working-dir /storage/noy/Randopt
WORKDIR /workspace

CMD ["/bin/bash"]
