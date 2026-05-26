# RandOpt Benchmark

Compare **RandOpt** (gradient-free random perturbation search) against **fine-tuning** and **linear probing** on three SSL foundation models across eight downstream tasks.

Research goal: show RandOpt matches fine-tuning performance while structurally preventing catastrophic forgetting — pretrained weights M are never modified.

Target venue: NeurIPS 2026 workshop → ICLR 2027 submission.

---

## The Three Methods

| Method | Encoder | Head | CF risk |
|--------|---------|------|---------|
| **Linear Probe** | Frozen | Trained (SGD) | None — encoder untouched |
| **Fine-Tuning** | Updated (SGD) | Trained (SGD) | High — weights drifted |
| **RandOpt** | Frozen (ensemble of perturbations) | Trained then fixed | **Zero** — M never modified |

### RandOpt algorithm

1. Freeze pretrained weights M
2. Warm-start a linear task head on frozen features
3. Sample N perturbations ε ~ N(0, σ²I), evaluate each M+ε on the task fitness metric (the actual metric, not a surrogate loss)
4. Select top-K seeds by fitness score
5. At inference: average features across K perturbed encoders (ensemble)

Key insight: because M is never modified and all ensemble members stay in N(M, σ²), catastrophic forgetting is structurally zero. RandOpt also has a direct advantage on **non-differentiable metrics** (MCC, episode accuracy, mIoU) because it optimizes the metric itself, not a proxy loss.

---

## Models × Tasks

### data2vec-audio Large (~315M) — `data2vec_audio`
- HuggingFace: `facebook/data2vec-audio-large`
- Hidden dim: 1024
- SSL: masked latent prediction on LibriSpeech

| Task | Config | Metric | Notes |
|------|--------|--------|-------|
| **ASR** | `configs/data2vec_asr.yaml` | WER ↓ | LibriSpeech 100h, CTC head |

### DINOv2 ViT-L/14 (~307M) — `dinov2`
- HuggingFace: `facebook/dinov2-large`
- Hidden dim: 1024
- SSL: self-distillation + masked patch prediction (iBOT)

| Task | Config | Metric | Notes |
|------|--------|--------|-------|
| **ImageNet cls** | `configs/dinov2_imagenet.yaml` | Acc ↑ | CIFAR-100 proxy (100 classes) |
| **Few-shot cls** | `configs/dinov2_fewshot.yaml` | 5-way-5-shot Acc ↑ | Non-differentiable → RandOpt advantage |

### BERT-Large (~340M) — `bert_large`
- HuggingFace: `bert-large-uncased`
- Hidden dim: 1024
- SSL: Masked Language Modeling + NSP

| Task | Config | Metric | Notes |
|------|--------|--------|-------|
| **RTE** | `configs/bert_rte.yaml` | Acc ↑ | Small dataset (2,490 pairs) |
| **MNLI** | `configs/bert_mnli.yaml` | Acc ↑ | Large dataset (10K subsample) |
| **CoLA** | `configs/bert_cola.yaml` | MCC ↑ | Non-differentiable → RandOpt advantage |

---

## Cluster Setup (Geoffrey, TAU)

**Environment:**
- Server: `ssh Geoffry` (132.66.52.64, user noy)
- GPUs: 8× RTX 2080 Ti (11GB each)
- Code dir: `/mnt/noy/Randopt/`
- Extra packages: `/mnt/noy/site-packages/` (jiwer, evaluate — installed there because /mnt5 is full)
- Python: `/mnt5/noy/miniconda3/envs/spectralfm/bin/python`
- HF cache: `/mnt/noy/.cache/hf`

**Run a single experiment on the cluster:**
```bash
ssh Geoffry
cd /mnt/noy/Randopt

PYTHONPATH=/mnt/noy/Randopt:/mnt/noy/site-packages \
HF_HOME=/mnt/noy/.cache/hf \
HF_DATASETS_CACHE=/mnt/noy/.cache/hf_datasets \
CUDA_VISIBLE_DEVICES=0 \
/mnt5/noy/miniconda3/envs/spectralfm/bin/python -m scripts.run \
  --config configs/bert_rte.yaml
```

**Or use the launcher script:**
```bash
ssh Geoffry
cd /mnt/noy/Randopt
CUDA_VISIBLE_DEVICES=0 ./run_server.sh --config configs/bert_rte.yaml
```

**Run all experiments in parallel (one GPU each):**
```bash
ssh Geoffry
cd /mnt/noy/Randopt

CUDA_VISIBLE_DEVICES=0 ./run_server.sh --config configs/bert_rte.yaml   > logs/bert_rte.log   2>&1 &
CUDA_VISIBLE_DEVICES=1 ./run_server.sh --config configs/bert_mnli.yaml  > logs/bert_mnli.log  2>&1 &
CUDA_VISIBLE_DEVICES=2 ./run_server.sh --config configs/bert_cola.yaml  > logs/bert_cola.log  2>&1 &
CUDA_VISIBLE_DEVICES=3 ./run_server.sh --config configs/dinov2_imagenet.yaml > logs/dinov2_cls.log  2>&1 &
CUDA_VISIBLE_DEVICES=4 ./run_server.sh --config configs/dinov2_fewshot.yaml  > logs/dinov2_fs.log   2>&1 &
CUDA_VISIBLE_DEVICES=5 ./run_server.sh --config configs/data2vec_asr.yaml    > logs/asr.log         2>&1 &

mkdir -p logs
wait && echo "All done"
```

**Sync code from local machine:**
```bash
rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' --exclude='results/' \
  /Users/noyhassid/PycharmProjects/Randopt/ \
  Geoffry:/mnt/noy/Randopt/
```

---

## Local Setup (Mac)

```bash
# Create env (Python 3.11 required — 3.13 incompatible with torch)
conda create -n randopt python=3.11 -y
conda activate randopt

# Install torch (CPU + MPS for Apple Silicon)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2"   # torch 2.2.2 requires numpy<2

# Install project
pip install -e .
```

Note: `transformers 5.x` requires `torch>=2.4`, which isn't available for Python 3.11 on macOS via pip.
If you see `[transformers] Disabling PyTorch`, it's a version mismatch — downgrade transformers: `pip install "transformers>=4.40,<5.0"`.

---

## Quick Start

```bash
# Validate pipeline end-to-end (tiny data, ~2 min each)
python -m scripts.run --config configs/dev_quick_bert.yaml
python -m scripts.run --config configs/dev_quick_dinov2.yaml
python -m scripts.run --config configs/dev_quick.yaml          # ASR

# Full experiments
python -m scripts.run --config configs/bert_rte.yaml
python -m scripts.run --config configs/bert_mnli.yaml
python -m scripts.run --config configs/bert_cola.yaml
python -m scripts.run --config configs/dinov2_imagenet.yaml
python -m scripts.run --config configs/dinov2_fewshot.yaml
python -m scripts.run --config configs/data2vec_asr.yaml

# Full sweep
python -m scripts.run_all            # all 6 experiments sequentially
python -m scripts.run_all --dev      # tiny quick sweep (3 combos)
python -m scripts.run_all --dry-run  # print commands only

# CLI overrides (work on any config)
python -m scripts.run --config configs/bert_rte.yaml --sigma 0.005 --n_candidates 200
python -m scripts.run --config configs/bert_rte.yaml --methods randopt

# Tests
python tests/test_core.py
python tests/test_reporting.py
```

---

## Results

After each run, `results/<name>/` contains:
- `results.json` — raw metrics for all methods
- `results_table.csv` — pivot table
- `bar_<model>_<task>.png` — per-task bar chart
- `overview_<model>.png` — all-tasks overview chart

---

## Project Structure

```
Randopt/
├── configs/
│   ├── dev_quick*.yaml          # Tiny debug configs (~2 min)
│   ├── bert_{rte,mnli,cola}.yaml
│   ├── dinov2_{imagenet,fewshot}.yaml
│   └── data2vec_asr.yaml
├── scripts/
│   ├── run.py                   # Main CLI — loads model+task+methods, reports results
│   ├── run_all.py               # Sweep runner (sequential)
│   └── run_server.sh            # Cluster launcher (sets PYTHONPATH, HF_HOME, CUDA)
├── src/
│   ├── interfaces.py            # ABCs: SSLModel, Task, AdaptationMethod, EvalResult
│   ├── registry.py              # @register_model/task/method decorators
│   ├── models/
│   │   ├── data2vec_audio.py    # facebook/data2vec-audio-large
│   │   ├── bert_large.py        # bert-large-uncased
│   │   └── dinov3.py            # facebook/dinov2-large (registry key: "dinov2")
│   ├── tasks/
│   │   ├── audio/asr.py         # LibriSpeech CTC, custom vocab, WER
│   │   ├── nlp/glue_tasks.py    # RTE, MNLI, CoLA (MCC)
│   │   └── vision/imagenet_cls.py  # CIFAR-100 cls + 5-way-5-shot episodes
│   ├── baselines/
│   │   ├── linear_probe.py      # Frozen encoder + trained head
│   │   └── finetune.py          # Full encoder + head fine-tuning
│   ├── randopt/
│   │   └── core.py              # PerturbationSampler, RandOpt, RandOptEnsemble
│   └── evaluation/
│       └── reporting.py         # Tables, bar charts, overview chart, JSON export
├── tests/
│   ├── test_core.py             # Registry, sampler, CTC vocab, EvalResult
│   └── test_reporting.py        # Reporting pipeline with mock data
└── run_server.sh                # Cluster launcher script
```

---

## Key Design Decisions

**`extract_features` has no `torch.no_grad()` inside.** Callers are responsible for wrapping in `no_grad` when appropriate:
- Linear probe train loop: `with torch.no_grad(): features = model.extract_features(batch)` ✓
- Fine-tuning train loop: no wrapper → gradients flow through encoder ✓
- All `evaluate()` methods: wrapped in `with torch.no_grad()` ✓
- RandOpt ensemble: wrapped in `with torch.no_grad()` ✓

**ASR task uses its own CTC vocab** (29-char hardcoded), not the model's tokenizer. `data2vec-audio-large` is a base SSL model with no built-in vocab file, so we use `Wav2Vec2FeatureExtractor` (not `Wav2Vec2Processor`) for audio preprocessing.

**Batch convention:** all tasks return `dict` batches. Audio: `input_values, attention_mask, labels`. Vision: `pixel_values, labels`. NLP: `input_ids, attention_mask, token_type_ids, labels`. Models pull the right keys internally in `extract_features`.

**Few-shot head is `nn.Identity()`** — features go directly to `evaluate()` for nearest-centroid episode classification. `train_linear_head` early-exits if the head has no parameters (avoids looping 50K images pointlessly).

---

## Adding Components

### New model
1. `src/models/my_model.py` → subclass `SSLModel`, implement `load/get_encoder/extract_features/hidden_dim/get_pretrained_state_dict`
2. `@register_model("my_model")`
3. `import src.models.my_model` in `scripts/run.py`
4. Add a config in `configs/`

### New task
1. `src/tasks/<domain>/my_task.py` → subclass `Task`, implement `load_data/build_head/get_loss_fn/evaluate/primary_metric`
2. `@register_task("my_task")`
3. `import src.tasks.<domain>.my_task` in `scripts/run.py`

### New method
1. `src/baselines/my_method.py` → subclass `AdaptationMethod`, implement `adapt` returning `(encoder, head)`
2. `@register_method("my_method")`
3. `import src.baselines.my_method` in `scripts/run.py`
