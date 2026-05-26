# CLAUDE.md — Project Context for Claude Code

## What This Is

RandOpt Benchmark: comparing **RandOpt** (random weight perturbation search) against **fine-tuning** and **linear probing** on pretrained SSL foundation models. The goal is to show RandOpt provides competitive task adaptation while structurally preventing catastrophic forgetting — pretrained weights M are never modified.

This is research code targeting an **ICLR 2027 submission** with a **NeurIPS 2026 workshop** forcing function.

---

## How to Run

```bash
# Install dependencies
pip install -e .

# ── Quick debug runs (tiny data, ~2–3 min each) ──
python -m scripts.run --config configs/dev_quick.yaml          # data2vec + ASR
python -m scripts.run --config configs/dev_quick_bert.yaml     # BERT-Large + RTE
python -m scripts.run --config configs/dev_quick_dinov2.yaml   # DINOv2 + CIFAR-100

# ── Full experiments (single model × task) ──
python -m scripts.run --config configs/data2vec_asr.yaml
python -m scripts.run --config configs/bert_rte.yaml
python -m scripts.run --config configs/bert_mnli.yaml
python -m scripts.run --config configs/bert_cola.yaml
python -m scripts.run --config configs/dinov2_imagenet.yaml
python -m scripts.run --config configs/dinov2_fewshot.yaml

# ── Full sweep (all implemented model × task combos) ──
python -m scripts.run_all                  # all experiments
python -m scripts.run_all --dev            # quick validation sweep (3 combos, tiny data)
python -m scripts.run_all --model bert_large  # filter by model
python -m scripts.run_all --dry-run        # print commands without running

# ── CLI overrides (work on any config) ──
python -m scripts.run --config configs/bert_rte.yaml --sigma 0.005 --methods randopt
python -m scripts.run --config configs/data2vec_asr.yaml --n_candidates 200 --top_k 10

# ── Tests ──
python tests/test_core.py
python tests/test_reporting.py
```

---

## Architecture

**Registry pattern.** Models, tasks, and methods are registered via decorators and discovered automatically. To add anything new: write one file, decorate the class, add one import in `scripts/run.py`.

### Key Files

| File | Purpose |
|------|---------|
| `src/registry.py` | `@register_model`, `@register_task`, `@register_method` decorators + lookup |
| `src/interfaces.py` | ABCs: `SSLModel`, `Task`, `AdaptationMethod`, `EvalResult` dataclass |
| `src/randopt/core.py` | RandOpt algorithm: `PerturbationSampler`, `RandOpt`, `RandOptEnsemble` |
| `src/baselines/linear_probe.py` | Frozen encoder + linear head. Also provides `train_linear_head()` shared utility |
| `src/baselines/finetune.py` | Standard SGD fine-tuning of encoder + head |
| `src/evaluation/reporting.py` | Comparison tables, bar charts, JSON export |
| `scripts/run.py` | Main CLI entry point — ties model + task + methods together |
| `scripts/run_all.py` | Sweep runner for all model × task combos; supports `--dev` and `--model` filters |

### Model files

| File | Registry key | Status |
|------|-------------|--------|
| `src/models/data2vec_audio.py` | `data2vec_audio` | ✅ implemented |
| `src/models/bert_large.py` | `bert_large` | ✅ implemented |
| `src/models/dinov3.py` | `dinov2` | ✅ implemented (uses `facebook/dinov2-large`) |

### Task files

| File | Registry key(s) | Domain | Status |
|------|----------------|--------|--------|
| `src/tasks/audio/asr.py` | `asr` | Audio | ✅ implemented |
| `src/tasks/nlp/glue_tasks.py` | `rte`, `mnli`, `cola` | NLP | ✅ implemented |
| `src/tasks/vision/imagenet_cls.py` | `imagenet_cls`, `fewshot_cls` | Vision | ✅ implemented |

### Config files

| Config | Model | Task | Notes |
|--------|-------|------|-------|
| `configs/dev_quick.yaml` | data2vec_audio | asr | 100 train samples, 2 epochs |
| `configs/dev_quick_bert.yaml` | bert_large | rte | 80 train samples, 2 epochs |
| `configs/dev_quick_dinov2.yaml` | dinov2 | imagenet_cls | 100 train samples, CIFAR-100 |
| `configs/data2vec_asr.yaml` | data2vec_audio | asr | 5K samples, 3 FT epochs |
| `configs/bert_rte.yaml` | bert_large | rte | Full RTE (2,490 train) |
| `configs/bert_mnli.yaml` | bert_large | mnli | 10K subsample of 393K |
| `configs/bert_cola.yaml` | bert_large | cola | Full CoLA (8,551 train), 2 RandOpt rounds |
| `configs/dinov2_imagenet.yaml` | dinov2 | imagenet_cls | CIFAR-100 proxy, 5K samples |
| `configs/dinov2_fewshot.yaml` | dinov2 | fewshot_cls | 5-way 5-shot, 200 episodes |

---

## Adding New Components

### Adding a New Model

1. Create `src/models/my_model.py`
2. Subclass `SSLModel`:
   - `load(device)` — download and load pretrained weights
   - `get_encoder()` — return the nn.Module
   - `extract_features(batch)` — forward pass, return `[B, seq_len, hidden_dim]`
   - `hidden_dim()` — return int
   - `get_pretrained_state_dict()` — return a **copy** of original weights (immutable)
   - Optional: `get_processor()` for audio/vision, `get_tokenizer()` for NLP
3. Decorate: `@register_model("my_model")`
4. Add `import src.models.my_model` in `scripts/run.py`

### Adding a New Task

1. Create `src/tasks/<domain>/my_task.py`
2. Subclass `Task`:
   - `primary_metric()` → `(metric_name, higher_is_better)`
   - `load_data(split)` → `DataLoader` (splits: "train", "val", "test")
   - `build_head(input_dim, device)` → `nn.Module`
   - `get_loss_fn()` → callable `(logits, batch) → loss`
   - `evaluate(model, head, dataloader, device)` → `dict` of metrics
   - Optional: `set_processor(processor)` / `set_tokenizer(tokenizer)` — wired automatically by `run.py` if the model exposes a matching getter
3. Decorate: `@register_task("my_task")`
4. Add import in `scripts/run.py`

### Adding a New Method

1. Create `src/baselines/my_method.py`
2. Subclass `AdaptationMethod`:
   - `adapt(model, task, device, config)` → `(encoder_or_wrapper, head)`
3. Decorate: `@register_method("my_method")`
4. Add import in `scripts/run.py`

---

## Batch Convention

All tasks pass batches as **dicts**. Minimum keys:
- Audio tasks: `{"input_values": Tensor, "attention_mask": Tensor, "labels": Tensor, ...}`
- Vision tasks: `{"pixel_values": Tensor, "labels": Tensor, ...}`
- NLP tasks: `{"input_ids": Tensor, "attention_mask": Tensor, "token_type_ids": Tensor, "labels": Tensor}`

Models receive these dicts in `extract_features(batch)` and pull out the right keys internally.

**Feature shape convention:** all models return `[B, seq_len, hidden_dim]` where `seq_len ≥ 1`. Task heads that need a single vector (classification) pool `[:, 0, :]` (CLS token). This uniform shape makes heads interchangeable across models.

---

## Models × Tasks × Datasets

### Model 1: data2vec-audio Large (~315M) — `data2vec_audio`
- HuggingFace: `facebook/data2vec-audio-large`
- Hidden dim: 1024
- `get_processor()` → `Wav2Vec2Processor` (auto-wired to task)
- Tasks:
  - **ASR** ✅ implemented — LibriSpeech, WER ↓ (`configs/data2vec_asr.yaml`)
  - **SID** — VoxCeleb1 (1,251 speakers), Accuracy ↑
  - **ER** — IEMOCAP (4 emotions, ~5K utterances), Accuracy ↑
  - **KS** — Speech Commands v1 (12 classes, 65K clips), Accuracy ↑
  - **IC** — Fluent Speech Commands (31 intents, 30K utterances), Accuracy ↑

### Model 2: DINOv2 ViT-L/14 (~307M) — `dinov2`
- HuggingFace: `facebook/dinov2-large`
- Registry key: `dinov2` (file is `src/models/dinov3.py`)
- Hidden dim: 1024
- `get_processor()` → `AutoImageProcessor` (auto-wired to task)
- `output_cls_only=True` by default → returns `[B, 1, 1024]` (CLS token)
- Tasks:
  - **ImageNet/CIFAR-100 classification** ✅ implemented — CIFAR-100 proxy by default (`configs/dinov2_imagenet.yaml`)
  - **Few-shot classification** ✅ implemented — 5-way 5-shot on CIFAR-100, nearest-centroid (`configs/dinov2_fewshot.yaml`)
  - **Semantic segmentation** — ADE20K (150 classes), mIoU ↑ (non-differentiable!)
  - **Depth estimation** — NYUv2, RMSE ↓
  - **Object detection** — COCO (80 classes), mAP ↑

### Model 3: BERT-Large (~340M) — `bert_large`
- HuggingFace: `bert-large-uncased`
- Hidden dim: 1024
- `get_tokenizer()` → `BertTokenizerFast` (auto-wired to task via `set_tokenizer`)
- `output_cls_only=True` by default → returns `[B, 1, 1024]` (CLS token)
- Tasks:
  - **RTE** ✅ implemented — 2,490 pairs, binary, Accuracy ↑ (`configs/bert_rte.yaml`)
  - **MNLI** ✅ implemented — 393K pairs, 3-class, Accuracy ↑ (`configs/bert_mnli.yaml`)
  - **CoLA** ✅ implemented — 8,551 sentences, MCC ↑ non-differentiable (`configs/bert_cola.yaml`)
  - **STS-B** — 5,749 pairs, Spearman ρ ↑ (regression, non-differentiable)
  - **SQuAD v2.0** — 130K QA pairs, F1/EM ↑

---

## NLP Task Details (`src/tasks/nlp/glue_tasks.py`)

All GLUE tasks share `GLUETask` base class and `CLSClassificationHead`. The head pools `features[:, 0, :]` from `[B, 1, 1024]` → `[B, 1024]` → `Linear(1024, n_classes)`.

The tokenizer is set via `task.set_tokenizer(model.get_tokenizer())` in `run.py` — no manual wiring needed.

GLUE "validation" split is used for both `val` and `test` (GLUE test labels are hidden).

| Task | `_glue_config` | Classes | Metric | Differentiable? |
|------|---------------|---------|--------|----------------|
| `rte` | `rte` | 2 | Accuracy ↑ | Yes |
| `mnli` | `mnli` | 3 | Accuracy ↑ | Yes |
| `cola` | `cola` | 2 | MCC ↑ | **No** — RandOpt advantage |

CoLA requires `scikit-learn` for `matthews_corrcoef`.

---

## Vision Task Details (`src/tasks/vision/imagenet_cls.py`)

**`imagenet_cls`**: standard classification. Set `use_cifar100_fallback: true` in task_config (default) to use CIFAR-100 (100 classes) without needing HuggingFace ImageNet access. Set to `false` + provide HF token for real ImageNet-1K (1000 classes).

**`fewshot_cls`**: episode-based 5-way K-shot classification using nearest-centroid in DINOv2 feature space. The "head" is `nn.Identity()` — features go directly to `evaluate()`. The primary metric (episode accuracy) is **non-differentiable**, giving RandOpt a structural advantage. `n_episodes` controls evaluation reliability (200 recommended).

---

## RandOpt Algorithm Summary

1. Keep pretrained weights **M frozen** (never modified — this is why CF=0)
2. Train a linear head on frozen features (warm start via `train_linear_head()`)
3. Sample N perturbations: ε ~ N(0, σ²I), compute M' = M + ε
4. Evaluate each M' on task fitness (the **actual metric**, not a surrogate loss)
5. Select top-K by fitness score across all rounds
6. Ensemble: at inference, average features across K perturbed models

Key hyperparameters: `sigma` (perturbation scale — most critical), `n_candidates` (N), `top_k` (K), `n_rounds`.

The `PerturbationSampler` uses a seeded CPU RNG — perturbations are fully reproducible from `seed` alone, no tensors stored. The `RandOptEnsemble` stores only the top-K seeds, not the weight tensors.

---

## Important Notes

- **SIZE RISK**: BERT-Large (340M) and data2vec-audio (315M) are both under 500M. If RandOpt results are weak on these but strong on DINOv2 (307M), the issue is likely model capacity, NOT RandOpt itself. Fallback: upgrade BERT → DeBERTa-v2-XXLarge (1.5B).
- **Non-differentiable metrics** — CoLA MCC, few-shot episode accuracy, mIoU, Spearman ρ — are where RandOpt has a structural advantage because it optimizes the metric directly instead of a surrogate CE loss.
- The `train_linear_head()` function in `src/baselines/linear_probe.py` is shared between `LinearProbe` and `RandOpt`. RandOpt calls it to warm-start the task head before perturbation search.
- Always restore pretrained weights after each method runs. `scripts/run.py` does this automatically before and after every method.
- Configs are YAML in `configs/`. CLI args always override config file values.
- `run.py` auto-wires `get_processor()` → `set_processor()` and `get_tokenizer()` → `set_tokenizer()` if both model and task expose the matching methods.

---

## Testing

```bash
# Core logic (registry, perturbation sampler, CTC vocab, EvalResult):
python tests/test_core.py

# Reporting pipeline (table + charts with mock data):
python tests/test_reporting.py
```

When implementing new models/tasks, add test cases to `tests/` and verify with the matching `dev_quick_*.yaml` config before running full experiments.

---

## Dependencies

Core: `torch`, `transformers`, `datasets`, `tqdm`, `pyyaml`, `matplotlib`

Per-task extras:
- ASR: `jiwer` (WER computation)
- CoLA: `scikit-learn` (MCC computation)
- Vision: `Pillow` (image loading via HuggingFace datasets)

Install all with:
```bash
pip install -e .
```
