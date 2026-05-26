# RandOpt Benchmark

**Compare RandOpt (random perturbation search) against fine-tuning and linear probing on SSL foundation models.**

## Quick Start

```bash
# Install
pip install -e .

# Run data2vec + ASR (first experiment):
python -m scripts.run --config configs/data2vec_asr.yaml

# Dev/debug mode (tiny data, fast):
python -m scripts.run --config configs/dev_quick.yaml

# Override anything from CLI:
python -m scripts.run --config configs/data2vec_asr.yaml --sigma 0.005 --n_candidates 200

# Run only one method:
python -m scripts.run --config configs/data2vec_asr.yaml --methods randopt

# List available models/tasks/methods:
python -m scripts.run --list
```

## Adding a New Model

1. Create `src/models/my_model.py`
2. Subclass `SSLModel` from `src/interfaces.py`
3. Decorate with `@register_model("my_model")`
4. Import in `scripts/run.py`

## Adding a New Task

1. Create `src/tasks/<domain>/my_task.py`
2. Subclass `Task` from `src/interfaces.py`
3. Decorate with `@register_task("my_task")`
4. Implement: `load_data()`, `build_head()`, `get_loss_fn()`, `evaluate()`
5. Import in `scripts/run.py`

## Adding a New Method

1. Create `src/baselines/my_method.py` (or `src/randopt/variant.py`)
2. Subclass `AdaptationMethod` from `src/interfaces.py`
3. Decorate with `@register_method("my_method")`
4. Implement: `adapt()` returning `(encoder, head)`
5. Import in `scripts/run.py`

## Project Structure

```
randopt-benchmark/
├── configs/                    # YAML configs per experiment
├── scripts/
│   ├── run.py                  # Main CLI entry point
│   └── run_all.py              # Full sweep runner
├── src/
│   ├── interfaces.py           # ABCs: SSLModel, Task, AdaptationMethod
│   ├── registry.py             # @register_model/task/method decorators
│   ├── models/                 # Model adapters (data2vec, dinov3, bert...)
│   ├── tasks/                  # Task definitions (asr, sid, mnli, segmentation...)
│   ├── baselines/              # Baseline methods (linear_probe, finetune)
│   ├── randopt/                # RandOpt implementation
│   └── evaluation/             # Reporting: tables, charts, JSON export
├── results/                    # Output directory (auto-created)
└── tests/
```

## Output

After running, check `results/<model>_<task>/`:
- `results_table.csv` — comparison table
- `results.json` — raw metrics
- `bar_<model>_<task>.png` — per-task bar chart
- `overview_<model>.png` — all-tasks overview chart
