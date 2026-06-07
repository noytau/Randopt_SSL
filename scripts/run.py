#!/usr/bin/env python3
"""
Main entry point for the RandOpt benchmark.

Usage:
  # Run all methods on data2vec + ASR (from config):
  python -m scripts.run --config configs/data2vec_asr.yaml

  # Override model/task/method from CLI:
  python -m scripts.run --model data2vec_audio --task asr --methods linear_probe finetune randopt

  # Override RandOpt params:
  python -m scripts.run --config configs/data2vec_asr.yaml --sigma 0.005 --n_candidates 200

  # Run only one method:
  python -m scripts.run --config configs/data2vec_asr.yaml --methods randopt

  # List available models/tasks/methods:
  python -m scripts.run --list
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

# ── Optional WandB ──────────────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ── Force-import all modules so @register decorators fire ──────────────────
# Each group is wrapped in try/except so missing optional deps (e.g.
# soundfile for audio, timm for vision) don't block other modalities.

# Audio (requires soundfile / torchaudio — optional)
try:
    import src.models.data2vec_audio               # noqa: F401
    import src.tasks.audio.asr                     # noqa: F401
    import src.tasks.audio.keyword_spotting        # noqa: F401
    import src.tasks.audio.speaker_id              # noqa: F401
    import src.tasks.audio.emotion_recognition     # noqa: F401
    import src.tasks.audio.intent_classification   # noqa: F401
except Exception as _e:
    logging.warning(f"Audio modules skipped (missing deps): {_e}")

# NLP (requires transformers — always available)
try:
    import src.models.bert_large                   # noqa: F401
    import src.tasks.nlp.glue_tasks                # noqa: F401
    import src.tasks.nlp.squad                     # noqa: F401
except Exception as _e:
    logging.warning(f"NLP modules skipped (missing deps): {_e}")

# Vision (requires timm / torchvision — optional)
try:
    import src.models.dinov3                       # noqa: F401
    import src.tasks.vision.imagenet_cls           # noqa: F401
    import src.tasks.vision.segmentation           # noqa: F401
    import src.tasks.vision.depth_estimation       # noqa: F401
except Exception as _e:
    logging.warning(f"Vision modules skipped (missing deps): {_e}")

# LLM (decoder models + generative tasks — always available)
try:
    import src.models.causal_lm                    # noqa: F401
    import src.tasks.llm.countdown                 # noqa: F401
    import src.tasks.llm.gsm8k                     # noqa: F401
except Exception as _e:
    logging.warning(f"LLM modules skipped (missing deps): {_e}")

# Methods (always available)
try:
    import src.baselines.linear_probe              # noqa: F401
    import src.baselines.finetune                  # noqa: F401
    import src.baselines.llm_baselines             # noqa: F401
    import src.baselines.sft_llm                   # noqa: F401
    import src.randopt.core                        # noqa: F401
except Exception as _e:
    logging.warning(f"Method modules skipped (missing deps): {_e}")

from src.interfaces import EvalResult
from src.registry import (
    MODEL_REGISTRY, TASK_REGISTRY, METHOD_REGISTRY,
    get_model, get_task, get_method, list_available,
)
from src.evaluation.reporting import (
    print_comparison_table,
    plot_bar_chart,
    plot_overview_chart,
    save_results_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # override any handlers set by early logging.warning() calls during imports
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RandOpt Benchmark: compare adaptation methods on SSL models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── Config file ──
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file.")

    # ── Overrides (take precedence over config file) ──
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model name. Available: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--task", type=str, default=None,
                        help=f"Task name. Available: {list(TASK_REGISTRY.keys())}")
    parser.add_argument("--methods", nargs="+", type=str, default=None,
                        help=f"Methods to compare. Available: {list(METHOD_REGISTRY.keys())}")

    # ── RandOpt params ──
    parser.add_argument("--sigma", type=float, default=None,
                        help="Single fixed σ for perturbation (backward compat).")
    parser.add_argument("--sigma_set", nargs="+", type=float, default=None,
                        help="Set Σ of σ values; each candidate draws σ ~ Uniform(Σ). "
                             "Matches Neural Thickets paper. Example: --sigma_set 0.001 0.003 0.005 0.01 0.03")
    parser.add_argument("--n_candidates", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--n_rounds", type=int, default=None)

    # ── Training params ──
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--head_epochs", type=int, default=None)
    parser.add_argument("--lr_encoder", type=float, default=None)
    parser.add_argument("--ft_epochs", type=int, default=None)

    # ── General ──
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--list", action="store_true",
                        help="List available models, tasks, and methods.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging (for local smoke tests).")
    parser.add_argument("--wandb_project", type=str, default="randopt-benchmark",
                        help="WandB project name.")

    return parser.parse_args()


def load_config(args) -> dict:
    """Merge YAML config with CLI overrides. CLI wins."""
    config = {}

    # Load YAML if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    # CLI overrides (only non-None values)
    for key in [
        "model", "task", "methods", "sigma", "sigma_set", "n_candidates", "top_k", "n_rounds",
        "head_lr", "head_epochs", "lr_encoder", "ft_epochs",
        "device", "seed", "output_dir", "no_wandb", "wandb_project",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    return config


def resolve_device(config: dict) -> torch.device:
    """Resolve device string to torch.device."""
    dev = config.get("device", "auto")
    if dev == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(dev)


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    if args.list:
        list_available()
        sys.exit(0)

    config = load_config(args)

    # ── Validate ──
    model_name = config.get("model")
    task_name = config.get("task")
    method_names = config.get("methods", ["linear_probe", "finetune", "randopt"])

    if not model_name or not task_name:
        logger.error("Must specify --model and --task (or use --config).")
        sys.exit(1)

    seed = config.get("seed", 42)
    set_seed(seed)
    device = resolve_device(config)
    output_dir = Path(config.get("output_dir", f"results/{model_name}_{task_name}"))

    logger.info("=" * 60)
    logger.info("  RandOpt Benchmark")
    logger.info("=" * 60)
    logger.info(f"  Model   : {model_name}")
    logger.info(f"  Task    : {task_name}")
    logger.info(f"  Methods : {method_names}")
    logger.info(f"  Device  : {device}")
    logger.info(f"  Seed    : {seed}")
    logger.info(f"  Output  : {output_dir}")
    logger.info("=" * 60)

    # ── WandB init ──
    use_wandb = WANDB_AVAILABLE and not config.get("no_wandb", False)
    if use_wandb:
        wandb_project = config.get("wandb_project", "randopt-benchmark")
        sigma_display = config.get("sigma_set") or config.get("sigma")
        try:
            wandb.init(
                project=wandb_project,
                name=f"{model_name}__{task_name}__seed{seed}",
                config={
                    "model": model_name,
                    "task": task_name,
                    "methods": method_names,
                    "sigma": sigma_display,
                    "n_candidates": config.get("n_candidates"),
                    "top_k": config.get("top_k"),
                    "n_rounds": config.get("n_rounds"),
                    "seed": seed,
                },
                reinit=True,
            )
            logger.info(f"  WandB  : {wandb_project} / {wandb.run.name}")
        except Exception as e:
            logger.warning(f"  WandB  : init failed ({e}) — continuing without logging.")
            use_wandb = False
    elif not WANDB_AVAILABLE:
        logger.info("  WandB  : not installed (pip install wandb to enable)")
    else:
        logger.info("  WandB  : disabled (--no_wandb)")

    # ── Load model ──
    logger.info(f"\n[1/4] Loading model: {model_name}")
    model_config = config.get("model_config", {})
    model = get_model(model_name, **model_config)
    model.load(device)

    # ── Load task ──
    logger.info(f"\n[2/4] Loading task: {task_name}")
    task_config = config.get("task_config", {})
    task = get_task(task_name, **task_config)

    # Wire model processor/tokenizer to task if needed
    if hasattr(task, "set_processor") and hasattr(model, "get_processor"):
        task.set_processor(model.get_processor())
    if hasattr(task, "set_tokenizer") and hasattr(model, "get_tokenizer"):
        task.set_tokenizer(model.get_tokenizer())

    # Propagate generation config to generative tasks
    if hasattr(task, "max_new_tokens") and config.get("max_new_tokens"):
        task.max_new_tokens = config["max_new_tokens"]

    # Pre-load data
    task.load_data("train")
    task.load_data("val")
    test_loader = task.load_data("test")

    # ── Run methods ──
    logger.info(f"\n[3/4] Running adaptation methods...")
    results = []

    for method_name in method_names:
        logger.info(f"\n{'─' * 40}")
        logger.info(f"  Method: {method_name}")
        logger.info(f"{'─' * 40}")

        # Reset model to pretrained weights before each method
        pretrained_state = model.get_pretrained_state_dict()
        model.get_encoder().load_state_dict(pretrained_state, strict=False)

        set_seed(seed)  # same seed for fair comparison
        t0 = time.time()

        method = get_method(method_name)
        encoder_or_ensemble, head = method.adapt(model, task, device, config)
        adapt_time = time.time() - t0

        # ── Evaluate on test set ──
        logger.info(f"  Evaluating on test set...")
        t0 = time.time()

        # Use whatever adapt() returned as the eval model.
        # If adapt() returned the original model object (linear_probe, finetune, sft, passatone),
        # eval_model IS model. If it returned a wrapper (RandOptEnsemble, MajorityVoteWrapper),
        # use that wrapper so its generate_ensemble() / generate() behaviour is exercised.
        eval_model = encoder_or_ensemble if encoder_or_ensemble is not model else model

        test_metrics = task.evaluate(eval_model, head, test_loader, device)
        eval_time = time.time() - t0

        metric_name, higher_is_better = task.primary_metric()
        result = EvalResult(
            model_name=model_name,
            task_name=task_name,
            method_name=method_name,
            metrics=test_metrics,
            primary_metric_name=metric_name,
            primary_metric_value=test_metrics[metric_name],
            higher_is_better=higher_is_better,
            extra={"adapt_time_sec": adapt_time, "eval_time_sec": eval_time},
        )
        results.append(result)
        logger.info(f"  ✓ {result.summary()}")
        logger.info(f"    Adapt: {adapt_time:.1f}s, Eval: {eval_time:.1f}s")

        # ── WandB log per-method result ──
        if use_wandb and wandb.run:
            log_dict = {f"{method_name}/{k}": v for k, v in test_metrics.items()}
            log_dict[f"{method_name}/adapt_time_sec"] = adapt_time
            log_dict[f"{method_name}/eval_time_sec"] = eval_time
            log_dict[f"{method_name}/total_time_sec"] = adapt_time + eval_time
            wandb.log(log_dict)

        # Restore pretrained weights after each method
        model.get_encoder().load_state_dict(pretrained_state, strict=False)

        # Free fragmented CUDA cache between methods to prevent OOM on large models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Report ──
    logger.info(f"\n[4/4] Generating reports...")
    print_comparison_table(results, output_dir)
    plot_bar_chart(results, output_dir)
    plot_overview_chart(results, output_dir)
    save_results_json(results, output_dir)

    logger.info(f"\n✓ All done! Results saved to {output_dir}/")

    # ── WandB finish ──
    if use_wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
