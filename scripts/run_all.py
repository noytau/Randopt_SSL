#!/usr/bin/env python3
"""
Run the full benchmark sweep: all models × all tasks × all methods.

Usage:
  # Run everything:
  python -m scripts.run_all

  # Run only one model (all its tasks):
  python -m scripts.run_all --model data2vec_audio

  # Dry run (print what would be executed):
  python -m scripts.run_all --dry-run
"""

import argparse
import subprocess
import sys

# ── Define the full sweep ────────────────────────────────────────────────────
# Each entry: (model, task, config_file_or_None)
# If config_file is None, uses CLI args only.

SWEEP = [
    # ── data2vec-audio (audio SSL) ──
    ("data2vec_audio", "asr",          "configs/data2vec_asr.yaml"),
    # ("data2vec_audio", "sid",        "configs/data2vec_sid.yaml"),     # TODO: VoxCeleb1
    # ("data2vec_audio", "er",         "configs/data2vec_er.yaml"),      # TODO: IEMOCAP
    # ("data2vec_audio", "ks",         "configs/data2vec_ks.yaml"),      # TODO: Speech Commands
    # ("data2vec_audio", "ic",         "configs/data2vec_ic.yaml"),      # TODO: Fluent Speech

    # ── DINOv2 ViT-L (vision SSL) ──
    ("dinov2",          "imagenet_cls", "configs/dinov2_imagenet.yaml"),
    ("dinov2",          "fewshot_cls",  "configs/dinov2_fewshot.yaml"),
    # ("dinov2",         "segmentation","configs/dinov2_seg.yaml"),      # TODO: ADE20K mIoU
    # ("dinov2",         "depth",       "configs/dinov2_depth.yaml"),    # TODO: NYUv2 RMSE
    # ("dinov2",         "detection",   "configs/dinov2_det.yaml"),      # TODO: COCO mAP

    # ── BERT-Large (NLP SSL) ──
    ("bert_large",      "rte",          "configs/bert_rte.yaml"),
    ("bert_large",      "mnli",         "configs/bert_mnli.yaml"),
    ("bert_large",      "cola",         "configs/bert_cola.yaml"),
    # ("bert_large",    "stsb",         "configs/bert_stsb.yaml"),       # TODO: Spearman ρ
    # ("bert_large",    "squad",        "configs/bert_squad.yaml"),      # TODO: F1/EM
]

# Quick dev sweep — tiny data for pipeline validation
SWEEP_DEV = [
    ("data2vec_audio", "asr",          "configs/dev_quick.yaml"),
    ("bert_large",     "rte",          "configs/dev_quick_bert.yaml"),
    ("dinov2",         "imagenet_cls", "configs/dev_quick_dinov2.yaml"),
]


def main():
    parser = argparse.ArgumentParser(description="Run full RandOpt benchmark sweep.")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to a single model.")
    parser.add_argument("--dev", action="store_true",
                        help="Use tiny dev sweep (SWEEP_DEV) for quick pipeline validation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    args = parser.parse_args()

    entries = SWEEP_DEV if args.dev else SWEEP
    if args.model:
        entries = [(m, t, c) for m, t, c in entries if m == args.model]

    if not entries:
        print(f"No entries found for model='{args.model}'")
        sys.exit(1)

    print(f"Running {len(entries)} experiments:")
    for model, task, config in entries:
        print(f"  • {model} × {task}")
    print()

    for i, (model, task, config) in enumerate(entries, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(entries)}] {model} × {task}")
        print(f"{'='*60}\n")

        cmd = [sys.executable, "-m", "scripts.run"]
        if config:
            cmd += ["--config", config]
        else:
            cmd += ["--model", model, "--task", task]

        if args.dry_run:
            print(f"  DRY RUN: {' '.join(cmd)}")
        else:
            result = subprocess.run(cmd, cwd=".")
            if result.returncode != 0:
                print(f"  ⚠ FAILED with exit code {result.returncode}")
            else:
                print(f"  ✓ Done.")


if __name__ == "__main__":
    main()
