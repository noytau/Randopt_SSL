"""
Evaluation reporting: comparison tables and bar charts.
Produces both terminal output and saved figures/CSVs.
"""

import json
import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.interfaces import EvalResult

logger = logging.getLogger(__name__)

# ── Consistent style ────────────────────────────────────────────────────────

METHOD_COLORS = {
    "linear_probe": "#5B9BD5",
    "finetune":     "#ED7D31",
    "randopt":      "#70AD47",
}
METHOD_ORDER = ["linear_probe", "finetune", "randopt"]
METHOD_DISPLAY = {
    "linear_probe": "Linear Probe",
    "finetune":     "Fine-Tuning",
    "randopt":      "RandOpt",
}


def results_to_dataframe(results: List[EvalResult]) -> pd.DataFrame:
    """Convert list of EvalResult to a DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "model": r.model_name,
            "task": r.task_name,
            "method": r.method_name,
            "metric_name": r.primary_metric_name,
            "metric_value": r.primary_metric_value,
            "higher_is_better": r.higher_is_better,
            **r.extra,
        })
    return pd.DataFrame(rows)


def print_comparison_table(results: List[EvalResult], output_dir: Path = None):
    """Print a nicely formatted comparison table to terminal and optionally save CSV."""
    df = results_to_dataframe(results)

    # Pivot: rows = (model, task), columns = method
    pivot = df.pivot_table(
        index=["model", "task", "metric_name", "higher_is_better"],
        columns="method",
        values="metric_value",
    ).reset_index()

    # Reorder columns
    method_cols = [m for m in METHOD_ORDER if m in pivot.columns]
    display_cols = ["model", "task", "metric_name"] + method_cols
    pivot = pivot[display_cols]

    # Format
    print("\n" + "=" * 80)
    print("  COMPARISON TABLE")
    print("=" * 80)

    from tabulate import tabulate

    # Rename for display
    rename = {m: METHOD_DISPLAY.get(m, m) for m in method_cols}
    display_df = pivot.rename(columns=rename)
    display_df = display_df.rename(columns={
        "model": "Model",
        "task": "Task",
        "metric_name": "Metric",
    })

    print(tabulate(display_df, headers="keys", tablefmt="rounded_outline",
                   floatfmt=".4f", showindex=False))
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "results_table.csv"
        pivot.to_csv(csv_path, index=False)
        logger.info(f"  Saved table to {csv_path}")

    return pivot


def plot_bar_chart(
    results: List[EvalResult],
    output_dir: Path,
    group_by: str = "task",
):
    """
    Generate bar charts comparing methods.
    One chart per (model, task) combination.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(results)

    # Group by model × task
    for (model, task), group in df.groupby(["model", "task"]):
        metric_name = group["metric_name"].iloc[0]
        higher_is_better = group["higher_is_better"].iloc[0]

        fig, ax = plt.subplots(figsize=(8, 5))

        methods = []
        values = []
        colors = []

        for method in METHOD_ORDER:
            row = group[group["method"] == method]
            if len(row) > 0:
                methods.append(METHOD_DISPLAY.get(method, method))
                values.append(row["metric_value"].iloc[0])
                colors.append(METHOD_COLORS.get(method, "#999999"))

        bars = ax.bar(methods, values, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * max(values),
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )

        direction = "↑ higher is better" if higher_is_better else "↓ lower is better"
        ax.set_ylabel(f"{metric_name} ({direction})", fontsize=13)
        ax.set_title(f"{model} — {task}", fontsize=15, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(values) * 1.15 if values else 1)

        plt.tight_layout()
        filename = f"bar_{model}_{task}.png"
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved chart: {filename}")


def plot_overview_chart(results: List[EvalResult], output_dir: Path):
    """
    Generate a single overview chart with all model × task × method results.
    Grouped bar chart with tasks on x-axis, methods as bar colors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(results)

    models = df["model"].unique()

    for model in models:
        model_df = df[df["model"] == model]
        tasks = model_df["task"].unique()

        fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 3), 6))
        x = np.arange(len(tasks))
        n_methods = len(METHOD_ORDER)
        width = 0.8 / n_methods

        for i, method in enumerate(METHOD_ORDER):
            method_df = model_df[model_df["method"] == method]
            vals = []
            for task in tasks:
                task_row = method_df[method_df["task"] == task]
                vals.append(task_row["metric_value"].iloc[0] if len(task_row) > 0 else 0)

            offset = (i - n_methods / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, vals, width,
                label=METHOD_DISPLAY.get(method, method),
                color=METHOD_COLORS.get(method, "#999999"),
                edgecolor="white", linewidth=1,
            )
            # Value labels
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=12)
        ax.set_title(f"RandOpt Benchmark — {model}", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        filename = f"overview_{model}.png"
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved overview chart: {filename}")


def save_results_json(results: List[EvalResult], output_dir: Path):
    """Save raw results as JSON for later analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for r in results:
        data.append({
            "model": r.model_name,
            "task": r.task_name,
            "method": r.method_name,
            "metrics": r.metrics,
            "primary_metric_name": r.primary_metric_name,
            "primary_metric_value": r.primary_metric_value,
            "higher_is_better": r.higher_is_better,
            "extra": r.extra,
        })
    path = output_dir / "results.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"  Saved results JSON to {path}")
