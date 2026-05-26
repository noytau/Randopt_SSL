"""Test reporting pipeline with mock data."""

import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.interfaces import EvalResult
from src.evaluation.reporting import (
    print_comparison_table,
    plot_bar_chart,
    plot_overview_chart,
    save_results_json,
)


def make_mock_results():
    """Create mock results simulating a full run."""
    return [
        # data2vec × ASR
        EvalResult("data2vec_audio", "asr", "linear_probe",
                    {"wer": 0.25}, "wer", 0.25, higher_is_better=False),
        EvalResult("data2vec_audio", "asr", "finetune",
                    {"wer": 0.08}, "wer", 0.08, higher_is_better=False),
        EvalResult("data2vec_audio", "asr", "randopt",
                    {"wer": 0.12}, "wer", 0.12, higher_is_better=False),

        # data2vec × SID (mock future task)
        EvalResult("data2vec_audio", "sid", "linear_probe",
                    {"accuracy": 0.72}, "accuracy", 0.72, higher_is_better=True),
        EvalResult("data2vec_audio", "sid", "finetune",
                    {"accuracy": 0.89}, "accuracy", 0.89, higher_is_better=True),
        EvalResult("data2vec_audio", "sid", "randopt",
                    {"accuracy": 0.85}, "accuracy", 0.85, higher_is_better=True),
    ]


def test_reporting():
    output_dir = Path("results/test_mock")

    results = make_mock_results()

    print("\n── Table ──")
    print_comparison_table(results, output_dir)

    print("\n── Charts ──")
    plot_bar_chart(results, output_dir)
    plot_overview_chart(results, output_dir)
    save_results_json(results, output_dir)

    # Verify files exist
    assert (output_dir / "results_table.csv").exists()
    assert (output_dir / "results.json").exists()
    assert (output_dir / "bar_data2vec_audio_asr.png").exists()
    assert (output_dir / "bar_data2vec_audio_sid.png").exists()
    assert (output_dir / "overview_data2vec_audio.png").exists()

    print("\n✓ All reporting tests passed!")
    print(f"  Check {output_dir}/ for generated files.")


if __name__ == "__main__":
    test_reporting()
