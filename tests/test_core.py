"""Basic tests for the registry and interface plumbing."""

import sys
sys.path.insert(0, ".")

import torch
from src.registry import MODEL_REGISTRY, TASK_REGISTRY, METHOD_REGISTRY


def test_registrations():
    """All expected models/tasks/methods are registered."""
    # Force imports
    import src.models.data2vec_audio
    import src.tasks.audio.asr
    import src.baselines.linear_probe
    import src.baselines.finetune
    import src.randopt.core

    assert "data2vec_audio" in MODEL_REGISTRY
    assert "asr" in TASK_REGISTRY
    assert "linear_probe" in METHOD_REGISTRY
    assert "finetune" in METHOD_REGISTRY
    assert "randopt" in METHOD_REGISTRY
    print("✓ test_registrations passed")


def test_perturbation_sampler():
    """PerturbationSampler produces deterministic, correctly-shaped noise."""
    from src.randopt.core import PerturbationSampler

    ref_state = {
        "layer.weight": torch.randn(64, 32),
        "layer.bias": torch.randn(64),
    }
    sampler = PerturbationSampler(sigma=0.01, device=torch.device("cpu"))

    eps1 = sampler.sample(ref_state, seed=42)
    eps2 = sampler.sample(ref_state, seed=42)
    eps3 = sampler.sample(ref_state, seed=99)

    # Same seed → same noise
    assert torch.allclose(eps1["layer.weight"], eps2["layer.weight"])
    # Different seed → different noise
    assert not torch.allclose(eps1["layer.weight"], eps3["layer.weight"])
    # Correct shapes
    assert eps1["layer.weight"].shape == (64, 32)
    assert eps1["layer.bias"].shape == (64,)
    # Correct scale (should be ~sigma)
    assert eps1["layer.weight"].std().item() < 0.05  # 5x sigma is generous
    print("✓ test_perturbation_sampler passed")


def test_perturbation_apply():
    """Perturbed weights = M + epsilon, and M is unchanged."""
    from src.randopt.core import PerturbationSampler

    ref_state = {"w": torch.ones(10, 10)}
    sampler = PerturbationSampler(sigma=0.01, device=torch.device("cpu"))
    eps = sampler.sample(ref_state, seed=42)
    perturbed = sampler.apply(ref_state, eps)

    # M + eps
    assert torch.allclose(perturbed["w"], ref_state["w"] + eps["w"])
    # Original unchanged
    assert torch.allclose(ref_state["w"], torch.ones(10, 10))
    print("✓ test_perturbation_apply passed")


def test_ctc_vocab():
    """CTC encode → decode roundtrip."""
    from src.tasks.audio.asr import text_to_ids, ids_to_text, BLANK_ID

    text = "hello world"
    ids = text_to_ids(text)
    assert len(ids) == len(text)

    # Simulate CTC output (repeat chars, insert blanks)
    ctc_output = []
    for idx in ids:
        ctc_output.extend([idx, idx, BLANK_ID])  # repeat + blank
    decoded = ids_to_text(ctc_output)
    assert decoded == text, f"Expected '{text}', got '{decoded}'"
    print("✓ test_ctc_vocab passed")


def test_eval_result():
    """EvalResult serialization."""
    from src.interfaces import EvalResult

    r = EvalResult(
        model_name="test_model",
        task_name="test_task",
        method_name="randopt",
        metrics={"wer": 0.05, "cer": 0.02},
        primary_metric_name="wer",
        primary_metric_value=0.05,
        higher_is_better=False,
    )
    assert "0.0500" in r.summary()
    assert "↓" in r.summary()
    print("✓ test_eval_result passed")


if __name__ == "__main__":
    test_registrations()
    test_perturbation_sampler()
    test_perturbation_apply()
    test_ctc_vocab()
    test_eval_result()
    print("\n✓ All tests passed!")
