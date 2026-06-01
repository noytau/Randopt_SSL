"""
LLM baselines for generative tasks: pass@1 and majority_vote@N.

These are the standard baselines from the Neural Thickets paper against which
RandOpt is compared on Countdown, GSM8K, and other generation tasks.

pass@1:
  Greedy decode from the pretrained model — no perturbation, no sampling.
  This is the minimum bar RandOpt must beat to show any benefit.

majority_vote@N:
  Sample N responses from the pretrained model (temperature > 0), take the
  majority vote. Equal inference compute to RandOpt (N forward passes),
  but uses the SAME base model for all N — no weight diversity.
  This isolates the contribution of *weight diversity* vs. simple sampling.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from src.interfaces import AdaptationMethod, SSLModel, Task
from src.registry import register_method

logger = logging.getLogger(__name__)


def _is_generative(task: Task) -> bool:
    from src.tasks.llm.base import GenerativeTask
    return isinstance(task, GenerativeTask)


# ── Pass@1 ────────────────────────────────────────────────────────────────────

@register_method("passatone")
class PassAtOne(AdaptationMethod):
    """
    Greedy decode from base model — no perturbation, no sampling.
    The identity baseline: measures what the pretrained model can do alone.
    """

    def name(self) -> str:
        return "passatone"

    def adapt(
        self,
        model: SSLModel,
        task: Task,
        device: torch.device,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module]:
        if not _is_generative(task):
            raise ValueError("passatone baseline is only for generative (LLM) tasks.")
        head = task.build_head(1, device)
        logger.info("  passatone: using base model with greedy decode (no perturbation).")
        return model, head


# ── Majority Vote @ N ─────────────────────────────────────────────────────────

class MajorityVoteWrapper(nn.Module):
    """
    Wraps a CausalLMModel to produce majority-voted outputs from N samples.
    At inference: generate N responses (with temperature), take majority vote.
    """

    def __init__(self, model, n_samples: int, temperature: float):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature

    def generate(self, prompts: List[str]) -> List[str]:
        """Generate n_samples responses per prompt, return majority vote."""
        per_run: List[List[str]] = []
        for _ in range(self.n_samples):
            responses = self.model.generate(
                prompts,
                temperature=self.temperature,
                do_sample=True,
            )
            per_run.append(responses)

        # Majority vote per prompt
        results = []
        for i in range(len(prompts)):
            candidates = [per_run[j][i] for j in range(self.n_samples)]
            winner = Counter(candidates).most_common(1)[0][0]
            results.append(winner)
        return results

    # Forward compatibility with GenerativeTask.evaluate()
    def generate_ensemble(self, prompts: List[str]) -> List[str]:
        return self.generate(prompts)


@register_method("majority_vote")
class MajorityVote(AdaptationMethod):
    """
    Sample N responses from the base model, take majority vote.
    Equal compute to RandOpt (N forward passes), but NO weight perturbation.
    Ablates whether RandOpt's benefit comes from weight diversity or sampling diversity.

    N is read from config["n_candidates"] for apples-to-apples comparison with RandOpt.
    Temperature defaults to 0.7 (enough diversity for meaningful voting).
    """

    def name(self) -> str:
        return "majority_vote"

    def adapt(
        self,
        model: SSLModel,
        task: Task,
        device: torch.device,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module]:
        if not _is_generative(task):
            raise ValueError("majority_vote baseline is only for generative (LLM) tasks.")

        n_samples = config.get("top_k", 50)   # match RandOpt ensemble size K (not N, for inference cost)
        temperature = config.get("mv_temperature", 0.7)

        logger.info(
            f"  majority_vote: {n_samples} samples at temperature={temperature} "
            f"(no perturbation, base model only)."
        )

        wrapper = MajorityVoteWrapper(model, n_samples=n_samples, temperature=temperature)
        head = task.build_head(1, device)
        return wrapper, head
