"""
GenerativeTask: base class for LLM generation tasks.

Unlike encoder tasks (which use extract_features → head → metric),
generative tasks call model.generate(prompts) → text → task-specific verifier.

Ensemble evaluation:
  - If model is RandOptEnsemble: each of K perturbed models generates independently,
    then majority vote over K responses determines the final answer.
  - If model is a plain CausalLMModel: single forward pass.

Subclasses must implement:
  format_prompt(sample: dict) -> str
  verify(response: str, sample: dict) -> bool
  load_data(split) -> DataLoader  (items must have key "prompt" and "expected")
"""

import logging
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.interfaces import Task

logger = logging.getLogger(__name__)


class GenerativeTask(Task):
    """
    Base class for generation-based evaluation tasks.
    No linear head — the model IS the predictor.
    """

    max_new_tokens: int = 256   # overridden by run.py from config["max_new_tokens"]

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        """LLMs don't need a head; return a no-op Identity module."""
        return nn.Identity().to(device)

    @abstractmethod
    def format_prompt(self, sample: Dict) -> str:
        """Format a dataset sample into a text prompt for the model."""
        ...

    @abstractmethod
    def verify(self, response: str, sample: Dict) -> bool:
        """Check if the model's response is correct for the given sample."""
        ...

    def get_answer_text(self, sample: Dict) -> Optional[str]:
        """
        Return the gold answer as a plain string, used for SFT training.
        Override in subclasses. Default returns None (task doesn't support SFT).
        """
        return None

    def evaluate(
        self,
        model: Any,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Run evaluation using generation + verification.

        Handles both plain CausalLMModel and RandOptEnsemble:
          - RandOptEnsemble: calls generate_ensemble() → majority vote per sample
          - CausalLMModel: calls generate() directly
        """
        from src.randopt.core import RandOptEnsemble

        correct = 0
        total = 0

        for batch in dataloader:
            prompts: List[str] = batch["prompts"]
            samples: List[Dict] = batch["samples"]   # raw sample dicts for verify()

            # Duck typing: RandOptEnsemble has generate_ensemble(), everything else has generate()
            gen_kwargs = {"max_new_tokens": self.max_new_tokens}
            if isinstance(model, RandOptEnsemble):
                responses = model.generate_ensemble(prompts, **gen_kwargs)
            elif hasattr(model, "generate"):
                responses = model.generate(prompts, **gen_kwargs)
            else:
                raise TypeError(
                    f"GenerativeTask.evaluate() requires a model with generate() method, "
                    f"got {type(model).__name__}"
                )

            for response, sample in zip(responses, samples):
                if self.verify(response, sample):
                    correct += 1
                total += 1

        accuracy = correct / max(total, 1)
        logger.info(f"  {self.name()} eval: {correct}/{total} correct ({accuracy:.3f})")
        return {"accuracy": accuracy}
