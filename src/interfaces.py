"""
Abstract interfaces for models, tasks, and adaptation methods.
Every concrete implementation inherits from these.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── Result Container ────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Standardized result from any method × task evaluation."""
    model_name: str
    task_name: str
    method_name: str
    metrics: Dict[str, float]          # e.g. {"accuracy": 0.95, "wer": 0.04}
    primary_metric_name: str           # which metric is "the" one for comparison
    primary_metric_value: float
    higher_is_better: bool
    extra: Dict[str, Any] = field(default_factory=dict)  # anything else (timing, etc.)

    def summary(self) -> str:
        sign = "↑" if self.higher_is_better else "↓"
        return (f"[{self.model_name} | {self.task_name} | {self.method_name}] "
                f"{self.primary_metric_name}={self.primary_metric_value:.4f} {sign}")


# ── Model Interface ─────────────────────────────────────────────────────────

class SSLModel(ABC):
    """
    Wraps a pretrained SSL backbone.
    Responsibilities: load weights, extract features, provide state_dict access.
    """

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Download / load pretrained weights."""
        ...

    @abstractmethod
    def get_encoder(self) -> nn.Module:
        """Return the encoder module (for feature extraction and weight perturbation)."""
        ...

    @abstractmethod
    def extract_features(self, batch: Any) -> torch.Tensor:
        """
        Run a batch through the frozen encoder. 
        Returns: [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim].
        """
        ...

    @abstractmethod
    def hidden_dim(self) -> int:
        """Dimensionality of the encoder output."""
        ...

    @abstractmethod
    def get_pretrained_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a copy of the original pretrained weights (never modified)."""
        ...


# ── Task Interface ──────────────────────────────────────────────────────────

class Task(ABC):
    """
    Defines a downstream task: data loading, head architecture, evaluation.
    """

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def primary_metric(self) -> Tuple[str, bool]:
        """Returns (metric_name, higher_is_better)."""
        ...

    @abstractmethod
    def load_data(self, split: str = "train") -> DataLoader:
        """Load data for a given split. Splits: 'train', 'val', 'test'."""
        ...

    @abstractmethod
    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        """
        Build a lightweight task head.
        For linear probe: just nn.Linear.
        For more complex tasks (ASR): may include CTC decoder, etc.
        """
        ...

    @abstractmethod
    def evaluate(
        self,
        model: SSLModel,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Run evaluation, return dict of metrics.
        Must include the primary metric from self.primary_metric().
        """
        ...


# ── Method Interface ────────────────────────────────────────────────────────

class AdaptationMethod(ABC):
    """
    An adaptation strategy: how to go from pretrained model → task-adapted model.
    """

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def adapt(
        self,
        model: SSLModel,
        task: Task,
        device: torch.device,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Adapt the model to the task.
        Returns: (adapted_encoder_or_original, trained_head)
        """
        ...
