"""
Linear Probe: freeze encoder, train only a linear head.
This is both a baseline and a utility used by RandOpt (to warm-start the head).
"""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.interfaces import AdaptationMethod, SSLModel, Task
from src.registry import register_method

logger = logging.getLogger(__name__)


def train_linear_head(
    model: SSLModel,
    task: Task,
    device: torch.device,
    config: Dict[str, Any],
) -> nn.Module:
    """
    Train a linear head on frozen encoder features.
    Shared between LinearProbe method and RandOpt head warm-start.
    """
    lr = config.get("lr", 1e-3)
    epochs = config.get("epochs", 10)
    weight_decay = config.get("weight_decay", 1e-4)

    train_loader = task.load_data("train")
    val_loader = task.load_data("val")
    head = task.build_head(model.hidden_dim(), device).to(device)
    metric_name, higher_is_better = task.primary_metric()

    # Few-shot and similar tasks use Identity heads with no parameters — skip training.
    if not list(head.parameters()):
        logger.info("  Head has no trainable parameters, skipping head training.")
        return head

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = task.get_loss_fn()

    best_score = float("-inf") if higher_is_better else float("inf")
    best_head_state = None

    for epoch in range(epochs):
        # ── Train ──
        head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"  Head epoch {epoch+1}/{epochs}", leave=False):
            batch = _to_device(batch, device)
            with torch.no_grad():
                features = model.extract_features(batch)
            logits = head(features)
            loss = criterion(logits, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ── Validate ──
        metrics = task.evaluate(model, head, val_loader, device)
        score = metrics[metric_name]

        improved = (score > best_score) if higher_is_better else (score < best_score)
        if improved:
            best_score = score
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}

        logger.info(
            f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
            f"{metric_name}={score:.4f} {'*' if improved else ''}"
        )

    # Restore best head
    if best_head_state is not None:
        head.load_state_dict(best_head_state)
    return head


def _to_device(batch: Dict, device: torch.device) -> Dict:
    """Move batch tensors to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@register_method("linear_probe")
class LinearProbe(AdaptationMethod):
    """Baseline: frozen encoder + trained linear head."""

    def name(self) -> str:
        return "linear_probe"

    def adapt(
        self,
        model: SSLModel,
        task: Task,
        device: torch.device,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module]:
        head_config = {
            "lr": config.get("head_lr", 1e-3),
            "epochs": config.get("head_epochs", 10),
            "weight_decay": config.get("weight_decay", 1e-4),
        }
        head = train_linear_head(model, task, device, head_config)
        encoder = model.get_encoder()
        return encoder, head
