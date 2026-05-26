"""
Full Fine-tuning: unfreeze encoder + train head end-to-end with SGD.
The standard baseline that RandOpt aims to match/beat while preventing CF.
"""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.interfaces import AdaptationMethod, SSLModel, Task
from src.registry import register_method

logger = logging.getLogger(__name__)


@register_method("finetune")
class FineTune(AdaptationMethod):
    """Baseline: unfreeze encoder, fine-tune everything end-to-end."""

    def name(self) -> str:
        return "finetune"

    def adapt(
        self,
        model: SSLModel,
        task: Task,
        device: torch.device,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module]:
        lr_encoder = config.get("lr_encoder", 1e-5)
        lr_head = config.get("lr_head", 1e-3)
        epochs = config.get("ft_epochs", 5)
        weight_decay = config.get("weight_decay", 1e-4)
        max_grad_norm = config.get("max_grad_norm", 1.0)

        encoder = model.get_encoder()
        head = task.build_head(model.hidden_dim(), device).to(device)
        train_loader = task.load_data("train")
        val_loader = task.load_data("val")
        metric_name, higher_is_better = task.primary_metric()
        criterion = task.get_loss_fn()

        # Separate learning rates for encoder vs head
        optimizer = torch.optim.AdamW([
            {"params": encoder.parameters(), "lr": lr_encoder},
            {"params": head.parameters(), "lr": lr_head},
        ], weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(train_loader)
        )

        best_score = float("-inf") if higher_is_better else float("inf")
        best_encoder_state = None
        best_head_state = None

        encoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"  FT epoch {epoch+1}/{epochs}", leave=False):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                features = model.extract_features(batch)
                logits = head(features)
                loss = criterion(logits, batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()),
                    max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # ── Validate ──
            encoder.eval()
            metrics = task.evaluate(model, head, val_loader, device)
            encoder.train()

            score = metrics[metric_name]
            improved = (score > best_score) if higher_is_better else (score < best_score)
            if improved:
                best_score = score
                best_encoder_state = {k: v.clone() for k, v in encoder.state_dict().items()}
                best_head_state = {k: v.clone() for k, v in head.state_dict().items()}

            logger.info(
                f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
                f"{metric_name}={score:.4f} {'*' if improved else ''}"
            )

        # Restore best
        if best_encoder_state is not None:
            encoder.load_state_dict(best_encoder_state)
            head.load_state_dict(best_head_state)

        encoder.eval()
        return encoder, head
