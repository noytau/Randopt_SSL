"""
RandOpt: Random Perturbation Search for Task Adaptation.

Core algorithm:
1. Keep pretrained weights M frozen (never modified)
2. Sample N Gaussian perturbations around M
3. Evaluate each perturbed model on a fitness function
4. Select top-K by fitness
5. Ensemble top-K for final prediction
"""

import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.interfaces import AdaptationMethod, EvalResult, SSLModel, Task
from src.registry import register_method

logger = logging.getLogger(__name__)


class PerturbationSampler:
    """Samples weight perturbations using counter-based RNG (reproducible from seed alone)."""

    def __init__(self, sigma: float, device: torch.device):
        self.sigma = sigma
        self.device = device

    def sample(
        self, reference_state: Dict[str, torch.Tensor], seed: int
    ) -> Dict[str, torch.Tensor]:
        """Sample epsilon ~ N(0, sigma^2 * I) for each parameter tensor."""
        rng = torch.Generator(device="cpu").manual_seed(seed)
        epsilon = {}
        for key, param in reference_state.items():
            noise = torch.randn(param.shape, generator=rng, device="cpu") * self.sigma
            epsilon[key] = noise.to(self.device)
        return epsilon

    def apply(
        self, reference_state: Dict[str, torch.Tensor], epsilon: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """M' = M + epsilon"""
        return {k: reference_state[k] + epsilon[k] for k in reference_state}


@register_method("randopt")
class RandOpt(AdaptationMethod):
    """
    RandOpt adaptation: search for task-specific perturbations around pretrained weights.
    CF prevention is structural: M is never modified, all ensemble members stay in N(M, sigma^2).
    """

    def name(self) -> str:
        return "randopt"

    def adapt(
        self,
        model: SSLModel,
        task: Task,
        device: torch.device,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module]:
        """
        RandOpt adaptation pipeline:
        1. Train a linear head on frozen features (warm start)
        2. Search for top-K perturbations using fitness = task metric on val set
        3. Return ensemble wrapper + head
        """
        # ── Config ──
        sigma = config.get("sigma", 0.01)
        n_candidates = config.get("n_candidates", 100)
        top_k = config.get("top_k", 5)
        n_rounds = config.get("n_rounds", 1)
        fitness_subsample = config.get("fitness_subsample", 1.0)  # fraction of val data

        logger.info(
            f"RandOpt: sigma={sigma}, N={n_candidates}, K={top_k}, rounds={n_rounds}"
        )

        # ── Step 1: Train linear head on frozen features ──
        head = self._train_head(model, task, device, config)

        # ── Step 2: Search perturbations ──
        M = model.get_pretrained_state_dict()
        sampler = PerturbationSampler(sigma=sigma, device=device)
        val_loader = task.load_data("val")
        metric_name, higher_is_better = task.primary_metric()

        all_candidates: List[Tuple[int, float]] = []

        for round_idx in range(n_rounds):
            logger.info(f"  Round {round_idx+1}/{n_rounds}")
            round_candidates = []

            for i in tqdm(range(n_candidates), desc=f"  Evaluating perturbations (round {round_idx+1})"):
                seed = round_idx * n_candidates + i
                epsilon = sampler.sample(M, seed)
                perturbed = sampler.apply(M, epsilon)

                # Load perturbed weights into encoder
                encoder = model.get_encoder()
                encoder.load_state_dict(perturbed, strict=False)

                # Evaluate fitness
                metrics = task.evaluate(model, head, val_loader, device)
                score = metrics[metric_name]
                if not higher_is_better:
                    score = -score  # normalize so higher = better

                round_candidates.append((seed, score))

            # Sort and keep top-K from this round
            round_candidates.sort(key=lambda x: x[1], reverse=True)
            all_candidates.extend(round_candidates[:top_k])

        # ── Step 3: Select global top-K ──
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        top_k_seeds = [c[0] for c in all_candidates[:top_k]]
        top_k_scores = [c[1] for c in all_candidates[:top_k]]

        logger.info(f"  Top-{top_k} seeds: {top_k_seeds}")
        logger.info(f"  Top-{top_k} scores: {[f'{s:.4f}' for s in top_k_scores]}")

        # ── Restore original weights ──
        encoder = model.get_encoder()
        encoder.load_state_dict(M, strict=False)

        # ── Build ensemble wrapper ──
        ensemble = RandOptEnsemble(
            model=model,
            sampler=sampler,
            reference_state=M,
            top_k_seeds=top_k_seeds,
        )

        return ensemble, head

    def _train_head(
        self, model: SSLModel, task: Task, device: torch.device, config: Dict[str, Any]
    ) -> nn.Module:
        """Train a linear head on frozen encoder features. Reused by RandOpt and linear probe."""
        from src.baselines.linear_probe import train_linear_head
        head_config = {
            "lr": config.get("head_lr", 1e-3),
            "epochs": config.get("head_epochs", 10),
            "batch_size": config.get("batch_size", 32),
        }
        return train_linear_head(model, task, device, head_config)


class RandOptEnsemble(nn.Module):
    """
    Wraps the pretrained model + top-K perturbation seeds.
    At inference: runs input through K perturbed models, averages predictions.
    """

    def __init__(
        self,
        model: SSLModel,
        sampler: PerturbationSampler,
        reference_state: Dict[str, torch.Tensor],
        top_k_seeds: List[int],
    ):
        super().__init__()
        self.model = model
        self.sampler = sampler
        self.reference_state = reference_state
        self.top_k_seeds = top_k_seeds

    def extract_features_ensemble(self, batch: Any) -> torch.Tensor:
        """Average features across K perturbed encoders."""
        all_features = []
        encoder = self.model.get_encoder()

        for seed in self.top_k_seeds:
            eps = self.sampler.sample(self.reference_state, seed)
            perturbed = self.sampler.apply(self.reference_state, eps)
            encoder.load_state_dict(perturbed, strict=False)

            with torch.no_grad():
                feats = self.model.extract_features(batch)
            all_features.append(feats)

        # Restore original weights
        encoder.load_state_dict(self.reference_state, strict=False)

        return torch.stack(all_features).mean(dim=0)
