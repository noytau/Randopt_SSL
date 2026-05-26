"""
ImageNet-1K Classification Task for DINOv2.

Uses HuggingFace imagenet-1k dataset (requires HF login / access).
For quick experiments without HF access, falls back to CIFAR-100 (100 classes)
which provides a good proxy for large-scale visual recognition.

Batch format: {"pixel_values": [B, 3, 224, 224], "labels": [B]}
DINOv2 extract_features → [B, 1, 1024] (CLS token)
Head: CLS pool → [B, 1024] → Linear(1024, n_classes)

Primary metric: Top-1 Accuracy ↑
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.interfaces import Task
from src.registry import register_task

logger = logging.getLogger(__name__)


# ── Head ──────────────────────────────────────────────────────────────────────

class VisionClassificationHead(nn.Module):
    """CLS-pool + linear classifier for vision tasks."""

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, 1, D] or [B, D] → logits [B, n_classes]"""
        if features.dim() == 3:
            x = features[:, 0, :]   # CLS token → [B, D]
        else:
            x = features
        return self.classifier(self.dropout(x))


# ── Dataset ──────────────────────────────────────────────────────────────────

class HFVisionDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace image classification dataset."""

    def __init__(self, hf_dataset, processor, image_key: str = "image",
                 label_key: str = "label"):
        self.dataset = hf_dataset
        self.processor = processor
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_key]

        # Ensure RGB
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        enc = self.processor(images=image, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)   # [3, H, W]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(item[self.label_key], dtype=torch.long),
        }


def vision_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


# ── Task ──────────────────────────────────────────────────────────────────────

@register_task("imagenet_cls")
class ImageNetClassificationTask(Task):
    """
    ImageNet-1K Top-1 classification with DINOv2.

    Falls back to CIFAR-100 if imagenet-1k is unavailable (no HF token).
    CIFAR-100: 100 classes, 50K train / 10K test, 32×32 upsampled to 224×224.

    Primary metric: Top-1 Accuracy ↑
    """

    def __init__(
        self,
        use_cifar100_fallback: bool = True,
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        self._use_cifar100 = use_cifar100_fallback
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloaders: Dict[str, DataLoader] = {}
        self._processor = None
        self._n_classes = 100 if use_cifar100_fallback else 1000

    def name(self) -> str:
        return "imagenet_cls"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("accuracy", True)

    def set_processor(self, processor):
        self._processor = processor

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        if self._use_cifar100:
            hf_split = {"train": "train", "val": "test", "test": "test"}.get(split, split)
            logger.info(f"  Loading CIFAR-100 {hf_split} (ImageNet proxy)...")
            ds = load_dataset("cifar100", split=hf_split, trust_remote_code=True)
            image_key, label_key = "img", "fine_label"
        else:
            hf_split = {"train": "train", "val": "validation", "test": "validation"}.get(split, split)
            logger.info(f"  Loading ImageNet-1K {hf_split}...")
            ds = load_dataset("imagenet-1k", split=hf_split, trust_remote_code=True)
            image_key, label_key = "image", "label"

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        dataset = HFVisionDataset(ds, self._processor, image_key=image_key, label_key=label_key)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=vision_collate,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} samples, {len(loader)} batches.")
        return loader

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        return VisionClassificationHead(input_dim, self._n_classes).to(device)

    def get_loss_fn(self):
        ce = nn.CrossEntropyLoss()

        def loss_fn(logits: torch.Tensor, batch: Dict) -> torch.Tensor:
            return ce(logits, batch["labels"].to(logits.device))

        return loss_fn

    def evaluate(
        self,
        model,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        head.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                from src.randopt.core import RandOptEnsemble
                if isinstance(model, RandOptEnsemble):
                    features = model.extract_features_ensemble(batch_dev)
                else:
                    features = model.extract_features(batch_dev)
                logits = head(features)
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_dev["labels"]).sum().item()
                total += batch_dev["labels"].shape[0]

        head.train()
        return {"accuracy": correct / max(total, 1)}


# ── Few-shot classification task ──────────────────────────────────────────────

@register_task("fewshot_cls")
class FewShotClassificationTask(Task):
    """
    5-way 5-shot classification using DINOv2 features on CIFAR-100.

    For each episode:
      - Sample 5 classes
      - 5 support examples per class (25 total)
      - 15 query examples per class (75 total)
    Nearest-centroid classifier in feature space.

    Primary metric: 5-way 5-shot accuracy ↑
    Note: RandOpt can optimize feature quality directly for this metric,
    which is non-differentiable (nearest-centroid matching).
    """

    def __init__(
        self,
        n_way: int = 5,
        n_shot: int = 5,
        n_query: int = 15,
        n_episodes: int = 100,
        batch_size: int = 32,
        num_workers: int = 2,
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
    ):
        self._n_way = n_way
        self._n_shot = n_shot
        self._n_query = n_query
        self._n_episodes = n_episodes
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloaders: Dict[str, DataLoader] = {}
        self._processor = None
        # For few-shot, we don't limit samples via max_* since episodes handle it

    def name(self) -> str:
        return "fewshot_cls"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("accuracy", True)

    def set_processor(self, processor):
        self._processor = processor

    def load_data(self, split: str = "train") -> DataLoader:
        """Return flat dataloader for feature extraction; actual few-shot eval uses evaluate()."""
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        hf_split = {"train": "train", "val": "test", "test": "test"}.get(split, split)
        logger.info(f"  Loading CIFAR-100 {hf_split} for few-shot...")
        ds = load_dataset("cifar100", split=hf_split, trust_remote_code=True)

        dataset = HFVisionDataset(ds, self._processor, image_key="img", label_key="fine_label")
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            collate_fn=vision_collate,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} samples.")
        return loader

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        """Few-shot uses nearest-centroid; head is identity (features go straight to evaluate)."""
        return nn.Identity()

    def get_loss_fn(self):
        """Dummy loss (few-shot doesn't train a head via gradient descent)."""
        def loss_fn(logits, batch):
            return torch.tensor(0.0)
        return loss_fn

    def evaluate(
        self,
        model,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Episode-based few-shot evaluation.
        1. Extract all features from dataloader.
        2. For each episode: sample N-way K-shot, compute centroids, classify queries.
        3. Average accuracy over episodes.
        """
        import random

        # Step 1: Extract all features
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                from src.randopt.core import RandOptEnsemble
                if isinstance(model, RandOptEnsemble):
                    feats = model.extract_features_ensemble(batch_dev)
                else:
                    feats = model.extract_features(batch_dev)

                # Pool: [B, 1, D] → [B, D]
                if feats.dim() == 3:
                    feats = feats[:, 0, :]
                all_features.append(feats.cpu())
                all_labels.extend(batch["labels"].tolist())

        all_features = torch.cat(all_features, dim=0)   # [N_total, D]

        # Group by label
        label_to_indices: Dict[int, List[int]] = {}
        for i, lbl in enumerate(all_labels):
            label_to_indices.setdefault(lbl, []).append(i)
        classes = [c for c, idxs in label_to_indices.items() if len(idxs) >= self._n_shot + self._n_query]

        if len(classes) < self._n_way:
            logger.warning(f"  Not enough classes for {self._n_way}-way episodes; using {len(classes)}")
            return {"accuracy": 0.0}

        # Step 2: Run episodes
        rng = random.Random(42)
        episode_accs = []

        for _ in range(self._n_episodes):
            episode_classes = rng.sample(classes, self._n_way)
            support_feats, query_feats, query_labels = [], [], []

            for cls_idx, cls in enumerate(episode_classes):
                idxs = rng.sample(label_to_indices[cls], self._n_shot + self._n_query)
                support_idxs = idxs[:self._n_shot]
                query_idxs = idxs[self._n_shot:]

                support_feats.append(all_features[support_idxs].mean(0))   # centroid [D]
                query_feats.extend(all_features[q] for q in query_idxs)
                query_labels.extend([cls_idx] * self._n_query)

            centroids = torch.stack(support_feats)          # [n_way, D]
            queries = torch.stack(query_feats)              # [n_query*n_way, D]
            q_labels = torch.tensor(query_labels)

            # Cosine nearest centroid
            centroids_n = nn.functional.normalize(centroids, dim=-1)
            queries_n = nn.functional.normalize(queries, dim=-1)
            sims = queries_n @ centroids_n.T                 # [n_q, n_way]
            preds = sims.argmax(dim=-1)
            acc = (preds == q_labels).float().mean().item()
            episode_accs.append(acc)

        mean_acc = sum(episode_accs) / len(episode_accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in episode_accs) / len(episode_accs)) ** 0.5
        return {"accuracy": mean_acc, "accuracy_std": std_acc}
