"""
Monocular Depth Estimation on NYU Depth V2.
Pixel-wise depth regression.
Metrics: RMSE ↓ (primary), δ₁ ↑ (% pixels where max(pred/gt, gt/pred) < 1.25)

IMPORTANT: Model must be configured with output_cls_only=False (DINOv2).
  Set model_config: {output_cls_only: false} in your YAML config.

Model: DINOv2 → [B, N+1, 1024] → skip CLS → [B, N, 1024] → Linear(1024, 1)
  → reshape → [B, 1, H_p, W_p] → upsample → [B, 1, 224, 224] → squeeze → [B, 224, 224]
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.interfaces import Task
from src.registry import register_task

logger = logging.getLogger(__name__)

IMG_SIZE = 224


# ── Head ──────────────────────────────────────────────────────────────────────

class DepthHead(nn.Module):
    """
    Takes [B, N+1, 1024] DINOv2 features (CLS + patches).
    Skips CLS, applies linear to get depth per patch, reshapes and upsamples.

    Output: [B, 224, 224] — pixel-wise depth predictions.
    """

    def __init__(self, input_dim: int, patch_size: int = 14, img_size: int = IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.n_patches_side = img_size // patch_size
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B, N+1, D]
        Returns: [B, img_size, img_size]
        """
        patch_features = features[:, 1:, :]   # [B, N, D], skip CLS
        B, N, D = patch_features.shape
        n_side = self.n_patches_side

        depth = self.proj(patch_features)   # [B, N, 1]
        depth = depth.reshape(B, n_side, n_side, 1)   # [B, H_p, W_p, 1]
        depth = depth.permute(0, 3, 1, 2)              # [B, 1, H_p, W_p]

        depth = F.interpolate(
            depth,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )   # [B, 1, img_size, img_size]

        return depth.squeeze(1)   # [B, img_size, img_size]


# ── Dataset ──────────────────────────────────────────────────────────────────

class DepthDataset(Dataset):
    """NYU Depth V2 dataset wrapper."""

    def __init__(self, hf_dataset, processor, img_size: int = IMG_SIZE):
        self.dataset = hf_dataset
        self.processor = processor
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        depth_map = item["depth_map"]

        # Ensure RGB image
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        enc = self.processor(images=image, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)   # [3, H, W]

        # Convert depth map to tensor and resize to img_size
        if hasattr(depth_map, "getdata"):
            # PIL image
            import numpy as np
            depth_np = np.array(depth_map, dtype=np.float32)
        elif isinstance(depth_map, torch.Tensor):
            depth_np = depth_map.numpy().astype("float32")
        else:
            import numpy as np
            depth_np = np.array(depth_map, dtype=np.float32)

        depth_t = torch.tensor(depth_np).unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        depth_resized = F.interpolate(
            depth_t,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze()   # [img_size, img_size]

        return {
            "pixel_values": pixel_values,
            "depth": depth_resized,   # [img_size, img_size] float
        }


def depth_collate(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    depth = torch.stack([item["depth"] for item in batch])
    return {"pixel_values": pixel_values, "depth": depth}


# ── Task ─────────────────────────────────────────────────────────────────────

@register_task("depth_estimation")
class DepthEstimationTask(Task):
    """
    Monocular Depth Estimation on NYU Depth V2.
    Primary metric: RMSE ↓ (lower is better).
    Also computes δ₁ = % pixels where max(pred/gt, gt/pred) < 1.25.

    IMPORTANT: DINOv2 model must be configured with output_cls_only=False.
    Set in YAML: model_config: {output_cls_only: false}
    The evaluate() method will temporarily set model._output_cls_only=False if needed.
    """

    def __init__(
        self,
        hf_dataset_id: str = "sayakpaul/nyu_depth_v2",
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        batch_size: int = 16,
        num_workers: int = 2,
        img_size: int = IMG_SIZE,
        patch_size: int = 14,   # DINOv2 ViT-L/14
    ):
        self._hf_dataset_id = hf_dataset_id
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._img_size = img_size
        self._patch_size = patch_size
        self._dataloaders: Dict[str, DataLoader] = {}
        self._processor = None

    def name(self) -> str:
        return "depth_estimation"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("rmse", False)   # lower is better

    def set_processor(self, processor):
        self._processor = processor

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        split_map = {"train": "train", "val": "validation", "test": "validation"}
        hf_split = split_map.get(split, split)

        logger.info(f"  Loading {self._hf_dataset_id} {hf_split}...")
        try:
            ds = load_dataset(self._hf_dataset_id, split=hf_split, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"  Failed to load {hf_split} split: {e}. Trying 'train'...")
            ds = load_dataset(self._hf_dataset_id, split="train", trust_remote_code=True)

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        dataset = DepthDataset(ds, self._processor, img_size=self._img_size)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=depth_collate,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} samples, {len(loader)} batches.")
        return loader

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        return DepthHead(
            input_dim,
            patch_size=self._patch_size,
            img_size=self._img_size,
        ).to(device)

    def get_loss_fn(self):
        l1 = nn.L1Loss()

        def loss_fn(preds: torch.Tensor, batch: Dict) -> torch.Tensor:
            depth_gt = batch["depth"].to(preds.device)
            return l1(preds, depth_gt)

        return loss_fn

    def evaluate(
        self,
        model,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Compute RMSE and δ₁ on given dataloader.
        Temporarily sets model._output_cls_only=False so patch tokens are returned.
        """
        head.eval()

        _original_cls_only = None
        if hasattr(model, "_output_cls_only"):
            _original_cls_only = model._output_cls_only
            model._output_cls_only = False

        sq_errors = []
        delta1_hits = 0
        total_pixels = 0

        with torch.no_grad():
            for batch in dataloader:
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                from src.randopt.core import RandOptEnsemble
                if isinstance(model, RandOptEnsemble):
                    features = model.extract_features_ensemble(batch_dev)
                else:
                    features = model.extract_features(batch_dev)

                if features.shape[1] == 1:
                    logger.warning(
                        "DepthEstimationTask.evaluate: features has only 1 token (CLS). "
                        "Set model output_cls_only=False for dense tasks."
                    )

                preds = head(features)   # [B, H, W]
                gt = batch_dev["depth"]  # [B, H, W]

                # Only evaluate where gt > 0 (valid depth)
                valid = gt > 0
                if valid.sum() == 0:
                    continue

                pred_valid = preds[valid]
                gt_valid = gt[valid]

                # RMSE accumulation
                sq_errors.append(((pred_valid - gt_valid) ** 2).cpu())

                # δ₁: max(pred/gt, gt/pred) < 1.25
                ratio = torch.max(pred_valid / (gt_valid + 1e-8), gt_valid / (pred_valid.clamp(min=1e-8)))
                delta1_hits += (ratio < 1.25).sum().item()
                total_pixels += valid.sum().item()

        if _original_cls_only is not None:
            model._output_cls_only = _original_cls_only

        if sq_errors:
            all_sq = torch.cat(sq_errors)
            rmse = all_sq.mean().sqrt().item()
        else:
            rmse = float("inf")

        delta1 = delta1_hits / max(total_pixels, 1)

        head.train()
        return {"rmse": rmse, "delta1": delta1}
