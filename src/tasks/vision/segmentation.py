"""
Semantic Segmentation Task on ADE20K (scene_parse_150).
150 semantic classes (pixel label 0 = background/unlabeled, ignored in loss).
Metric: mIoU ↑ (mean Intersection over Union, non-differentiable)

IMPORTANT: Model must be configured with output_cls_only=False (DINOv2).
  Set model_config: {output_cls_only: false} in your YAML config,
  or pass it when constructing the model.

Model: DINOv2 → [B, N+1, 1024] → skip CLS → [B, N, 1024] → Linear → [B, N, 150]
  → reshape → [B, 150, H_p, W_p] → bilinear upsample → [B, 150, 224, 224]
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

N_CLASSES = 150    # ADE20K classes 1-150 (0 = background, ignored)
IMG_SIZE = 224


# ── Head ──────────────────────────────────────────────────────────────────────

class SegmentationHead(nn.Module):
    """
    Takes [B, N+1, 1024] DINOv2 features (CLS + patches).
    Skips CLS token, applies linear to patch tokens, reshapes and upsamples.

    Output: [B, n_classes, img_size, img_size]
    """

    def __init__(self, input_dim: int, n_classes: int = N_CLASSES, patch_size: int = 16,
                 img_size: int = IMG_SIZE):
        super().__init__()
        self.n_classes = n_classes
        self.img_size = img_size
        # Number of patches along each side for a 224×224 image with patch_size=14 (DINOv2 ViT-L/14)
        self.n_patches_side = img_size // patch_size
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B, N+1, D] where N+1 = CLS + patches
        Returns: [B, n_classes, img_size, img_size]
        """
        # Skip CLS token
        patch_features = features[:, 1:, :]   # [B, N, D]
        B, N, D = patch_features.shape
        n_side = self.n_patches_side

        # Apply linear projection
        logits = self.classifier(patch_features)   # [B, N, n_classes]

        # Reshape to spatial grid
        logits = logits.reshape(B, n_side, n_side, self.n_classes)   # [B, H_p, W_p, C]
        logits = logits.permute(0, 3, 1, 2)                           # [B, C, H_p, W_p]

        # Upsample to full resolution
        logits = F.interpolate(
            logits,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )   # [B, n_classes, img_size, img_size]

        return logits


# ── Dataset ──────────────────────────────────────────────────────────────────

class ADE20KDataset(Dataset):
    """ADE20K (scene_parse_150) dataset wrapper for semantic segmentation."""

    def __init__(self, hf_dataset, processor, img_size: int = IMG_SIZE):
        self.dataset = hf_dataset
        self.processor = processor
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        annotation = item["annotation"]

        # Ensure RGB
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        # Process image with DINOv2 processor
        enc = self.processor(images=image, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)   # [3, H, W]

        # Resize annotation to img_size × img_size with nearest-neighbor
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage
        if not isinstance(annotation, torch.Tensor):
            annotation_t = torch.tensor(
                list(annotation.getdata()), dtype=torch.long
            ).reshape(annotation.size[1], annotation.size[0])
        else:
            annotation_t = annotation.long()

        # Resize annotation using nearest neighbor (preserve label integers)
        annotation_pil = PILImage.fromarray(annotation_t.numpy().astype("uint8"))
        annotation_resized = annotation_pil.resize(
            (self.img_size, self.img_size), PILImage.NEAREST
        )
        annotation_tensor = torch.tensor(
            list(annotation_resized.getdata()), dtype=torch.long
        ).reshape(self.img_size, self.img_size)

        return {
            "pixel_values": pixel_values,
            "labels": annotation_tensor,   # [img_size, img_size], values 0-150
        }


def segmentation_collate(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# ── Task ─────────────────────────────────────────────────────────────────────

@register_task("segmentation")
class SegmentationTask(Task):
    """
    Semantic Segmentation on ADE20K (scene_parse_150).
    150 classes + background (ignored at pixel label 0).
    Primary metric: mIoU ↑ (mean Intersection over Union).

    IMPORTANT: DINOv2 model must be configured with output_cls_only=False.
    Set in YAML: model_config: {output_cls_only: false}
    The evaluate() method will temporarily set model._output_cls_only=False if needed.
    """

    def __init__(
        self,
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        batch_size: int = 16,
        num_workers: int = 2,
        img_size: int = IMG_SIZE,
        patch_size: int = 14,   # DINOv2 ViT-L/14
    ):
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
        return "segmentation"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("miou", True)

    def set_processor(self, processor):
        self._processor = processor

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        split_map = {"train": "train", "val": "validation", "test": "validation"}
        hf_split = split_map[split]

        logger.info(f"  Loading scene_parse_150 {hf_split}...")
        ds = load_dataset("scene_parse_150", split=hf_split, trust_remote_code=True)

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        dataset = ADE20KDataset(ds, self._processor, img_size=self._img_size)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=segmentation_collate,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} samples, {len(loader)} batches.")
        return loader

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        return SegmentationHead(
            input_dim,
            n_classes=N_CLASSES,
            patch_size=self._patch_size,
            img_size=self._img_size,
        ).to(device)

    def get_loss_fn(self):
        ce = nn.CrossEntropyLoss(ignore_index=0)

        def loss_fn(logits: torch.Tensor, batch: Dict) -> torch.Tensor:
            labels = batch["labels"].to(logits.device)   # [B, H, W]
            return ce(logits, labels)

        return loss_fn

    def evaluate(
        self,
        model,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Compute mIoU on given dataloader.
        Temporarily sets model._output_cls_only=False so patch tokens are returned.
        """
        head.eval()

        # Ensure we get all patch tokens (not just CLS)
        _original_cls_only = None
        if hasattr(model, "_output_cls_only"):
            _original_cls_only = model._output_cls_only
            model._output_cls_only = False

        # Confusion matrix over classes 1..N_CLASSES
        n_cls = N_CLASSES + 1   # include background (index 0) for accumulation
        confusion = torch.zeros(n_cls, n_cls, dtype=torch.long)

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
                        "SegmentationTask.evaluate: features has only 1 token (CLS). "
                        "Set model output_cls_only=False for dense tasks."
                    )

                logits = head(features)   # [B, 150, H, W]
                preds = logits.argmax(dim=1).cpu()   # [B, H, W], values 0-149 → we offset +1? No.
                # logits has N_CLASSES=150 channels, argmax gives 0-149
                # But ADE20K labels are 1-150 for classes, 0 for background
                # Our head outputs class 0-149, so shift preds by +1 to match label space
                preds = preds + 1   # now 1-150

                labels = batch["labels"].cpu()   # [B, H, W], values 0-150

                mask = (labels >= 0) & (labels < n_cls) & (preds >= 0) & (preds < n_cls)
                confusion_flat = (
                    n_cls * labels[mask].long() + preds[mask].long()
                )
                confusion += torch.bincount(
                    confusion_flat, minlength=n_cls * n_cls
                ).reshape(n_cls, n_cls)

        # Restore model state
        if _original_cls_only is not None:
            model._output_cls_only = _original_cls_only

        # Compute mIoU over classes 1..N_CLASSES (skip background=0)
        iou_list = []
        for cls in range(1, n_cls):
            tp = confusion[cls, cls].item()
            fn = confusion[cls, :].sum().item() - tp
            fp = confusion[:, cls].sum().item() - tp
            denom = tp + fn + fp
            if denom > 0:
                iou_list.append(tp / denom)

        miou = sum(iou_list) / max(len(iou_list), 1)
        head.train()
        return {"miou": miou}
