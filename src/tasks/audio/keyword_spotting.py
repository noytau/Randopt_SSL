"""
Keyword Spotting Task.
Default: PolyAI/minds14 en-US (14 banking-intent classes, real audio, Parquet-native).

Production note: swap hf_dataset_id to "google/speech_commands" with hf_config="v0.02"
and n_classes=35 once a Parquet-native Speech Commands dataset is available on the hub.

Metric: Accuracy ↑
Model: data2vec_audio → [B, S, 1024] → mean-pool → [B, 1024] → Linear(1024, n_classes)
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.interfaces import Task
from src.registry import register_task
from src.tasks.audio.audio_utils import (
    AudioClassificationHead,
    AudioClassificationDataset,
    audio_classification_collate,
)

logger = logging.getLogger(__name__)

# Default: minds14 en-US — 14 banking-intent classes, fully Parquet-native on HF Hub.
# Only "train" split exists; we carve val/test from the tail of train.
DEFAULT_HF_DATASET = "PolyAI/minds14"
DEFAULT_HF_CONFIG   = "en-US"
DEFAULT_N_CLASSES   = 14
DEFAULT_LABEL_KEY   = "intent_class"
DEFAULT_AUDIO_KEY   = "audio"


@register_task("keyword_spotting")
class KeywordSpottingTask(Task):
    """
    Keyword / Command Spotting task.
    Default dataset: PolyAI/minds14 en-US (14 banking-intent classes).
    minds14 has only a "train" split; val/test are carved from the tail.

    To use Speech Commands v0.02 (35 classes) once a Parquet version is available:
      hf_dataset_id: google/speech_commands
      hf_config:     v0.02
      n_classes:     35
      label_key:     label
    """

    def __init__(
        self,
        hf_dataset_id: str = DEFAULT_HF_DATASET,
        hf_config: str = DEFAULT_HF_CONFIG,
        n_classes: int = DEFAULT_N_CLASSES,
        label_key: str = DEFAULT_LABEL_KEY,
        audio_key: str = DEFAULT_AUDIO_KEY,
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        batch_size: int = 16,
        num_workers: int = 2,
    ):
        self._hf_dataset_id = hf_dataset_id
        self._hf_config     = hf_config
        self._n_classes     = n_classes
        self._label_key     = label_key
        self._audio_key     = audio_key
        self._max_train     = max_train_samples
        self._max_val       = max_val_samples
        self._max_test      = max_test_samples
        self._batch_size    = batch_size
        self._num_workers   = num_workers
        self._dataloaders: Dict[str, DataLoader] = {}
        self._processor = None
        self._full_train_ds = None   # cached for split carving

    def name(self) -> str:
        return "keyword_spotting"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("accuracy", True)

    def set_processor(self, processor):
        self._processor = processor

    def _load_full_train(self):
        """Load the full training split once; cache for reuse."""
        if self._full_train_ds is not None:
            return self._full_train_ds
        from datasets import load_dataset
        logger.info(f"  Loading {self._hf_dataset_id}/{self._hf_config} train...")
        if self._hf_config:
            ds = load_dataset(self._hf_dataset_id, self._hf_config, split="train")
        else:
            ds = load_dataset(self._hf_dataset_id, split="train")
        self._full_train_ds = ds
        return ds

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        # Try dedicated splits first; fall back to carving from train
        split_map = {"train": "train", "val": "validation", "test": "test"}
        hf_split = split_map.get(split, split)

        try:
            if self._hf_config:
                ds = load_dataset(self._hf_dataset_id, self._hf_config, split=hf_split)
            else:
                ds = load_dataset(self._hf_dataset_id, split=hf_split)
            logger.info(f"  Loaded {self._hf_dataset_id} {hf_split}: {len(ds)} samples.")
        except Exception:
            # Dataset only has "train" (e.g., minds14) — carve val/test from tail
            full = self._load_full_train()
            n = len(full)
            n_val  = self._max_val  if self._max_val  else max(int(n * 0.1), 1)
            n_test = self._max_test if self._max_test else max(int(n * 0.1), 1)
            if split == "train":
                ds = full.select(range(n - n_val - n_test))
            elif split == "val":
                ds = full.select(range(n - n_val - n_test, n - n_test))
            else:  # test
                ds = full.select(range(n - n_test, n))
            logger.info(f"  Carved {split} from train tail: {len(ds)} samples.")

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        dataset = AudioClassificationDataset(
            ds,
            self._processor,
            label_key=self._label_key,
            audio_key=self._audio_key,
            max_length_sec=10.0,
        )
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=audio_classification_collate,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self._dataloaders[split] = loader
        logger.info(f"  {split}: {len(dataset)} samples, {len(loader)} batches.")
        return loader

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        return AudioClassificationHead(input_dim, self._n_classes).to(device)

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
                batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
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
