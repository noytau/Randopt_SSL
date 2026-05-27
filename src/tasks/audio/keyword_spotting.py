"""
Keyword Spotting Task on Google Speech Commands v0.02.
35 keyword classes.
Metric: Accuracy ↑

Model: data2vec_audio → [B, S, 1024] → mean-pool → [B, 1024] → Linear(1024, 35)
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

N_CLASSES = 35


@register_task("keyword_spotting")
class KeywordSpottingTask(Task):
    """
    Keyword Spotting on Google Speech Commands v0.02.
    35 command classes (yes, no, up, down, left, right, on, off, stop, go, ...).
    Primary metric: Accuracy ↑
    """

    def __init__(
        self,
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        batch_size: int = 16,
        num_workers: int = 2,
    ):
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloaders: Dict[str, DataLoader] = {}
        self._processor = None

    def name(self) -> str:
        return "keyword_spotting"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("accuracy", True)

    def set_processor(self, processor):
        self._processor = processor

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        split_map = {"train": "train", "val": "validation", "test": "test"}
        hf_split = split_map[split]

        logger.info(f"  Loading speech_commands v0.02 {hf_split}...")
        ds = load_dataset(
            "speech_commands",
            "v0.02",
            split=hf_split,
            trust_remote_code=True,
        )

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        dataset = AudioClassificationDataset(
            ds,
            self._processor,
            label_key="label",
            audio_key="audio",
            max_length_sec=1.0,   # speech commands are ~1s clips
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
        logger.info(f"  Loaded {len(dataset)} samples, {len(loader)} batches.")
        return loader

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        return AudioClassificationHead(input_dim, N_CLASSES).to(device)

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
