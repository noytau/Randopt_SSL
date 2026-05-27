"""
Intent Classification Task on SUPERB IC (Intent Classification).
31 intent classes from Fluent Speech Commands.
Metric: Accuracy ↑

Model: data2vec_audio → [B, S, 1024] → mean-pool → [B, 1024] → Linear(1024, 31)
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

# SUPERB IC (Fluent Speech Commands) has 31 intent classes
DEFAULT_N_CLASSES = 31


@register_task("intent_classification")
class IntentClassificationTask(Task):
    """
    Intent Classification on SUPERB IC (Fluent Speech Commands dataset).
    31 intent classes (combinations of action, object, location).
    Primary metric: Accuracy ↑
    """

    def __init__(
        self,
        hf_dataset_id: str = "PolyAI/minds14",
        hf_config: str = "de-DE",
        n_classes: int = 14,
        audio_key: str = "audio",
        label_key: str = "intent_class",
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        batch_size: int = 16,
        num_workers: int = 2,
    ):
        self._hf_dataset_id = hf_dataset_id
        self._hf_config = hf_config
        self._n_classes = n_classes
        self._audio_key = audio_key
        self._label_key = label_key
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloaders: Dict[str, DataLoader] = {}
        self._processor = None

    def name(self) -> str:
        return "intent_classification"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("accuracy", True)

    def set_processor(self, processor):
        self._processor = processor

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        split_map = {"train": "train", "val": "validation", "test": "test"}
        hf_split = split_map.get(split, split)

        logger.info(f"  Loading {self._hf_dataset_id}/{self._hf_config} {hf_split}...")

        try:
            if self._hf_config:
                ds = load_dataset(self._hf_dataset_id, self._hf_config, split=hf_split)
            else:
                ds = load_dataset(self._hf_dataset_id, split=hf_split)
        except Exception:
            # Dataset only has "train" — carve val/test from tail
            if self._hf_config:
                full = load_dataset(self._hf_dataset_id, self._hf_config, split="train")
            else:
                full = load_dataset(self._hf_dataset_id, split="train")
            n = len(full)
            n_val  = self._max_val  if self._max_val  else max(int(n * 0.1), 1)
            n_test = self._max_test if self._max_test else max(int(n * 0.1), 1)
            if split == "train":
                ds = full.select(range(n - n_val - n_test))
            elif split == "val":
                ds = full.select(range(n - n_val - n_test, n - n_test))
            else:
                ds = full.select(range(n - n_test, n))
            logger.info(f"  Carved {split} from train: {len(ds)} samples.")

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        # SUPERB IC uses "file" as audio key (path string or audio dict)
        # Wrap in a custom dataset that can handle both formats
        dataset = _IntentDataset(ds, self._processor, self._audio_key, self._label_key)
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


# ── Internal dataset for SUPERB IC ───────────────────────────────────────────

import numpy as np
from torch.utils.data import Dataset


class _IntentDataset(Dataset):
    """
    Dataset for SUPERB IC. Handles both dict-audio and file-path audio keys.
    SUPERB IC stores audio as a dict with 'array' and 'sampling_rate' (or as file path).
    """

    def __init__(self, hf_dataset, processor, audio_key: str, label_key: str,
                 max_length_sec: float = 10.0):
        self.dataset = hf_dataset
        self.processor = processor
        self.audio_key = audio_key
        self.label_key = label_key
        self.max_length = int(max_length_sec * 16_000)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_val = item[self.audio_key]

        if isinstance(audio_val, dict):
            array = audio_val["array"]
            sampling_rate = audio_val["sampling_rate"]
        elif isinstance(audio_val, str):
            # File path
            import soundfile as sf
            array, sampling_rate = sf.read(audio_val, dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
        else:
            array = np.array(audio_val, dtype=np.float32)
            sampling_rate = 16_000

        if len(array) > self.max_length:
            array = array[: self.max_length]

        inputs = self.processor(
            array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.squeeze(0)

        label = item[self.label_key]
        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }
