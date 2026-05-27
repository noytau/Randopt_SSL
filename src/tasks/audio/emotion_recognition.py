"""
Emotion Recognition Task.
Default: PolyAI/minds14 zh-CN (14 intent classes, real audio, Parquet-native proxy).

Production note: replace with CREMA-D (6 classes) or IEMOCAP (4 classes) once the
audio files are downloaded to the cluster. Set in YAML:
  hf_dataset_id: crema_d    # or a local HF dataset loaded with load_from_disk
  n_classes: 6
  label_key: label

Metric: Accuracy ↑ + weighted F1
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

# CREMA-D has 6 emotion classes:
# 0=anger, 1=disgust, 2=fear, 3=happiness, 4=neutral, 5=sadness
DEFAULT_N_CLASSES = 6


@register_task("emotion_recognition")
class EmotionRecognitionTask(Task):
    """
    Emotion Recognition on CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset).
    6 classes: anger, disgust, fear, happiness, neutral, sadness.
    Primary metric: Accuracy ↑ (also reports weighted F1).
    """

    def __init__(
        self,
        hf_dataset_id: str = "PolyAI/minds14",
        hf_config: str = "zh-CN",
        n_classes: int = 14,
        audio_key: str = "audio",
        label_key: str = "intent_class",
        max_train_samples: int = None,
        max_val_samples: int = 500,
        max_test_samples: int = 500,
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
        return "emotion_recognition"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("accuracy", True)

    def set_processor(self, processor):
        self._processor = processor

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        logger.info(f"  Loading {self._hf_dataset_id}/{self._hf_config} for emotion recognition...")

        split_map = {"train": "train", "val": "validation", "test": "test"}
        hf_split = split_map.get(split, split)

        try:
            if self._hf_config:
                ds = load_dataset(self._hf_dataset_id, self._hf_config, split=hf_split)
            else:
                ds = load_dataset(self._hf_dataset_id, split=hf_split)
        except Exception:
            # Dataset only has "train" (e.g., CREMA-D, minds14) — carve val/test from tail
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
        from sklearn.metrics import f1_score

        head.eval()
        all_preds = []
        all_labels = []

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
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].tolist())

        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / max(len(all_labels), 1)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        head.train()
        return {"accuracy": accuracy, "f1_weighted": f1}
