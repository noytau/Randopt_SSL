"""
Shared audio utilities for classification tasks.

Provides:
  - AudioClassificationHead: mean-pool + linear for [B, S, D] features
  - AudioClassificationDataset: generic HF dataset wrapper for audio classification
  - audio_classification_collate: pads input_values and builds attention_mask
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ── Head ──────────────────────────────────────────────────────────────────────

class AudioClassificationHead(nn.Module):
    """
    Mean-pool [B, S, D] → [B, D] → dropout → Linear(D, n_classes).
    Also handles [B, D] inputs (no pooling needed).
    """

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, S, D] or [B, D] → logits: [B, n_classes]"""
        if features.dim() == 3:
            x = features.mean(dim=1)   # mean-pool over time → [B, D]
        else:
            x = features               # already [B, D]
        return self.classifier(self.dropout(x))


# ── Dataset ──────────────────────────────────────────────────────────────────

class AudioClassificationDataset(Dataset):
    """
    Generic wrapper around a HuggingFace audio classification dataset.
    Handles audio extraction, truncation, and processor calls.
    """

    def __init__(
        self,
        hf_dataset,
        processor,
        label_key: str = "label",
        audio_key: str = "audio",
        max_length_sec: float = 10.0,
    ):
        self.dataset = hf_dataset
        self.processor = processor
        self.label_key = label_key
        self.audio_key = audio_key
        self.max_length = int(max_length_sec * 16_000)  # assume 16kHz

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_dict = item[self.audio_key]
        array = audio_dict["array"]
        sampling_rate = audio_dict["sampling_rate"]

        # Truncate to max length
        if len(array) > self.max_length:
            array = array[: self.max_length]

        inputs = self.processor(
            array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.squeeze(0)   # [T]

        label = item[self.label_key]
        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ── Collate ───────────────────────────────────────────────────────────────────

def audio_classification_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pad input_values to batch-max length.
    Build attention_mask: 1 where audio exists, 0 for padding.
    Stack labels as-is.
    """
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len = max(x.shape[0] for x in input_values)
    padded = torch.zeros(len(batch), max_len)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, x in enumerate(input_values):
        padded[i, : x.shape[0]] = x
        attention_mask[i, : x.shape[0]] = 1

    return {
        "input_values": padded,
        "attention_mask": attention_mask,
        "labels": torch.stack(labels),
    }
