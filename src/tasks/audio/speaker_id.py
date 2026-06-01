"""
Speaker Identification Task on LibriSpeech.
Classifies audio into speaker IDs.
Metric: Accuracy ↑

Model: data2vec_audio → [B, S, 1024] → mean-pool → [B, 1024] → Linear(1024, n_speakers)
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.interfaces import Task
from src.registry import register_task
from src.tasks.audio.audio_utils import (
    AudioClassificationHead,
    audio_classification_collate,
    _decode_audio_item,
)

logger = logging.getLogger(__name__)

# Default: LibriSpeech clean train.100 has 251 speakers
DEFAULT_N_SPEAKERS = 251


# ── Dataset ──────────────────────────────────────────────────────────────────

class LibriSpeechSIDDataset(Dataset):
    """LibriSpeech dataset wrapper for speaker identification.

    Uses Audio(decode=False) + soundfile to avoid torchcodec dependency.
    LibriSpeech audio is 16 kHz so no resampling is needed.
    """

    def __init__(
        self,
        hf_dataset,
        processor,
        speaker_to_label: Dict[int, int],
        max_length_sec: float = 10.0,
    ):
        # Disable HF auto-decode to avoid torchcodec dependency
        try:
            from datasets import Audio
            hf_dataset = hf_dataset.cast_column("audio", Audio(decode=False))
        except Exception:
            pass
        self.dataset = hf_dataset
        self.processor = processor
        self.speaker_to_label = speaker_to_label
        self.max_length = int(max_length_sec * 16_000)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        array, sampling_rate = _decode_audio_item(item["audio"])

        # LibriSpeech is 16 kHz — resample only if something unexpected
        if sampling_rate != 16_000:
            try:
                import torchaudio.functional as TAF
                waveform = torch.from_numpy(array).unsqueeze(0)
                waveform = TAF.resample(waveform, orig_freq=sampling_rate, new_freq=16_000)
                array = waveform.squeeze(0).numpy()
            except ImportError:
                import numpy as np, scipy.signal as sps
                n = int(len(array) * 16_000 / sampling_rate)
                array = sps.resample(array, n).astype(np.float32)
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

        speaker_id = item["speaker_id"]
        label = self.speaker_to_label[speaker_id]

        return {
            "input_values": input_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ── Task ─────────────────────────────────────────────────────────────────────

@register_task("speaker_id")
class SpeakerIDTask(Task):
    """
    Speaker Identification on LibriSpeech clean.
    Maps speaker_id integers to contiguous class labels.
    n_speakers is inferred from the training split on first load.
    Primary metric: Accuracy ↑
    """

    def __init__(
        self,
        librispeech_subset: str = "clean",
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        batch_size: int = 16,
        num_workers: int = 2,
    ):
        self._subset = librispeech_subset
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloaders: Dict[str, DataLoader] = {}
        self._processor = None
        self._speaker_to_label: Dict[int, int] = {}
        self._n_speakers: int = DEFAULT_N_SPEAKERS
        self._full_train_ds = None   # cached for carving val/test from train tail

    def name(self) -> str:
        return "speaker_id"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("accuracy", True)

    def set_processor(self, processor):
        self._processor = processor

    def _build_speaker_mapping(self, ds) -> Dict[int, int]:
        """Build a speaker_id → contiguous label mapping from a dataset."""
        speaker_ids = sorted(set(ds["speaker_id"]))
        return {spk: idx for idx, spk in enumerate(speaker_ids)}

    def _get_full_train(self):
        """Load the full train.clean.100 split once; cache for carving."""
        if self._full_train_ds is not None:
            return self._full_train_ds
        from datasets import load_dataset
        logger.info(f"  Loading librispeech_asr/{self._subset} train.clean.100...")
        ds = load_dataset("librispeech_asr", split="train.clean.100")
        self._full_train_ds = ds
        return ds

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        # Carve val/test from the tail of train.clean.100 so all splits share
        # the same speaker vocabulary (train.clean.100 and validation.clean have
        # DIFFERENT speaker sets; carving avoids KeyError on unknown speaker IDs).
        full = self._get_full_train()
        n = len(full)

        n_val  = self._max_val  if self._max_val  else max(int(n * 0.05), 1)
        n_test = self._max_test if self._max_test else max(int(n * 0.05), 1)

        if split == "train":
            ds = full.select(range(n - n_val - n_test))
        elif split == "val":
            ds = full.select(range(n - n_val - n_test, n - n_test))
        else:  # test
            ds = full.select(range(n - n_test, n))
        logger.info(f"  Carved {split} from train.clean.100 tail: {len(ds)} samples.")

        # Build speaker mapping from FULL train on first call (before carving)
        if not self._speaker_to_label:
            self._speaker_to_label = self._build_speaker_mapping(full)
            self._n_speakers = len(self._speaker_to_label)
            logger.info(f"  Speaker mapping built: {self._n_speakers} unique speakers.")

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        dataset = LibriSpeechSIDDataset(
            ds,
            self._processor,
            self._speaker_to_label,
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
        return AudioClassificationHead(input_dim, self._n_speakers).to(device)

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
