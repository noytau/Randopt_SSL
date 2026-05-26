"""
ASR Task: Automatic Speech Recognition on LibriSpeech.
Uses CTC (Connectionist Temporal Classification) decoding.
Metric: Word Error Rate (WER) — lower is better.
"""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.interfaces import Task
from src.registry import register_task

logger = logging.getLogger(__name__)


# ── Vocab / tokenization ─────────────────────────────────────────────────────

# Standard 29-char CTC vocab for LibriSpeech
CTC_VOCAB = list("abcdefghijklmnopqrstuvwxyz '") + ["|", "<pad>", "<unk>"]
BLANK_ID = CTC_VOCAB.index("<pad>")  # CTC blank token
VOCAB_SIZE = len(CTC_VOCAB)
CHAR_TO_ID = {c: i for i, c in enumerate(CTC_VOCAB)}


def text_to_ids(text: str) -> list:
    """Convert transcription text to token IDs."""
    text = text.lower().strip()
    return [CHAR_TO_ID.get(c, CHAR_TO_ID["<unk>"]) for c in text]


def ids_to_text(ids: list) -> str:
    """Convert token IDs back to text (CTC greedy decode)."""
    result = []
    prev = BLANK_ID
    for idx in ids:
        if idx != prev and idx != BLANK_ID:
            if idx < len(CTC_VOCAB):
                result.append(CTC_VOCAB[idx])
        prev = idx
    text = "".join(result)
    # Replace word boundary token with space
    text = text.replace("|", " ")
    return text.strip()


# ── CTC Head ─────────────────────────────────────────────────────────────────

class CTCHead(nn.Module):
    """Simple linear projection to vocab for CTC loss."""

    def __init__(self, input_dim: int, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.proj = nn.Linear(input_dim, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, S, D] → logits: [B, S, V]"""
        return self.proj(features)


# ── Dataset ──────────────────────────────────────────────────────────────────

class LibriSpeechDataset(torch.utils.data.Dataset):
    """Wraps HuggingFace librispeech_asr dataset."""

    def __init__(self, hf_dataset, processor, max_length_sec: float = 15.0):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = int(max_length_sec * 16_000)  # 16kHz

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]["array"]

        # Truncate long audio
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]

        # Process audio
        inputs = self.processor(
            audio, sampling_rate=16_000, return_tensors="pt", padding=False
        )
        input_values = inputs.input_values.squeeze(0)

        # Encode text
        text = item["text"]
        labels = torch.tensor(text_to_ids(text), dtype=torch.long)

        return {
            "input_values": input_values,
            "labels": labels,
            "text": text,
        }


def collate_fn(batch):
    """Pad audio and labels to batch max length."""
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    texts = [item["text"] for item in batch]

    # Pad audio
    max_audio_len = max(x.shape[0] for x in input_values)
    padded_audio = torch.zeros(len(batch), max_audio_len)
    attention_mask = torch.zeros(len(batch), max_audio_len, dtype=torch.long)
    for i, x in enumerate(input_values):
        padded_audio[i, :x.shape[0]] = x
        attention_mask[i, :x.shape[0]] = 1

    # Pad labels
    max_label_len = max(x.shape[0] for x in labels)
    padded_labels = torch.full((len(batch), max_label_len), BLANK_ID, dtype=torch.long)
    label_lengths = torch.tensor([x.shape[0] for x in labels], dtype=torch.long)
    for i, x in enumerate(labels):
        padded_labels[i, :x.shape[0]] = x

    return {
        "input_values": padded_audio,
        "attention_mask": attention_mask,
        "labels": padded_labels,
        "label_lengths": label_lengths,
        "texts": texts,
    }


# ── Task ─────────────────────────────────────────────────────────────────────

@register_task("asr")
class ASRTask(Task):
    """
    Automatic Speech Recognition on LibriSpeech.
    Metric: WER (Word Error Rate) — lower is better.
    """

    def __init__(
        self,
        librispeech_subset: str = "clean",
        train_split: str = "train.100",
        max_train_samples: int = None,
        max_val_samples: int = 2000,
        max_test_samples: int = None,
        batch_size: int = 8,
        num_workers: int = 2,
    ):
        self._subset = librispeech_subset
        self._train_split = train_split
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloaders = {}
        self._processor = None

    def name(self) -> str:
        return "asr"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("wer", False)  # lower is better

    def set_processor(self, processor):
        """Set the audio processor from the model (avoids redundant loading)."""
        self._processor = processor

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        # Map split names to HF split identifiers
        split_map = {
            "train": f"train.{self._subset}" if self._train_split is None else self._train_split,
            "val": f"validation.{self._subset}",
            "test": f"test.{self._subset}",
        }
        hf_split = split_map[split]

        logger.info(f"  Loading LibriSpeech {hf_split}...")
        ds = load_dataset(
            "librispeech_asr",
            split=hf_split,
            trust_remote_code=True,
        )

        # Subsample if requested
        max_samples = {
            "train": self._max_train,
            "val": self._max_val,
            "test": self._max_test,
        }.get(split)
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))

        dataset = LibriSpeechDataset(ds, self._processor)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_fn,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} samples, {len(loader)} batches.")
        return loader

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        return CTCHead(input_dim, VOCAB_SIZE).to(device)

    def get_loss_fn(self):
        """CTC loss for ASR."""
        ctc_loss = nn.CTCLoss(blank=BLANK_ID, reduction="mean", zero_infinity=True)

        def loss_fn(logits: torch.Tensor, batch: Dict) -> torch.Tensor:
            # logits: [B, S, V] → CTC expects [S, B, V]
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            input_lengths = torch.full(
                (logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device
            )
            labels = batch["labels"].to(logits.device)
            label_lengths = batch["label_lengths"].to(logits.device)
            return ctc_loss(log_probs, labels, input_lengths, label_lengths)

        return loss_fn

    def evaluate(
        self,
        model,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Compute WER on the given dataloader."""
        from jiwer import wer as compute_wer

        head.eval()
        all_preds = []
        all_refs = []

        with torch.no_grad():
            for batch in dataloader:
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}

                # Check if model is a RandOpt ensemble
                from src.randopt.core import RandOptEnsemble
                if isinstance(model, RandOptEnsemble):
                    features = model.extract_features_ensemble(batch_dev)
                else:
                    features = model.extract_features(batch_dev)

                logits = head(features)
                pred_ids = logits.argmax(dim=-1)  # greedy CTC decode

                for i in range(pred_ids.shape[0]):
                    pred_text = ids_to_text(pred_ids[i].cpu().tolist())
                    ref_text = batch["texts"][i].lower().strip()
                    all_preds.append(pred_text if pred_text else "<empty>")
                    all_refs.append(ref_text)

        # Compute WER
        wer_score = compute_wer(all_refs, all_preds)

        head.train()
        return {"wer": wer_score}
