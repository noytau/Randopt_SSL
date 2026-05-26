"""
GLUE NLP Tasks for BERT-Large: RTE, MNLI, CoLA.

All three tasks share the same skeleton but differ in:
  - Dataset name / config
  - Number of classes
  - Primary metric (accuracy or MCC)
  - Sentence pair vs. single sentence input

Batch format:
  {"input_ids": [B, L], "attention_mask": [B, L], "token_type_ids": [B, L], "labels": [B]}

BERT extract_features returns [B, 1, 1024] (CLS only).
Task heads pool that to [B, 1024] then project to n_classes.
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.interfaces import Task
from src.registry import register_task

logger = logging.getLogger(__name__)


# ── Shared head ───────────────────────────────────────────────────────────────

class CLSClassificationHead(nn.Module):
    """
    Takes [B, 1, D] CLS features → pools to [B, D] → linear → [B, n_classes].
    """

    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.pool = lambda x: x[:, 0, :]   # take first (CLS) token
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, 1, D] or [B, D] → logits: [B, n_classes]"""
        if features.dim() == 3:
            x = features[:, 0, :]   # [B, D]
        else:
            x = features             # already [B, D]
        return self.classifier(self.dropout(x))


# ── Shared dataset ────────────────────────────────────────────────────────────

class GLUEDataset(torch.utils.data.Dataset):
    """Generic GLUE dataset wrapper."""

    def __init__(self, hf_dataset, tokenizer, max_length: int, label_key: str,
                 sent1_key: str, sent2_key: str = None):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_key = label_key
        self.sent1_key = sent1_key
        self.sent2_key = sent2_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text_a = item[self.sent1_key]
        text_b = item[self.sent2_key] if self.sent2_key else None

        enc = self.tokenizer(
            text_a, text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = item[self.label_key]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def glue_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}


# ── Shared base class ─────────────────────────────────────────────────────────

class GLUETask(Task):
    """
    Base class for GLUE tasks.
    Subclasses set: _task_name, _glue_config, _n_classes, _label_key,
                    _sent1_key, _sent2_key, _primary_metric, _higher_is_better.
    """

    _task_name: str = None
    _glue_config: str = None
    _n_classes: int = 2
    _label_key: str = "label"
    _sent1_key: str = "sentence1"
    _sent2_key: str = None
    _primary_metric: str = "accuracy"
    _higher_is_better: bool = True

    def __init__(
        self,
        max_length: int = 128,
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        self._max_length = max_length
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloaders: Dict[str, DataLoader] = {}
        self._tokenizer = None

    def name(self) -> str:
        return self._task_name

    def primary_metric(self) -> Tuple[str, bool]:
        return (self._primary_metric, self._higher_is_better)

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    # also support set_processor for uniform interface with run.py
    def set_processor(self, processor):
        self._tokenizer = processor

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        # GLUE validation split is "validation", not "val"
        hf_split = {"train": "train", "val": "validation", "test": "validation"}.get(split, split)

        logger.info(f"  Loading GLUE/{self._glue_config} {hf_split}...")
        ds = load_dataset("glue", self._glue_config, split=hf_split, trust_remote_code=True)

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        dataset = GLUEDataset(
            ds, self._tokenizer, self._max_length,
            self._label_key, self._sent1_key, self._sent2_key,
        )
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=glue_collate,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} samples, {len(loader)} batches.")
        return loader

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        return CLSClassificationHead(input_dim, self._n_classes).to(device)

    def get_loss_fn(self):
        ce = nn.CrossEntropyLoss()

        def loss_fn(logits: torch.Tensor, batch: Dict) -> torch.Tensor:
            labels = batch["labels"].to(logits.device)
            return ce(logits, labels)

        return loss_fn

    def evaluate(
        self,
        model,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        head.eval()
        all_preds, all_labels = [], []

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
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].tolist())

        result = self._compute_metrics(all_preds, all_labels)
        head.train()
        return result

    def _compute_metrics(self, preds: List[int], labels: List[int]) -> Dict[str, float]:
        """Override in subclasses for non-accuracy metrics."""
        correct = sum(p == l for p, l in zip(preds, labels))
        accuracy = correct / max(len(labels), 1)
        return {"accuracy": accuracy}


# ── RTE Task ──────────────────────────────────────────────────────────────────

@register_task("rte")
class RTETask(GLUETask):
    """
    Recognizing Textual Entailment (RTE) from GLUE.
    Binary: entailment (0) vs. not-entailment (1).
    2,490 train pairs, 277 val pairs.
    Primary metric: Accuracy ↑
    Headline result: stable across methods (CF anchor).
    """
    _task_name = "rte"
    _glue_config = "rte"
    _n_classes = 2
    _label_key = "label"
    _sent1_key = "sentence1"
    _sent2_key = "sentence2"
    _primary_metric = "accuracy"
    _higher_is_better = True


# ── MNLI Task ─────────────────────────────────────────────────────────────────

@register_task("mnli")
class MNLITask(GLUETask):
    """
    Multi-Genre Natural Language Inference (MNLI) from GLUE.
    3-class: entailment (0), neutral (1), contradiction (2).
    393K train pairs, 9.8K val pairs (matched).
    Primary metric: Accuracy ↑  (CF anchor — large dataset)
    """
    _task_name = "mnli"
    _glue_config = "mnli"
    _n_classes = 3
    _label_key = "label"
    _sent1_key = "premise"
    _sent2_key = "hypothesis"
    _primary_metric = "accuracy"
    _higher_is_better = True

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        # MNLI has "validation_matched" and "validation_mismatched"
        hf_split = {
            "train": "train",
            "val": "validation_matched",
            "test": "validation_matched",
        }.get(split, split)

        logger.info(f"  Loading GLUE/mnli {hf_split}...")
        ds = load_dataset("glue", "mnli", split=hf_split, trust_remote_code=True)

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        dataset = GLUEDataset(
            ds, self._tokenizer, self._max_length,
            self._label_key, self._sent1_key, self._sent2_key,
        )
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=glue_collate,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} samples, {len(loader)} batches.")
        return loader


# ── CoLA Task ─────────────────────────────────────────────────────────────────

@register_task("cola")
class CoLATask(GLUETask):
    """
    Corpus of Linguistic Acceptability (CoLA) from GLUE.
    Binary: grammatically acceptable (1) vs. not (0).
    8,551 train sentences, 1,043 val sentences.
    Primary metric: Matthews Correlation Coefficient (MCC) ↑
    MCC is NON-DIFFERENTIABLE — RandOpt structural advantage!
    """
    _task_name = "cola"
    _glue_config = "cola"
    _n_classes = 2
    _label_key = "label"
    _sent1_key = "sentence"
    _sent2_key = None
    _primary_metric = "mcc"
    _higher_is_better = True

    def _compute_metrics(self, preds: List[int], labels: List[int]) -> Dict[str, float]:
        """Matthews Correlation Coefficient — non-differentiable, RandOpt's structural advantage."""
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(labels, preds) if len(set(labels)) > 1 else 0.0
        accuracy = sum(p == l for p, l in zip(preds, labels)) / max(len(labels), 1)
        return {"mcc": mcc, "accuracy": accuracy}
