"""
SQuAD v2 Question Answering Task for BERT-Large.
Extractive QA: predict start/end token positions (span extraction).
Unanswerable questions: start=end=0 (CLS position).
Metric: F1 ↑

IMPORTANT: BERT model must be configured with output_cls_only=False to get the
  full sequence [B, L, 1024]. Set model_config: {output_cls_only: false} in YAML.
  The evaluate() method temporarily forces model._output_cls_only=False.

Batch format:
  {"input_ids": [B, L], "attention_mask": [B, L], "token_type_ids": [B, L],
   "start_positions": [B], "end_positions": [B]}
"""

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.interfaces import Task
from src.registry import register_task

logger = logging.getLogger(__name__)


# ── Head ──────────────────────────────────────────────────────────────────────

class SpanExtractionHead(nn.Module):
    """
    Takes [B, L, D] full-sequence features.
    Applies two separate linears to produce start_logits [B, L] and end_logits [B, L].
    Returns stacked tensor [B, 2, L].
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.start_linear = nn.Linear(input_dim, 1)
        self.end_linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: [B, L, D]
        Returns: [B, 2, L]  — dim 1: 0=start, 1=end
        """
        start_logits = self.start_linear(features).squeeze(-1)   # [B, L]
        end_logits = self.end_linear(features).squeeze(-1)        # [B, L]
        return torch.stack([start_logits, end_logits], dim=1)    # [B, 2, L]


# ── Dataset ──────────────────────────────────────────────────────────────────

class SQuADDataset(Dataset):
    """SQuAD v2 dataset wrapper. Handles tokenization and span extraction."""

    def __init__(self, hf_dataset, tokenizer, max_length: int = 384, stride: int = 128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self._processed = self._process_all()

    def _process_all(self) -> List[Dict]:
        processed = []
        for item in self.dataset:
            context = item["context"]
            question = item["question"]
            answers = item["answers"]

            enc = self.tokenizer(
                question,
                context,
                max_length=self.max_length,
                truncation="only_second",
                stride=self.stride,
                padding="max_length",
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).squeeze(0)
            offset_mapping = enc["offset_mapping"].squeeze(0)

            # Determine start/end positions
            if len(answers["text"]) == 0:
                # Unanswerable: CLS position (index 0)
                start_position = torch.tensor(0, dtype=torch.long)
                end_position = torch.tensor(0, dtype=torch.long)
            else:
                answer_text = answers["text"][0]
                answer_start_char = answers["answer_start"][0]
                answer_end_char = answer_start_char + len(answer_text)

                # Find token positions corresponding to answer span
                start_pos, end_pos = 0, 0
                sequence_ids = enc.sequence_ids(0)

                # Find context range in token sequence (sequence_id=1 means context)
                context_start_tok = None
                context_end_tok = None
                for i, sid in enumerate(sequence_ids):
                    if sid == 1:
                        if context_start_tok is None:
                            context_start_tok = i
                        context_end_tok = i

                if context_start_tok is None:
                    start_pos, end_pos = 0, 0
                else:
                    # Find start token
                    start_pos = context_start_tok
                    for i in range(context_start_tok, (context_end_tok or context_start_tok) + 1):
                        if i < len(offset_mapping):
                            tok_start, tok_end = offset_mapping[i].tolist()
                            if tok_start <= answer_start_char:
                                start_pos = i
                            if tok_end < answer_end_char:
                                end_pos = i
                            if tok_start > answer_end_char:
                                break

                    # Clamp to sequence length
                    start_pos = min(start_pos, self.max_length - 1)
                    end_pos = min(max(end_pos, start_pos), self.max_length - 1)

                start_position = torch.tensor(start_pos, dtype=torch.long)
                end_position = torch.tensor(end_pos, dtype=torch.long)

            processed.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "start_positions": start_position,
                "end_positions": end_position,
                # Keep answer text for eval
                "answer_texts": answers["text"],
                "context": context,
            })

        return processed

    def __len__(self) -> int:
        return len(self._processed)

    def __getitem__(self, idx):
        return self._processed[idx]


def squad_collate(batch: List[Dict]) -> Dict:
    keys_tensor = ["input_ids", "attention_mask", "token_type_ids",
                   "start_positions", "end_positions"]
    result = {k: torch.stack([item[k] for item in batch]) for k in keys_tensor}
    result["answer_texts"] = [item["answer_texts"] for item in batch]
    result["contexts"] = [item["context"] for item in batch]
    return result


# ── Task ─────────────────────────────────────────────────────────────────────

@register_task("squad")
class SQuADTask(Task):
    """
    Extractive Question Answering on SQuAD v2 (with unanswerable questions).
    Primary metric: F1 ↑

    IMPORTANT: BERT model must be configured with output_cls_only=False.
    Set in YAML: model_config: {output_cls_only: false}
    The evaluate() method temporarily forces model._output_cls_only=False.
    """

    def __init__(
        self,
        max_length: int = 384,
        stride: int = 128,
        max_train_samples: int = None,
        max_val_samples: int = None,
        max_test_samples: int = None,
        batch_size: int = 16,
        num_workers: int = 2,
    ):
        self._max_length = max_length
        self._stride = stride
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloaders: Dict[str, DataLoader] = {}
        self._tokenizer = None

    def name(self) -> str:
        return "squad"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("f1", True)

    def set_processor(self, processor):
        self._tokenizer = processor

    def set_tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        split_map = {"train": "train", "val": "validation", "test": "validation"}
        hf_split = split_map[split]

        logger.info(f"  Loading squad_v2 {hf_split}...")
        ds = load_dataset("squad_v2", split=hf_split, )

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        logger.info(f"  Tokenizing {len(ds)} examples (max_length={self._max_length})...")
        dataset = SQuADDataset(ds, self._tokenizer, self._max_length, self._stride)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=squad_collate,
            num_workers=self._num_workers,
            pin_memory=True,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} examples, {len(loader)} batches.")
        return loader

    def build_head(self, input_dim: int, device: torch.device) -> nn.Module:
        return SpanExtractionHead(input_dim).to(device)

    def get_loss_fn(self):
        ce = nn.CrossEntropyLoss()

        def loss_fn(logits: torch.Tensor, batch: Dict) -> torch.Tensor:
            # logits: [B, 2, L]
            start_logits = logits[:, 0, :]   # [B, L]
            end_logits = logits[:, 1, :]      # [B, L]
            start_positions = batch["start_positions"].to(logits.device)
            end_positions = batch["end_positions"].to(logits.device)
            loss_start = ce(start_logits, start_positions)
            loss_end = ce(end_logits, end_positions)
            return (loss_start + loss_end) / 2

        return loss_fn

    def evaluate(
        self,
        model,
        head: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        Compute token-level F1 between predicted spans and gold answers.
        Temporarily sets model._output_cls_only=False to get full sequence features.
        """
        head.eval()

        _original_cls_only = None
        if hasattr(model, "_output_cls_only"):
            _original_cls_only = model._output_cls_only
            model._output_cls_only = False

        all_f1 = []
        all_em = []

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

                logits = head(features)   # [B, 2, L]
                start_logits = logits[:, 0, :]   # [B, L]
                end_logits = logits[:, 1, :]      # [B, L]

                pred_starts = start_logits.argmax(dim=-1).cpu().tolist()
                pred_ends = end_logits.argmax(dim=-1).cpu().tolist()
                gold_starts = batch["start_positions"].tolist()
                gold_ends = batch["end_positions"].tolist()
                answer_texts_batch = batch["answer_texts"]

                for i in range(len(pred_starts)):
                    ps, pe = pred_starts[i], pred_ends[i]
                    gs, ge = gold_starts[i], gold_ends[i]
                    gold_answers = answer_texts_batch[i]

                    # Token-level F1 (simplified: compare predicted span tokens with gold span tokens)
                    pred_tokens = set(range(min(ps, pe), max(ps, pe) + 1))
                    gold_tokens = set(range(min(gs, ge), max(gs, ge) + 1))

                    # Handle unanswerable: gold_start=gold_end=0
                    is_unanswerable = (gs == 0 and ge == 0) or (len(gold_answers) == 0)
                    is_pred_no_answer = (ps == 0 and pe == 0)

                    if is_unanswerable:
                        f1 = 1.0 if is_pred_no_answer else 0.0
                        em = 1.0 if is_pred_no_answer else 0.0
                    else:
                        if is_pred_no_answer:
                            f1, em = 0.0, 0.0
                        else:
                            intersection = len(pred_tokens & gold_tokens)
                            precision = intersection / max(len(pred_tokens), 1)
                            recall = intersection / max(len(gold_tokens), 1)
                            f1 = (2 * precision * recall / max(precision + recall, 1e-8))
                            em = 1.0 if pred_tokens == gold_tokens else 0.0

                    all_f1.append(f1)
                    all_em.append(em)

        if _original_cls_only is not None:
            model._output_cls_only = _original_cls_only

        head.train()
        return {
            "f1": sum(all_f1) / max(len(all_f1), 1),
            "exact_match": sum(all_em) / max(len(all_em), 1),
        }
