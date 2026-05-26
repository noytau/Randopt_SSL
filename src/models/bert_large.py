"""
BERT-Large model adapter.
Wraps HuggingFace bert-large-uncased for the RandOpt benchmark.

SSL pretraining: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP).
Output: [B, seq_len, 1024] from Transformer encoder (last_hidden_state).
       or pooled [B, 1024] via CLS token.
"""

import logging
from copy import deepcopy
from typing import Any, Dict

import torch
import torch.nn as nn

from src.interfaces import SSLModel
from src.registry import register_model

logger = logging.getLogger(__name__)


@register_model("bert_large")
class BERTLargeModel(SSLModel):
    """
    BERT-Large Uncased (~340M params).
    SSL: Masked Language Modeling + Next Sentence Prediction on BooksCorpus + Wikipedia.
    Output: [B, seq_len, 1024] full sequence, or [B, 1024] CLS-pooled.

    For classification tasks we return the CLS token: [B, 1, 1024] → tasks pool to [B, 1024].
    """

    HF_MODEL_ID = "bert-large-uncased"

    def __init__(self, model_id: str = None, output_cls_only: bool = True):
        self._model_id = model_id or self.HF_MODEL_ID
        self._output_cls_only = output_cls_only
        self._model = None
        self._tokenizer = None
        self._pretrained_state = None
        self._device = None

    def name(self) -> str:
        return "bert_large"

    def load(self, device: torch.device) -> None:
        from transformers import BertModel, BertTokenizerFast

        logger.info(f"Loading {self._model_id}...")
        self._tokenizer = BertTokenizerFast.from_pretrained(self._model_id)
        self._model = BertModel.from_pretrained(self._model_id)
        self._model.to(device)
        self._model.eval()
        self._device = device

        # Store immutable copy of pretrained weights
        self._pretrained_state = {
            k: v.clone().to(device) for k, v in self._model.state_dict().items()
        }
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(f"  Loaded. Parameters: {n_params:.1f}M")

    def get_encoder(self) -> nn.Module:
        return self._model

    def get_tokenizer(self):
        return self._tokenizer

    def extract_features(self, batch: Any) -> torch.Tensor:
        """
        Extract encoder features from text batch.

        batch expected keys:
          - "input_ids":      [B, L] token IDs
          - "attention_mask": [B, L] 1/0 mask
          - "token_type_ids": [B, L] optional segment IDs

        Returns:
          - [B, 1, 1024] if output_cls_only=True  (CLS token)
          - [B, L, 1024] if output_cls_only=False (full sequence)
        """
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self._device)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
        )

        if self._output_cls_only:
            # CLS token: [B, 1024] → reshape to [B, 1, 1024] for uniform interface
            return outputs.last_hidden_state[:, :1, :]   # [B, 1, 1024]
        else:
            return outputs.last_hidden_state  # [B, L, 1024]

    def hidden_dim(self) -> int:
        return 1024  # BERT-Large

    def get_pretrained_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._pretrained_state.items()}
