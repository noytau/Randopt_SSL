"""
Data2Vec Audio Large model adapter.
Wraps HuggingFace facebook/data2vec-audio-large for the benchmark.
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from src.interfaces import SSLModel
from src.registry import register_model

logger = logging.getLogger(__name__)


@register_model("data2vec_audio")
class Data2VecAudioModel(SSLModel):
    """
    data2vec-audio Large (~315M params).
    SSL pretraining: masked latent prediction (self-distillation) on LibriSpeech.
    Output: [batch, seq_len, 1024] from the Transformer encoder.
    """

    HF_MODEL_ID = "facebook/data2vec-audio-large"

    def __init__(self, model_id: str = None):
        self._model_id = model_id or self.HF_MODEL_ID
        self._model = None
        self._processor = None
        self._pretrained_state = None
        self._device = None

    def name(self) -> str:
        return "data2vec_audio"

    def load(self, device: torch.device) -> None:
        from transformers import Data2VecAudioModel as HFModel, Wav2Vec2Processor

        logger.info(f"Loading {self._model_id}...")
        self._processor = Wav2Vec2Processor.from_pretrained(self._model_id)
        self._model = HFModel.from_pretrained(self._model_id)
        self._model.to(device)
        self._model.eval()
        self._device = device

        # Store a copy of pretrained weights (immutable reference for RandOpt)
        self._pretrained_state = {
            k: v.clone().to(device) for k, v in self._model.state_dict().items()
        }
        logger.info(
            f"  Loaded. Parameters: {sum(p.numel() for p in self._model.parameters()) / 1e6:.1f}M"
        )

    def get_encoder(self) -> nn.Module:
        return self._model

    def get_processor(self):
        return self._processor

    def extract_features(self, batch: Any) -> torch.Tensor:
        """
        Extract encoder features from audio batch.
        
        batch expected keys:
          - "input_values": [B, T] raw waveform tensor
          - "attention_mask": [B, T] optional
          
        Returns: [B, S, 1024] where S is the downsampled sequence length.
        """
        input_values = batch["input_values"].to(self._device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        with torch.no_grad():
            outputs = self._model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
        return outputs.last_hidden_state  # [B, S, 1024]

    def hidden_dim(self) -> int:
        return 1024  # data2vec-audio Large

    def get_pretrained_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._pretrained_state.items()}
