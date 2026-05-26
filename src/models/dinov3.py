"""
DINOv2 ViT-L/14 model adapter.

Note: "DINOv3" is the project name; the actual HuggingFace model is DINOv2
(facebook/dinov2-large), which is the latest publicly available DINO variant.
If facebook/dinov3-vitl14 becomes available, swap the HF_MODEL_ID constant.

SSL pretraining: Self-distillation with no labels (DINO) + iBOT objective.
Output: [B, num_patches, 1024] patch features + [B, 1024] CLS token.
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from src.interfaces import SSLModel
from src.registry import register_model

logger = logging.getLogger(__name__)


@register_model("dinov2")
class DINOv2Model(SSLModel):
    """
    DINOv2 ViT-Large/14 (~307M params).
    SSL: self-distillation with masked patch prediction (iBOT).
    Output: [B, num_patches+1, 1024] — first token is CLS.

    For dense tasks (segmentation, detection): use all patch tokens.
    For classification tasks: use CLS token only → [B, 1, 1024].
    """

    HF_MODEL_ID = "facebook/dinov2-large"

    def __init__(self, model_id: str = None, output_cls_only: bool = True):
        self._model_id = model_id or self.HF_MODEL_ID
        self._output_cls_only = output_cls_only
        self._model = None
        self._processor = None
        self._pretrained_state = None
        self._device = None

    def name(self) -> str:
        return "dinov2"

    def load(self, device: torch.device) -> None:
        from transformers import Dinov2Model as HFModel, AutoImageProcessor

        logger.info(f"Loading {self._model_id}...")
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = HFModel.from_pretrained(self._model_id)
        self._model.to(device)
        self._model.eval()
        self._device = device

        # Store immutable pretrained weights
        self._pretrained_state = {
            k: v.clone().to(device) for k, v in self._model.state_dict().items()
        }
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(f"  Loaded. Parameters: {n_params:.1f}M")

    def get_encoder(self) -> nn.Module:
        return self._model

    def get_processor(self):
        return self._processor

    def extract_features(self, batch: Any) -> torch.Tensor:
        """
        Extract vision features from image batch.

        batch expected keys:
          - "pixel_values": [B, 3, H, W] preprocessed images

        Returns:
          - [B, 1, 1024]        if output_cls_only=True   (CLS token)
          - [B, N+1, 1024]      if output_cls_only=False  (CLS + patch tokens)
        """
        pixel_values = batch["pixel_values"].to(self._device)

        with torch.no_grad():
            outputs = self._model(
                pixel_values=pixel_values,
                output_hidden_states=False,
            )

        # last_hidden_state: [B, N+1, 1024] where index 0 = CLS
        if self._output_cls_only:
            return outputs.last_hidden_state[:, :1, :]   # [B, 1, 1024]
        else:
            return outputs.last_hidden_state              # [B, N+1, 1024]

    def hidden_dim(self) -> int:
        return 1024  # DINOv2 ViT-Large

    def get_pretrained_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._pretrained_state.items()}
