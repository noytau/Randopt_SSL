"""
Causal LM model adapters for the RandOpt benchmark.

Wraps decoder-only LLMs (Qwen2.5-Instruct, Llama, etc.) from HuggingFace.
Unlike encoder SSLModels, CausalLMModels:
  - Use generate() instead of extract_features() for inference
  - Perturb ALL transformer layers (no frozen CNN front-end)
  - Use majority vote ensemble (not feature averaging)

Registered models:
  qwen2_5_0_5b  — Qwen/Qwen2.5-0.5B-Instruct  (~0.5B params)
  qwen2_5_1_5b  — Qwen/Qwen2.5-1.5B-Instruct  (~1.5B params)
  qwen2_5_3b    — Qwen/Qwen2.5-3B-Instruct     (~3B params)
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from src.interfaces import SSLModel
from src.registry import register_model

logger = logging.getLogger(__name__)


class CausalLMModel(SSLModel):
    """
    Abstract base class for decoder-only causal language models.

    Subclasses only need to set HF_MODEL_ID and override name().
    The rest of the interface is generic over any AutoModelForCausalLM.
    """

    HF_MODEL_ID: str = None

    def __init__(self, model_id: str = None, dtype: str = "float16"):
        self._model_id = model_id or self.HF_MODEL_ID
        self._dtype_str = dtype
        self._model = None
        self._tokenizer = None
        self._pretrained_state: Optional[Dict[str, torch.Tensor]] = None
        self._device: Optional[torch.device] = None

    def load(self, device: torch.device) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self._dtype_str, torch.float16)

        logger.info(f"Loading {self._model_id} (dtype={self._dtype_str})...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Decoder-only models must use left-padding so all prompts in a batch
        # end at the same position before generation begins.
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self._model.to(device)
        self._model.eval()
        self._device = device

        # Store immutable copy of pretrained weights on CPU.
        # Keeping it on GPU doubles VRAM usage (e.g. 3B FP16 = 6 GB model + 6 GB copy)
        # which causes OOM during evaluation once the KV cache grows.
        # load_state_dict() handles the CPU→GPU transfer automatically when restoring.
        self._pretrained_state = {
            k: v.detach().cpu() for k, v in self._model.state_dict().items()
        }
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info(f"  Loaded {self._model_id}. Parameters: {n_params:.1f}M")

    def get_encoder(self) -> nn.Module:
        """Return the full model (all layers are perturbed for LLMs)."""
        return self._model

    def get_tokenizer(self):
        return self._tokenizer

    def extract_features(self, batch: Any) -> torch.Tensor:
        """Not used for causal LMs — use generate() instead."""
        raise NotImplementedError(
            f"{self.__class__.__name__} is a generative model. "
            "Use generate() or GenerativeTask.evaluate() instead of extract_features()."
        )

    def hidden_dim(self) -> int:
        """Returns the model's hidden dimension (for compatibility; not used in generation)."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        cfg = self._model.config
        return getattr(cfg, "hidden_size", getattr(cfg, "d_model", 4096))

    def get_pretrained_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self._pretrained_state.items()}

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> List[str]:
        """
        Run autoregressive generation on a list of prompt strings.

        Args:
            prompts: list of raw text prompts (already formatted).
            max_new_tokens: max tokens to generate per prompt.
            temperature: sampling temperature (0.0 = greedy).
            do_sample: if True, sample; else greedy.

        Returns:
            list of generated strings (prompt NOT included in output).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        encoded = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            out = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip the prompt)
        new_tokens = out[:, prompt_len:]
        responses = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return responses


# ── Concrete Models ────────────────────────────────────────────────────────────

@register_model("qwen2_5_0_5b")
class Qwen25_0_5B(CausalLMModel):
    """Qwen2.5-0.5B-Instruct — lightest model, good for development & ablations."""
    HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    def name(self) -> str:
        return "qwen2_5_0_5b"


@register_model("qwen2_5_1_5b")
class Qwen25_1_5B(CausalLMModel):
    """Qwen2.5-1.5B-Instruct — mid-size, balance of speed and quality."""
    HF_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

    def name(self) -> str:
        return "qwen2_5_1_5b"


@register_model("qwen2_5_3b")
class Qwen25_3B(CausalLMModel):
    """Qwen2.5-3B-Instruct — primary model for paper replication experiments."""
    HF_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

    def name(self) -> str:
        return "qwen2_5_3b"
