"""
Supervised Fine-Tuning (SFT) baseline for LLM generative tasks.

This is the gradient-based counterpart to RandOpt. It fine-tunes the full
LLM on (prompt, gold_answer) pairs using standard cross-entropy loss — the
same data RandOpt uses as a fitness oracle, but with gradient updates.

Training setup:
  - Causal LM objective: next-token prediction on the full (prompt + answer) sequence
  - Loss masked to answer tokens only (prompt tokens get label=-100, ignored by CE)
  - Mixed precision (bfloat16 if available, else float16)
  - Gradient checkpointing to reduce peak memory

Config keys consumed:
  ft_epochs    : int   (default 3)   — full passes over train set
  lr_encoder   : float (default 2e-5) — LLM fine-tuning LR (much lower than encoder tasks)
  batch_size   : int   (default 4)
  grad_accum   : int   (default 4)   — effective batch = batch_size × grad_accum
  max_grad_norm: float (default 1.0)

Why this is the right comparison for RandOpt:
  Both methods see the same train data and the same number of forward passes (roughly).
  SFT uses those passes to compute gradients and update weights.
  RandOpt uses those passes to score random perturbations and select the best K.
  The question: can you learn a task without ever computing a gradient?
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.interfaces import AdaptationMethod, SSLModel, Task
from src.registry import register_method

logger = logging.getLogger(__name__)


@register_method("sft")
class SupervisedFineTune(AdaptationMethod):
    """
    Standard supervised fine-tuning of a CausalLMModel on task train data.
    Gradient-based baseline for generative LLM tasks.
    """

    def name(self) -> str:
        return "sft"

    def adapt(
        self,
        model: SSLModel,
        task: Task,
        device: torch.device,
        config: Dict[str, Any],
    ) -> Tuple[nn.Module, nn.Module]:
        from src.models.causal_lm import CausalLMModel
        from src.tasks.llm.base import GenerativeTask

        if not isinstance(model, CausalLMModel):
            raise TypeError(f"SFT baseline requires CausalLMModel, got {type(model).__name__}")
        if not isinstance(task, GenerativeTask):
            raise TypeError(f"SFT baseline requires GenerativeTask, got {type(task).__name__}")

        # ── Config ──
        n_epochs    = config.get("ft_epochs", 3)
        lr          = config.get("lr_encoder", 2e-5)
        grad_accum  = config.get("grad_accum", 4)
        max_norm    = config.get("max_grad_norm", 1.0)
        batch_size  = config.get("task_config", {}).get("batch_size", 4)

        logger.info(
            f"SFT: epochs={n_epochs}, lr={lr}, grad_accum={grad_accum}, "
            f"effective_batch={batch_size * grad_accum}"
        )

        lm = model.get_encoder()           # the actual nn.Module (AutoModelForCausalLM)
        tokenizer = model.get_tokenizer()
        train_loader = task.load_data("train")

        # ── Memory-saving settings ──
        lm.train()
        if hasattr(lm, "gradient_checkpointing_enable"):
            lm.gradient_checkpointing_enable()
            logger.info("  Gradient checkpointing enabled.")

        # ── Optimizer ──
        optimizer = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=0.01)

        # Mixed precision scaler (fp16 on CUDA, disabled on MPS/CPU)
        use_amp = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

        # ── Training loop ──
        total_steps = 0
        optimizer.zero_grad()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for step, batch in enumerate(tqdm(train_loader, desc=f"  SFT epoch {epoch+1}/{n_epochs}")):
                prompts: List[str]  = batch["prompts"]
                answers: List[str]  = batch.get("answers", [""] * len(prompts))

                # Skip samples with no gold answer
                valid = [(p, a) for p, a in zip(prompts, answers) if a and a.strip()]
                if not valid:
                    continue
                prompts_v, answers_v = zip(*valid)

                loss = self._compute_loss(
                    lm, tokenizer, list(prompts_v), list(answers_v), device, use_amp, amp_dtype
                )
                if loss is None:
                    continue

                loss = loss / grad_accum
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * grad_accum
                n_batches += 1

                if (step + 1) % grad_accum == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lm.parameters(), max_norm)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    total_steps += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"  SFT epoch {epoch+1}/{n_epochs}: avg_loss={avg_loss:.4f}, steps={total_steps}")

        # Flush remaining gradients
        if (n_batches % grad_accum) != 0:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lm.parameters(), max_norm)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        lm.eval()
        if hasattr(lm, "gradient_checkpointing_disable"):
            lm.gradient_checkpointing_disable()

        # Return the fine-tuned model directly (no separate head)
        head = task.build_head(1, device)
        return model, head

    def _compute_loss(
        self,
        lm: nn.Module,
        tokenizer,
        prompts: List[str],
        answers: List[str],
        device: torch.device,
        use_amp: bool,
        amp_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Tokenize (prompt + answer) sequences and compute causal LM loss
        masked to answer tokens only.

        Label layout:
          [prompt tokens: -100, -100, ...] [answer tokens: t0, t1, t2, ...]
          CE loss is averaged over answer tokens only.
        """
        # Full sequences: prompt + answer
        full_texts = [p + a for p, a in zip(prompts, answers)]

        try:
            full_enc = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            prompt_enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=False,
            )
        except Exception as e:
            logger.warning(f"  Tokenization error: {e}")
            return None

        input_ids      = full_enc["input_ids"].to(device)
        attention_mask = full_enc["attention_mask"].to(device)

        # Build labels: -100 for prompt tokens, real IDs for answer tokens
        labels = input_ids.clone()
        for i in range(len(prompts)):
            # Number of prompt tokens for this sample (un-padded)
            prompt_tok_count = prompt_enc["attention_mask"][i].sum().item()
            labels[i, :int(prompt_tok_count)] = -100

        # Mask padding tokens from loss too
        labels[attention_mask == 0] = -100

        ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else torch.no_grad.__class__()
        # Use autocast for forward, not no_grad (we need gradients)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            outputs = lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return outputs.loss
