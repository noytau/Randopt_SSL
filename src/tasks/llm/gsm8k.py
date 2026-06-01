"""
GSM8K Task — grade-school math word problems.

Dataset: openai/gsm8k (HuggingFace), 7,473 train / 1,319 test problems.
Each problem has a natural language question and a step-by-step solution ending with "#### <answer>".

Verifier: parse the last "#### <number>" from the model's response, compare to gold answer.

Primary metric: accuracy ↑

Prompt format:
  "Solve the following math problem step by step, then write your final answer
   after ####.

   Problem: <question>

   Solution:"
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.registry import register_task
from src.tasks.llm.base import GenerativeTask

logger = logging.getLogger(__name__)


def _extract_answer(text: str) -> Optional[float]:
    """Extract the numeric answer after '####' in the model's response."""
    # Look for #### <number>
    match = re.search(r'####\s*([\d,\.\-]+)', text)
    if match:
        num_str = match.group(1).replace(',', '').strip()
        try:
            return float(num_str)
        except ValueError:
            pass
    # Fallback: last number in the text
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    if nums:
        try:
            return float(nums[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def _format_prompt(sample: Dict) -> str:
    return (
        "Solve the following math problem step by step, then write your final answer after ####.\n\n"
        f"Problem: {sample['question']}\n\n"
        "Solution:"
    )


class GSM8KDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def gsm8k_collate(batch: List[Dict]) -> Dict:
    return {
        "prompts": [item["prompt"] for item in batch],
        "answers": [item.get("answer_text", "") for item in batch],  # full chain-of-thought for SFT
        "samples": batch,
    }


@register_task("gsm8k")
class GSM8KTask(GenerativeTask):
    """
    GSM8K: Grade-School Math (OpenAI).
    7,473 train + 1,319 test problems.
    Primary metric: exact-match accuracy on numeric answer.
    """

    def __init__(
        self,
        max_train_samples: int = None,
        max_val_samples: int = 200,
        max_test_samples: int = 500,
        batch_size: int = 4,
        num_workers: int = 0,
    ):
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._dataloaders: Dict[str, DataLoader] = {}

    def name(self) -> str:
        return "gsm8k"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("accuracy", True)

    def set_tokenizer(self, tokenizer): pass
    def set_processor(self, processor): pass

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        from datasets import load_dataset

        hf_split = {"train": "train", "val": "test", "test": "test"}.get(split, split)
        logger.info(f"  Loading gsm8k ({hf_split})...")
        ds = load_dataset("openai/gsm8k", "main", split=hf_split)

        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)
        if max_s and len(ds) > max_s:
            ds = ds.select(range(max_s))

        samples = []
        for item in ds:
            # Gold answer is after "####" in item["answer"]
            gold_match = re.search(r'####\s*([\d,\.\-]+)', item["answer"])
            gold = float(gold_match.group(1).replace(',', '')) if gold_match else None
            sample = {
                "question": item["question"],
                "answer_text": item["answer"],
                "gold_answer": gold,
            }
            sample["prompt"] = _format_prompt(sample)
            samples.append(sample)

        dataset = GSM8KDataset(samples)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=gsm8k_collate,
            num_workers=self._num_workers,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} GSM8K samples ({split}), {len(loader)} batches.")
        return loader

    def format_prompt(self, sample: Dict) -> str:
        return _format_prompt(sample)

    def get_answer_text(self, sample: Dict) -> Optional[str]:
        """Gold answer for SFT: full chain-of-thought reasoning + #### answer."""
        return sample.get("answer_text")

    def verify(self, response: str, sample: Dict) -> bool:
        """Check if the predicted answer matches the gold answer."""
        gold = sample.get("gold_answer")
        if gold is None:
            return False
        pred = _extract_answer(response)
        if pred is None:
            return False
        return abs(pred - gold) < 1e-6
