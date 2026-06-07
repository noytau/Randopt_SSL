"""
Countdown Task — arithmetic target construction.

Given a set of numbers (e.g. [2, 3, 5, 10]) and a target (e.g. 19),
produce an arithmetic expression using +, −, ×, ÷ that equals the target.
Each input number may be used at most once.

Verifier: parse the model's expression, eval() it safely, check == target.

Dataset: Jiayi-Pan/Countdown-Tasks-3to4 — dedicated countdown benchmark, 3–4 numbers.
Fallback: synthetic generation (random numbers + guaranteed-solvable targets).

Gold solutions for SFT are computed on-the-fly with a brute-force solver
(enumerate all permutations × operator combinations × bracket structures).

Prompt format:
  "Using the numbers [2, 3, 5, 10] and the operations +, -, *, /,
   write an arithmetic expression that equals 19.
   Each number may be used at most once. Output only the expression."

This is the primary benchmark task from the Neural Thickets paper (arXiv 2603.12228).
"""

import ast
import logging
import operator
import random
import re
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.registry import register_task
from src.tasks.llm.base import GenerativeTask

logger = logging.getLogger(__name__)

# Safe operators for expression evaluation
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}


def _safe_eval(expr: str) -> Optional[float]:
    """Evaluate a simple arithmetic expression safely (no exec, no builtins)."""
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode="eval")
        return _eval_node(tree.body)
    except Exception:
        return None


def _eval_node(node) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Num):  # Python <3.8 compat
        return float(node.n)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if type(node.op) is ast.Div and right == 0:
            raise ZeroDivisionError
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class CountdownDataset(Dataset):
    """Wraps Countdown problem samples for DataLoader."""

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def countdown_collate(batch: List[Dict]) -> Dict:
    """Collate Countdown samples into a batch with prompts + raw sample dicts."""
    return {
        "prompts": [item["prompt"] for item in batch],
        "answers": [item.get("solution", "") for item in batch],  # gold answer for SFT
        "samples": batch,   # keep raw for verify()
    }


# ── Synthetic data generator (fallback) ───────────────────────────────────────

def _generate_synthetic(n: int, seed: int = 42) -> List[Dict]:
    """
    Generate n synthetic Countdown problems with guaranteed solutions.
    Each problem: 4–6 numbers drawn from [1, 25], target = result of random expression.
    """
    rng = random.Random(seed)
    problems = []
    attempts = 0

    while len(problems) < n and attempts < n * 20:
        attempts += 1
        n_nums = rng.randint(4, 6)
        nums = [rng.randint(1, 25) for _ in range(n_nums)]

        # Build a random valid expression
        perm = list(rng.sample(nums, len(nums)))
        ops_pool = ['+', '-', '*']  # avoid division to keep integers
        expr_nums = perm[:rng.randint(2, len(perm))]
        ops = [rng.choice(ops_pool) for _ in range(len(expr_nums) - 1)]
        expr = str(expr_nums[0])
        for op, num in zip(ops, expr_nums[1:]):
            expr += f" {op} {num}"
        target = _safe_eval(expr)
        if target is None or not (1 <= target <= 999) or int(target) != target:
            continue
        target = int(target)

        sample = {
            "numbers": nums,
            "target": target,
            "solution": expr,
        }
        sample["prompt"] = _format_prompt(sample)
        problems.append(sample)

    logger.info(f"Generated {len(problems)} synthetic Countdown problems.")
    return problems


def _solve_countdown(nums: List[int], target: int) -> Optional[str]:
    """
    Brute-force solver: find any arithmetic expression over nums that equals target.
    Tries all permutations × operator sequences × bracket structures.
    Returns the expression string, or None if unsolvable.
    Used to generate gold solutions for SFT training.
    """
    ops = ['+', '-', '*', '/']

    def apply(a: float, b: float, op: str) -> Optional[float]:
        # NOTE: do NOT use a dict literal here — Python evaluates all values
        # eagerly, so {... '/':(a/b)} raises ZeroDivisionError before the guard.
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/':
            if abs(b) < 1e-9:
                return None
            return a / b
        return None

    # For n numbers, build all possible expression trees.
    # We use a recursive approach: pick two values, combine, recurse.
    def solve(vals: List[Tuple[float, str]]) -> Optional[str]:
        if len(vals) == 1:
            v, expr = vals[0]
            if abs(v - target) < 1e-6:
                return expr
            return None
        for i in range(len(vals)):
            for j in range(len(vals)):
                if i == j:
                    continue
                a, ea = vals[i]
                b, eb = vals[j]
                rest = [vals[k] for k in range(len(vals)) if k != i and k != j]
                for op in ops:
                    r = apply(a, b, op)
                    if r is None:
                        continue
                    # Build expression string with parens to be unambiguous
                    if op in ('*', '/'):
                        new_expr = f"({ea} {op} {eb})"
                    else:
                        new_expr = f"({ea} {op} {eb})"
                    result = solve(rest + [(r, new_expr)])
                    if result is not None:
                        return result
        return None

    for perm in permutations(nums):
        vals = [(float(n), str(n)) for n in perm]
        result = solve(vals)
        if result is not None:
            # Strip outermost parens for cleanliness
            return result.strip('()')
    return None


def _format_prompt(sample: Dict) -> str:
    nums_str = ", ".join(str(n) for n in sample["numbers"])
    target = sample["target"]
    return (
        f"Using the numbers [{nums_str}] and the operations +, -, *, /, "
        f"create an arithmetic expression that equals {target}. "
        f"Each number may be used at most once. "
        f"Think step by step, then give your final answer as a single arithmetic expression."
    )


# ── Task ───────────────────────────────────────────────────────────────────────

@register_task("countdown")
class CountdownTask(GenerativeTask):
    """
    Countdown arithmetic task (Neural Thickets paper, primary benchmark).

    Primary metric: accuracy (fraction of problems solved exactly).
    Dataset: allenai/tulu-3-sft-mixture countdown split, or synthetic fallback.
    """

    def __init__(
        self,
        max_train_samples: int = None,
        max_val_samples: int = 200,
        max_test_samples: int = 500,
        batch_size: int = 8,
        num_workers: int = 0,
        use_synthetic: bool = False,
        seed: int = 42,
    ):
        self._max_train = max_train_samples
        self._max_val = max_val_samples
        self._max_test = max_test_samples
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._use_synthetic = use_synthetic
        self._seed = seed
        self._dataloaders: Dict[str, DataLoader] = {}

    def name(self) -> str:
        return "countdown"

    def primary_metric(self) -> Tuple[str, bool]:
        return ("accuracy", True)

    def set_tokenizer(self, tokenizer):
        pass  # GenerativeTask uses model.generate() — no tokenizer needed at task level

    def set_processor(self, processor):
        pass

    def load_data(self, split: str = "train") -> DataLoader:
        if split in self._dataloaders:
            return self._dataloaders[split]

        samples = self._load_samples(split)
        dataset = CountdownDataset(samples)
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=(split == "train"),
            collate_fn=countdown_collate,
            num_workers=self._num_workers,
        )
        self._dataloaders[split] = loader
        logger.info(f"  Loaded {len(dataset)} Countdown samples ({split}), {len(loader)} batches.")
        return loader

    def _load_samples(self, split: str) -> List[Dict]:
        """Load from HuggingFace tulu-3-sft-mixture or fall back to synthetic."""
        max_s = {"train": self._max_train, "val": self._max_val, "test": self._max_test}.get(split)

        if not self._use_synthetic:
            try:
                return self._load_hf(split, max_s)
            except Exception as e:
                logger.warning(f"  HF dataset load failed ({e}); falling back to synthetic.")

        # Synthetic fallback
        n = max_s or {"train": 1000, "val": 200, "test": 500}.get(split, 200)
        seed_offset = {"train": 0, "val": 10000, "test": 20000}.get(split, 0)
        return _generate_synthetic(n, seed=self._seed + seed_offset)

    def _load_hf(self, split: str, max_s: Optional[int]) -> List[Dict]:
        """
        Load Countdown problems from Jiayi-Pan/Countdown-Tasks-3to4.

        This is a dedicated Countdown benchmark dataset (3–4 numbers, integer targets).
        Schema: {'nums': [int, ...], 'target': int}  — no gold solution provided.
        Gold solutions are computed with _solve_countdown() so SFT has training signal.

        All three splits are carved from the dataset on the first call and cached.
        """
        from datasets import load_dataset

        if not hasattr(self, "_hf_cache"):
            n_target = (
                (self._max_train or 1000)
                + (self._max_val or 200)
                + (self._max_test or 500)
                + 200
            )
            logger.info(
                f"  Loading Jiayi-Pan/Countdown-Tasks-3to4 "
                f"(want ≥{n_target} solved problems)…"
            )
            ds = load_dataset(
                "Jiayi-Pan/Countdown-Tasks-3to4",
                split="train",
                streaming=True,
            )

            all_samples: List[Dict] = []
            n_unsolvable = 0
            for item in ds:
                nums   = [int(x) for x in item["nums"]]
                target = int(item["target"])
                sol = _solve_countdown(nums, target)
                if sol is None:
                    n_unsolvable += 1
                    continue  # skip problems our solver can't crack
                sample = {"numbers": nums, "target": target, "solution": sol}
                sample["prompt"] = _format_prompt(sample)
                all_samples.append(sample)
                if len(all_samples) >= n_target:
                    break

            logger.info(
                f"  Loaded {len(all_samples)} solvable problems "
                f"({n_unsolvable} skipped as unsolvable by brute-force solver)."
            )
            if not all_samples:
                raise RuntimeError("No solvable countdown problems found.")

            # Deterministic train / val / test split
            rng = random.Random(self._seed)
            rng.shuffle(all_samples)
            n_total = len(all_samples)
            n_val  = max(100, n_total // 10)
            n_test = max(200, n_total // 10)
            self._hf_cache: Dict[str, List[Dict]] = {
                "train": all_samples[n_val + n_test:],
                "val":   all_samples[:n_val],
                "test":  all_samples[n_val: n_val + n_test],
            }
            logger.info(
                f"  HF cache: train={len(self._hf_cache['train'])}, "
                f"val={len(self._hf_cache['val'])}, test={len(self._hf_cache['test'])}"
            )

        chosen = self._hf_cache.get(split, self._hf_cache["train"])
        if max_s and len(chosen) > max_s:
            chosen = chosen[:max_s]
        logger.info(f"  {len(chosen)} samples selected for {split} split.")
        return chosen

    def format_prompt(self, sample: Dict) -> str:
        return _format_prompt(sample)

    def get_answer_text(self, sample: Dict) -> Optional[str]:
        """Gold answer for SFT: the arithmetic expression that solves the problem."""
        sol = sample.get("solution", "")
        return sol if sol else None

    def verify(self, response: str, sample: Dict) -> bool:
        """
        Parse the model's response and check if it contains an arithmetic expression
        that equals the target. Handles chain-of-thought output: scans every line,
        tries to extract and evaluate any arithmetic expression.
        """
        target = sample["target"]

        # Split on newlines and common delimiters like "Answer:", "="
        lines = response.strip().split('\n')
        candidates = []
        for line in lines:
            line = line.strip()
            # Strip markdown code fences and backticks
            line = line.strip('`').strip()
            # Strip leading labels like "Answer:", "Expression:", "Result:"
            line = re.sub(r'^[A-Za-z ]+:\s*', '', line)
            # Strip trailing "= <number>" so "2*(10-3)+5 = 19" → "2*(10-3)+5"
            line = re.sub(r'\s*=\s*[-\d.]+\s*$', '', line).strip()
            candidates.append(line)

        for expr in candidates:
            if not re.search(r'\d', expr):
                continue
            if not re.search(r'[+\-*/]', expr):
                continue
            result = _safe_eval(expr)
            if result is not None and abs(result - target) < 1e-6:
                return True

        return False
