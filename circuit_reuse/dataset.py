from __future__ import annotations

import random
import os
from dataclasses import dataclass
from typing import List, Tuple, Iterable
from datasets import load_dataset


@dataclass
class Example:
    """Stores a clean prompt/target pair and a corresponding corrupted version."""
    prompt: str
    target: str
    corrupted_prompt: str
    corrupted_target: str


class AdditionDataset:
    def __init__(self, num_examples: int = 100, digits: int = 2) -> None:
        self.num_examples = num_examples
        self.digits = digits
        self._examples: List[Example] = []
        self._generate_examples()

    def _generate_single_example(self) -> Tuple[str, str]:
        low = 10 ** (self.digits - 1)
        high = (10**self.digits) - 1
        a = random.randint(low, high)
        b = random.randint(low, high)
        prompt = f"Compute: {a} + {b} = "
        target = str(a + b)
        return prompt, target

    def _generate_examples(self) -> None:
        self._examples = []
        pairs = [self._generate_single_example() for _ in range(self.num_examples)]
        shuffled = pairs[:]
        random.shuffle(shuffled)
        for (prompt, target), (corrupted_prompt, corrupted_target) in zip(pairs, shuffled):
            self._examples.append(Example(prompt, target, corrupted_prompt, corrupted_target))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]

    def __iter__(self) -> Iterable[Example]:
        return iter(self._examples)


class BooleanDataset:
    def __init__(
        self,
        num_examples: int = 100,
        min_ops: int = 2,
        max_ops: int = 6,
        allow_parentheses: bool = True,
        allow_not: bool = True,
    ) -> None:
        self.num_examples = num_examples
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.allow_parentheses = allow_parentheses
        self.allow_not = allow_not
        self._examples: List[Example] = []
        self._generate_examples()

    def _rand_bool(self) -> str:
        return random.choice(["true", "false"])

    def _maybe_not(self, token: str) -> str:
        if self.allow_not and random.random() < 0.3:
            return "not " + token
        return token

    def _gen_expr(self, remaining_ops: int) -> str:
        if remaining_ops == 0:
            return self._maybe_not(self._rand_bool())
        left_ops = random.randint(0, remaining_ops - 1)
        right_ops = remaining_ops - 1 - left_ops
        left = self._gen_expr(left_ops)
        right = self._gen_expr(right_ops)
        op = random.choice(["and", "or"])
        expr = f"{left} {op} {right}"
        if self.allow_parentheses and random.random() < 0.5:
            expr = f"({expr})"
        return expr

    def _evaluate(self, expr: str) -> bool:
        py_expr = expr.replace("true", "True").replace("false", "False")
        return bool(eval(py_expr))  # noqa: S307 (controlled vocabulary)

    def _corrupt_expr(self, expr: str) -> str:
        """Create a corrupted version of an expression by flipping one boolean literal."""
        literals = ["true", "false"]
        found_literals = [(m.start(), m.end()) for m in re.finditer(r"\b(true|false)\b", expr)]
        if not found_literals:
            return expr  # No literal to flip
        start, end = random.choice(found_literals)
        original_literal = expr[start:end]
        flipped_literal = "false" if original_literal == "true" else "true"
        return expr[:start] + flipped_literal + expr[end:]

    def _generate_examples(self) -> None:
        self._examples = []
        seen = set()
        attempts = 0
        max_attempts = self.num_examples * 20

        while len(self._examples) < self.num_examples and attempts < max_attempts:
            attempts += 1
            n_ops = random.randint(self.min_ops, self.max_ops)
            expr = self._gen_expr(n_ops)
            if expr in seen:
                continue
            seen.add(expr)

            prompt = f"Evaluate: {expr} = "
            target = str(self._evaluate(expr)).lower()

            corrupted_expr = self._corrupt_expr(expr)
            corrupted_prompt = f"Evaluate: {corrupted_expr} = "
            corrupted_target = str(self._evaluate(corrupted_expr)).lower()

            self._examples.append(Example(prompt, target, corrupted_prompt, corrupted_target))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]

    def __iter__(self) -> Iterable[Example]:
        return iter(self._examples)


class MMLUDataset:
    def __init__(
        self, subject: str = "high_school_european_history", split: str = "test", num_examples: int | None = None
    ) -> None:
        ds = load_dataset("cais/mmlu", subject, split=split)
        self._examples: List[Example] = []
        for i, item in enumerate(ds):
            q = item["question"]
            choices = item["choices"]
            ans_idx = item["answer"]

            # Clean example
            prompt = f"{q}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: "
            target = chr(ord("A") + ans_idx)

            # Corrupted example (shuffled choices)
            shuffled_choices = choices[:]
            random.shuffle(shuffled_choices)
            correct_answer_text = choices[ans_idx]
            new_ans_idx = shuffled_choices.index(correct_answer_text)

            corrupted_prompt = (
                f"{q}\n"
                f"A. {shuffled_choices[0]}\n"
                f"B. {shuffled_choices[1]}\n"
                f"C. {shuffled_choices[2]}\n"
                f"D. {shuffled_choices[3]}\n"
                f"Answer: "
            )
            corrupted_target = chr(ord("A") + new_ans_idx)

            self._examples.append(Example(prompt, target, corrupted_prompt, corrupted_target))
            if num_examples is not None and i + 1 >= num_examples:
                break

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]

    def __iter__(self) -> Iterable[Example]:
        return iter(self._examples)


class IOIDataset:
    """Loads the Indirect Object Identification (IOI) dataset from MIB-bench."""

    def __init__(self, split: str = "test", num_examples: int | None = None) -> None:
        ds = load_dataset("mib-bench/ioi", split=split)
        self._examples: List[Example] = []
        count = 0
        for item in ds:
            # Clean example
            prompt = item["prompt"]
            target = item["choices"][item["answerKey"]]

            # Corrupted example using the 'random_names_counterfactual'
            corrupted_item = item["random_names_counterfactual"]
            corrupted_prompt = corrupted_item["prompt"]
            corrupted_target = corrupted_item["choices"][corrupted_item["answerKey"]]

            self._examples.append(Example(prompt, target, corrupted_prompt, corrupted_target))
            count += 1
            if num_examples is not None and count >= num_examples:
                break

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]

    def __iter__(self) -> Iterable[Example]:
        return iter(self._examples)


class MCQADataset:
    """Loads the CopyColors MCQA dataset from MIB-bench."""

    def __init__(
        self,
        split: str = "test",
        num_examples: int | None = None,
        n: int = 4,
    ) -> None:
        if not (2 <= n <= 10):
            raise ValueError(f"MCQA 'n' must be in [2,10], got {n}")
        
        config = f"{n}_answer_choices"

        # Hugging Face dataset requires a config per n-way subset
        # e.g., "4_answer_choices". See dataset files for details.
        # https://huggingface.co/datasets/mib-bench/copycolors_mcqa
        ds = load_dataset("mib-bench/copycolors_mcqa", config, split=split)
        self._examples: List[Example] = []
        count = 0
        for item in ds:
            # Clean example
            prompt = item["prompt"]
            target = item["choices"]["label"][item["answerKey"]]

            # Corrupted example using 'answerPosition_counterfactual'
            corrupted_item = item["answerPosition_counterfactual"]
            corrupted_prompt = corrupted_item["prompt"]
            corrupted_target = corrupted_item["choices"]["label"][corrupted_item["answerKey"]]

            self._examples.append(Example(prompt, target, corrupted_prompt, corrupted_target))
            count += 1
            if num_examples is not None and count >= num_examples:
                break

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]

    def __iter__(self) -> Iterable[Example]:
        return iter(self._examples)


class ARCDataset:
    """Loads ARC (Easy or Challenge) datasets from MIB-bench."""

    def __init__(self, name: str, split: str = "test", num_examples: int | None = None) -> None:
        assert name in ("arc_easy", "arc_challenge")
        ds = load_dataset(f"mib-bench/{name}", split=split)
        self._examples: List[Example] = []
        count = 0
        for item in ds:
            # Clean example
            prompt = item["prompt"]
            # The 'label' field contains the actual choice characters (e.g., 'A', 'B', 'C', 'D')
            target = item["choices"]["label"][item["answerKey"]]

            # Corrupted example using 'answerPosition_counterfactual'
            corrupted_item = item["answerPosition_counterfactual"]
            corrupted_prompt = corrupted_item["prompt"]
            corrupted_target = corrupted_item["choices"]["label"][corrupted_item["answerKey"]]

            self._examples.append(Example(prompt, target, corrupted_prompt, corrupted_target))
            count += 1
            if num_examples is not None and count >= num_examples:
                break

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]

    def __iter__(self) -> Iterable[Example]:
        return iter(self._examples)


DATASET_DISPLAY_NAMES: dict[str, str] = {
    "addition": "Addition",
    "boolean": "Boolean",
    "mmlu": "MMLU",
    "ioi": "IOI",
    "mcqa": "Colored Objects MCQA",
    "arc_easy": "ARC (Easy)",
    "arc_challenge": "ARC (Challenge)",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B Instruct",
    "gemma-2-2b": "Gemma 2 2B",
    "gemma-2-2b-it": "Gemma 2 2B Instruct",
}


def get_task_display_name(task: str) -> str:
    if task in DATASET_DISPLAY_NAMES:
        return DATASET_DISPLAY_NAMES[task]
    return task.replace("_", " ").title()


def get_model_display_name(model: str) -> str:
    if model in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model]
    parts = model.split("-")
    if not parts:
        return model
    first = parts[0].title()
    rest = []
    for p in parts[1:]:
        if p.endswith("b") and p[:-1].replace(".", "").isdigit():
            rest.append(p[:-1] + p[-1].upper())
        else:
            rest.append(p)
    return " ".join([first] + rest)


def get_dataset(task: str, num_examples: int = 100, digits: int = 2) -> Iterable[Example]:
    if task == "addition":
        return AdditionDataset(num_examples=num_examples, digits=digits)
    if task == "boolean":
        return BooleanDataset(num_examples=num_examples)
    if task == "mmlu":
        return MMLUDataset(split="test", num_examples=num_examples)
    if task == "ioi":
        return IOIDataset(split="test", num_examples=num_examples)
    if task == "mcqa":
        return MCQADataset(split="test", num_examples=num_examples)
    if task in ("arc_easy", "arc_challenge"):
        return ARCDataset(name=task, split="test", num_examples=num_examples)
    raise ValueError(f"Unsupported task: {task}")


__all__ = [
    "Example",
    "AdditionDataset",
    "BooleanDataset",
    "MMLUDataset",
    "IOIDataset",
    "MCQADataset",
    "ARCDataset",
    "get_dataset",
    "get_task_display_name",
    "get_model_display_name",
]