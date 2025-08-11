from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple, Iterable

@dataclass
class ArithmeticExample:
	prompt: str
	target: str


class AdditionDataset:
	def __init__(self, num_examples: int = 100, digits: int = 2) -> None:
		self.num_examples = num_examples
		self.digits = digits
		self._examples: List[ArithmeticExample] = []
		self._generate_examples()

	def _generate_examples(self) -> None:
		self._examples = []
		low = 10 ** (self.digits - 1)
		high = 10 ** self.digits - 1
		for _ in range(self.num_examples):
			a = random.randint(low, high)
			b = random.randint(low, high)
			prompt = f"Compute: {a} + {b} = "
			target = str(a + b)
			self._examples.append(ArithmeticExample(prompt, target))

	def __len__(self) -> int:
		return len(self._examples)

	def __getitem__(self, idx: int) -> ArithmeticExample:
		return self._examples[idx]

	def __iter__(self) -> Iterable[ArithmeticExample]:
		return iter(self._examples)


class BooleanDataset:
	def __init__(self, num_examples: int = 100) -> None:
		self.num_examples = num_examples
		self._examples: List[ArithmeticExample] = []
		self._generate_examples()

	def _generate_examples(self) -> None:
		ops = ["AND", "OR"]
		for _ in range(self.num_examples):
			a = random.choice([True, False])
			b = random.choice([True, False])
			op = random.choice(ops)
			if random.random() < 0.5:
				a = not a
			if random.random() < 0.5:
				b = not b
			prompt = f"Respond with either true or false. {str(a).lower()} {op} {str(b).lower()} = "
			target_bool = (a and b) if op == "AND" else (a or b)
			target = str(target_bool).lower()
			self._examples.append(ArithmeticExample(prompt, target))

	def __len__(self) -> int:
		return len(self._examples)

	def __getitem__(self, idx: int) -> ArithmeticExample:
		return self._examples[idx]

	def __iter__(self) -> Iterable[ArithmeticExample]:
		return iter(self._examples)

class MMLUDataset:
	def __init__(self, subject: str = "high_school_european_history", split: str = "test", num_examples: int | None = None) -> None:
		try:
			from datasets import load_dataset  # type: ignore
		except ImportError as e:  # pragma: no cover - optional dep
			raise ImportError("HuggingFace 'datasets' required for MMLU") from e
		ds = load_dataset("cais/mmlu", subject, split=split)
		self._examples: List[ArithmeticExample] = []
		for i, item in enumerate(ds):
			q = item["question"]
			choices = item["choices"]
			ans = item["answer"]
			parts = [f"({chr(ord('A') + idx)}) {c}" for idx, c in enumerate(choices)]
			prompt = f"{q} " + " ".join(parts)
			self._examples.append(ArithmeticExample(prompt, str(ans)))
			if num_examples is not None and i + 1 >= num_examples:
				break

	def __len__(self) -> int:
		return len(self._examples)

	def __getitem__(self, idx: int) -> ArithmeticExample:
		return self._examples[idx]

	def __iter__(self) -> Iterable[ArithmeticExample]:
		return iter(self._examples)


class MIBDatasetHF:
	def __init__(self, name: str = "ioi", split: str = "test", num_examples: int | None = None) -> None:
		try:
			from datasets import load_dataset  # type: ignore
		except ImportError as e:  # pragma: no cover
			raise ImportError("HuggingFace 'datasets' required for MIB") from e
		ds = load_dataset(f"mib-bench/{name}", split=split)
		self._examples: List[ArithmeticExample] = []
		count = 0
		for item in ds:
			prompt_text = item.get("prompt") or item.get("template") or item.get("question")
			choices = item.get("choices", [])
			answer_key = item.get("answerKey", -1)
			if prompt_text is None or not choices or answer_key is None or answer_key < 0:
				continue
			parts = [f"({chr(ord('A') + idx)}) {c}" for idx, c in enumerate(choices)]
			prompt = f"{prompt_text} " + " ".join(parts)
			if answer_key >= len(choices):
				continue
			target = chr(ord('A') + answer_key)
			self._examples.append(ArithmeticExample(prompt.strip(), target))
			count += 1
			if num_examples is not None and count >= num_examples:
				break

	def __len__(self) -> int:
		return len(self._examples)

	def __getitem__(self, idx: int) -> ArithmeticExample:
		return self._examples[idx]

	def __iter__(self) -> Iterable[ArithmeticExample]:
		return iter(self._examples)


def get_dataset(task: str, num_examples: int = 100, digits: int = 2) -> Iterable[ArithmeticExample]:
	if task == "addition":
		return AdditionDataset(num_examples=num_examples, digits=digits)
	if task == "boolean":
		return BooleanDataset(num_examples=num_examples)
	if task == "mmlu":
		return MMLUDataset(split='test', num_examples=num_examples)
	if task.startswith("mib_"):
		return MIBDatasetHF(name=task[len("mib_"):], split="test", num_examples=num_examples)
	raise ValueError(f"Unsupported task: {task}")

__all__ = [
	"ArithmeticExample",
	"AdditionDataset",
	"BooleanDataset",
	"MMLUDataset",
	"MIBDatasetHF",
	"get_dataset",
]
