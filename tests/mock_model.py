"""Very small mock model for tests (minimal API)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

"""Fallback to lists if torch unavailable."""

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

class MockConfig:
    def __init__(self, n_layers: int = 1, n_heads: int = 1, device: str = "cpu"):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device

class MockModel:
    def __init__(self, predictions: Dict[str, int]):
        """Store mapping prompt->predicted token id."""
        self.predictions = predictions
        self.cfg = MockConfig()

    # ------- tokenisation / conversion utilities --------
    def to_tokens(self, prompt: str, prepend_bos: bool = True):
        """Return prompt string (acts as tokens)."""
        return prompt

    def to_single_token(self, target: str) -> int:
        """Map first digit/letter to token id."""
        first = target.strip()[0] if target else ""
        if first.isdigit():
            return int(first)
        # map letters to token ids for multiple choice tasks
        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3, "a": 0, "b": 1, "c": 2, "d": 3}
        return letter_map.get(first, 0)

    # ------- forward pass --------
    def __call__(self, tokens):
        """Return logits with chosen id high."""
        prompt = tokens
        vocab_size = 10
        # find the predicted id from the mapping; default to 0
        pred_id = self.predictions.get(prompt, 0)
        if torch is not None:
            logits = torch.full((1, 1, vocab_size), -1e9)
            logits[0, 0, pred_id] = 10.0
            return logits
        else:
            # create nested list filled with -1e9
            logits = [[[-1e9 for _ in range(vocab_size)]]]
            logits[0][0][pred_id] = 10.0
            return logits

    # ------- hook context manager --------
    @contextmanager
    def hooks(self, hooks):
        # ignore hooks for the mock
        yield

    def zero_grad(self):
        pass

    # convenience: store predicted id for current prompt
    def set_prediction_for_prompt(self, prompt: str):
        self._last_pred = self.predictions.get(prompt, 0)

__all__ = ["MockModel"]