from __future__ import annotations

from typing import Iterable
import torch

from .dataset import VLMExample
from .evaluate import _normalise_answer, _extract_answer


class CircuitAnalyzer:
    def __init__(
        self,
        model,
        processor,
        *,
        module_filter=None,
        max_new_tokens: int = 5,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        if module_filter is None:
            def default_filter(name: str, module: torch.nn.Module) -> bool:
                n = name.lower()
                return ("cross" in n and "attn" in n) or ("mm" in n and "attn" in n)
            module_filter = default_filter
        # Identify candidate modules
        self.modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if module_filter(name, module)
        ]

    def _prepare_inputs(self, ex: VLMExample):
        img_or_path = ex.image if ex.image is not None else ex.image_path
        if hasattr(self.processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_or_path},
                        {"type": "text", "text": ex.question},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inp = self.processor(text=[text], images=[img_or_path], return_tensors="pt", padding=True)
        else:
            sig = self.processor.__call__.__code__.co_varnames
            image_key = "images" if "images" in sig else "pixel_values"
            inp = self.processor(
                **{image_key: img_or_path},
                text=ex.question,
                return_tensors="pt",
            )
        return {k: v.to(self.model.device) for k, v in inp.items()}

    def _predict(self, inputs):
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        if hasattr(self.processor, "batch_decode"):
            raw = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        elif hasattr(getattr(self.processor, "tokenizer", None), "batch_decode"):
            raw = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        else:
            # Last resort: model tokenizer attribute
            tok = getattr(self.model, "tokenizer", None)
            raw = tok.batch_decode(out, skip_special_tokens=True)[0] if tok is not None else ""
        return _extract_answer(raw)

    def analyze_pair(self, clean: VLMExample, biased: VLMExample):
        # Compute baseline on biased input
        biased_inputs = self._prepare_inputs(biased)
        baseline_pred = self._predict(biased_inputs)
        baseline_correct = _normalise_answer(baseline_pred) == _normalise_answer(clean.ground_truth)

        # Capture clean activations
        clean_inputs = self._prepare_inputs(clean)
        clean_outputs = {}

        hooks = []

        def save_hook(name):
            def hook_fn(module, inp, out):
                clean_outputs[name] = out.detach()
                return out
            return hook_fn

        for name, module in self.modules:
            hooks.append(module.register_forward_hook(save_hook(name)))
        # Forward pass on clean to collect activations
        _ = self._predict(clean_inputs)
        # Remove hooks
        for h in hooks:
            h.remove()

        # Evaluate patched runs for each module
        effects = {}
        for name, module in self.modules:
            # Define patch hook that replaces output with clean activation
            def patch_fn(module, inp, out, name=name):
                clean_out = clean_outputs[name]
                # Broadcast if shape differs
                if clean_out.shape != out.shape:
                    reps = [out.size(i) // clean_out.size(i) if clean_out.size(i) != out.size(i) else 1
                            for i in range(len(out.shape))]
                    clean_out = clean_out.repeat(*reps)
                return clean_out

            handle = module.register_forward_hook(patch_fn)
            patched_pred = self._predict(biased_inputs)
            handle.remove()
            patched_correct = _normalise_answer(patched_pred) == _normalise_answer(clean.ground_truth)
            # Effect defined as change in correctness
            effects[name] = int(patched_correct) - int(baseline_correct)

        return effects

    def analyze_dataset(
        self,
        pairs: Iterable[tuple[VLMExample, VLMExample]],
    ):
        counts = {name: 0.0 for name, _ in self.modules}
        n_pairs = 0
        for clean_ex, biased_ex in pairs:
            per_mod = self.analyze_pair(clean_ex, biased_ex)
            for name, eff in per_mod.items():
                counts[name] += eff
            n_pairs += 1
        if n_pairs == 0:
            return {name: 0.0 for name, _ in self.modules}
        return {name: counts[name] / n_pairs for name, _ in self.modules}


__all__ = ["CircuitAnalyzer"]