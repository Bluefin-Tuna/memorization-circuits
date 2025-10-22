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
        target_layers="early",
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.target_layers = target_layers
        
        if module_filter is None:
            def default_filter(name: str, module: torch.nn.Module) -> bool:
                n = name.lower()
                if "visual.blocks" in n and n.endswith(".attn") and "." not in n.split(".attn")[0].split("blocks.")[1]:
                    return True
                if n == "model.visual.merger":
                    return True
                if "language_model.layers" in n and n.endswith(".self_attn"):
                    try:
                        layer_num = int(n.split("layers.")[1].split(".")[0])
                        if target_layers == "early":
                            return layer_num < 8
                        elif target_layers == "all":
                            return True
                        elif isinstance(target_layers, list):
                            return layer_num in target_layers
                    except (IndexError, ValueError):
                        pass
                return False
            module_filter = default_filter
        self.modules = [
            (name, module)
            for name, module in self.model.named_modules()
            if module_filter(name, module)
        ]

    def _prepare_inputs(self, ex: VLMExample):
        img_or_path = ex.image if ex.image is not None else ex.image_path
        # Try to use chat template if available
        try:
            if hasattr(self.processor, "apply_chat_template") and hasattr(self.processor, "chat_template") and self.processor.chat_template is not None:
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
                raise AttributeError("No chat template")
        except (ValueError, AttributeError, RuntimeError):
            # Fallback for processors without chat template or if chat template fails
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
            tok = getattr(self.model, "tokenizer", None)
            raw = tok.batch_decode(out, skip_special_tokens=True)[0] if tok is not None else ""
        return _extract_answer(raw)

    def analyze_pair(self, clean: VLMExample, biased: VLMExample):
        biased_inputs = self._prepare_inputs(biased)
        baseline_pred = self._predict(biased_inputs)
        baseline_correct = _normalise_answer(baseline_pred) == _normalise_answer(clean.ground_truth)

        clean_inputs = self._prepare_inputs(clean)
        clean_outputs = {}
        clean_shapes = {}

        hooks = []

        def save_hook(name):
            def hook_fn(module, inp, out):
                if isinstance(out, tuple):
                    clean_outputs[name] = tuple(o.detach().clone() if isinstance(o, torch.Tensor) else o for o in out)
                    clean_shapes[name] = out[0].shape if isinstance(out[0], torch.Tensor) else None
                else:
                    clean_outputs[name] = out.detach().clone()
                    clean_shapes[name] = out.shape
                return out
            return hook_fn

        for name, module in self.modules:
            hooks.append(module.register_forward_hook(save_hook(name)))
        _ = self._predict(clean_inputs)
        for h in hooks:
            h.remove()

        effects = {}
        for name, module in self.modules:
            clean_shape = clean_shapes.get(name)
            if clean_shape is None:
                effects[name] = 0.0
                continue
            
            def patch_fn(module, inp, out, name=name, expected_shape=clean_shape):
                clean_out = clean_outputs[name]
                
                if isinstance(out, tuple) and isinstance(clean_out, tuple):
                    if out[0].shape != expected_shape:
                        return out
                    return (clean_out[0],) + out[1:]
                
                if out.shape != expected_shape:
                    return out
                
                return clean_out

            handle = module.register_forward_hook(patch_fn)
            try:
                fresh_biased_inputs = self._prepare_inputs(biased)
                patched_pred = self._predict(fresh_biased_inputs)
                patched_correct = _normalise_answer(patched_pred) == _normalise_answer(clean.ground_truth)
                effects[name] = int(patched_correct) - int(baseline_correct)
            except RuntimeError as e:
                effects[name] = 0.0
            finally:
                handle.remove()

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

    def analyze_heads(
        self,
        pairs: Iterable[tuple[VLMExample, VLMExample]],
        attention_modules: list[str] = None,
    ):
        if attention_modules is None:
            attention_modules = [
                name for name, _ in self.modules
                if "language_model.layers" in name and "self_attn" in name
            ]
        
        head_effects = {}
        n_pairs = 0
        
        for clean_ex, biased_ex in pairs:
            biased_inputs = self._prepare_inputs(biased_ex)
            baseline_pred = self._predict(biased_inputs)
            baseline_correct = _normalise_answer(baseline_pred) == _normalise_answer(clean_ex.ground_truth)
            
            clean_inputs = self._prepare_inputs(clean_ex)
            clean_head_outputs = {}
            
            hooks = []
            
            def save_head_hook(module_name):
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple):
                        clean_head_outputs[module_name] = tuple(o.detach() if isinstance(o, torch.Tensor) else o for o in out)
                    else:
                        clean_head_outputs[module_name] = out.detach()
                    return out
                return hook_fn
            
            for name, module in self.model.named_modules():
                if name in attention_modules:
                    hooks.append(module.register_forward_hook(save_head_hook(name)))
            
            _ = self._predict(clean_inputs)
            
            for h in hooks:
                h.remove()
            
            for module_name in attention_modules:
                attn_module = dict(self.model.named_modules())[module_name]
                
                if hasattr(attn_module, 'num_heads'):
                    num_heads = attn_module.num_heads
                elif hasattr(attn_module, 'num_attention_heads'):
                    num_heads = attn_module.num_attention_heads
                else:
                    continue
                
                hidden_size = attn_module.q_proj.out_features
                head_dim = hidden_size // num_heads
                
                if module_name in clean_head_outputs:
                    clean_out_sample = clean_head_outputs[module_name]
                    if isinstance(clean_out_sample, tuple):
                        clean_out_sample = clean_out_sample[0]
                    clean_seq_len = clean_out_sample.shape[1]
                else:
                    continue
                
                for head_idx in range(num_heads):
                    def patch_head_fn(module, inp, out, head_idx=head_idx, module_name=module_name):
                        clean_out = clean_head_outputs[module_name]
                        
                        if isinstance(out, tuple):
                            actual_out = out[0]
                            if isinstance(clean_out, tuple):
                                actual_clean = clean_out[0]
                            else:
                                actual_clean = clean_out
                        else:
                            actual_out = out
                            actual_clean = clean_out
                        
                        batch_size, seq_len, _ = actual_out.shape
                        clean_batch, clean_seq, _ = actual_clean.shape
                        
                        if seq_len != clean_seq:
                            min_seq = min(seq_len, clean_seq)
                            out_heads = actual_out.view(batch_size, seq_len, num_heads, head_dim)
                            clean_heads = actual_clean.view(clean_batch, clean_seq, num_heads, head_dim)
                            
                            out_heads[:, :min_seq, head_idx, :] = clean_heads[:, :min_seq, head_idx, :]
                            
                            patched = out_heads.view(batch_size, seq_len, -1)
                        else:
                            out_heads = actual_out.view(batch_size, seq_len, num_heads, head_dim)
                            clean_heads = actual_clean.view(clean_batch, clean_seq, num_heads, head_dim)
                            
                            out_heads[:, :, head_idx, :] = clean_heads[:, :, head_idx, :]
                            
                            patched = out_heads.view(batch_size, seq_len, -1)
                        
                        if isinstance(out, tuple):
                            return (patched,) + out[1:]
                        return patched
                    
                    handle = attn_module.register_forward_hook(patch_head_fn)
                    patched_pred = self._predict(biased_inputs)
                    handle.remove()
                    
                    patched_correct = _normalise_answer(patched_pred) == _normalise_answer(clean_ex.ground_truth)
                    effect = int(patched_correct) - int(baseline_correct)
                    
                    key = (module_name, head_idx)
                    head_effects[key] = head_effects.get(key, 0.0) + effect
            
            n_pairs += 1
        
        if n_pairs == 0:
            return {}
        
        return {key: eff / n_pairs for key, eff in head_effects.items()}

    def ablate_heads(
        self,
        examples: Iterable[VLMExample],
        heads_to_ablate: list[tuple[str, int]],
    ):
        correct = 0
        total = 0
        bias_aligned_errors = 0
        total_errors = 0
        
        ablation_map = {}
        for module_name, head_idx in heads_to_ablate:
            ablation_map.setdefault(module_name, []).append(head_idx)
        
        hooks = []
        
        for module_name, head_indices in ablation_map.items():
            attn_module = dict(self.model.named_modules())[module_name]
            
            if hasattr(attn_module, 'num_heads'):
                num_heads = attn_module.num_heads
            elif hasattr(attn_module, 'num_attention_heads'):
                num_heads = attn_module.num_attention_heads
            else:
                continue
            
            hidden_size = attn_module.q_proj.out_features
            head_dim = hidden_size // num_heads
            
            def ablation_hook(module, inp, out, head_indices=head_indices):
                if isinstance(out, tuple):
                    actual_out = out[0]
                else:
                    actual_out = out
                
                batch_size, seq_len, _ = actual_out.shape
                out_heads = actual_out.view(batch_size, seq_len, num_heads, head_dim)
                
                for head_idx in head_indices:
                    out_heads[:, :, head_idx, :] = 0.0
                
                ablated = out_heads.view(batch_size, seq_len, -1)
                
                if isinstance(out, tuple):
                    return (ablated,) + out[1:]
                return ablated
            
            hooks.append(attn_module.register_forward_hook(ablation_hook))
        
        for ex in examples:
            inputs = self._prepare_inputs(ex)
            pred = self._predict(inputs)
            
            norm_pred = _normalise_answer(pred)
            norm_gt = _normalise_answer(ex.ground_truth)
            norm_bias = _normalise_answer(ex.expected_bias)
            
            is_correct = norm_pred == norm_gt
            if is_correct:
                correct += 1
            else:
                total_errors += 1
                if norm_pred == norm_bias:
                    bias_aligned_errors += 1
            total += 1
        
        for h in hooks:
            h.remove()
        
        accuracy = correct / total if total > 0 else 0.0
        baer = (bias_aligned_errors / total_errors) if total_errors > 0 else 0.0
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "total_errors": total_errors,
            "bias_aligned_errors": bias_aligned_errors,
            "baer": baer,
        }


__all__ = ["CircuitAnalyzer"]