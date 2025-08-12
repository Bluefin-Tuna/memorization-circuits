from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

@dataclass(frozen=True)
class Component:
	layer: int
	kind: str  # 'head' or 'mlp'
	index: int
	def __hash__(self) -> int:
		return hash((self.layer, self.kind, self.index))
	def __repr__(self) -> str:  # pragma: no cover - repr trivial
		return f"{self.kind}[layer={self.layer}, index={self.index}]"

class CircuitExtractor:
	def __init__(self, model: Any, top_k: Optional[int] = 5) -> None:
		self.model = model
		self.top_k = top_k
		self.model.cfg.use_split_qkv_input = True
		self.model.cfg.use_attn_result = True
		self.model.cfg.use_hook_mlp_in = True

	def _register_hooks(self):
		activations_head = {}
		activations_mlp = {}
		gradients_head = {}
		gradients_mlp = {}
		n_layers = self.model.cfg.n_layers
  
		def f_head(layer: int):
			def hook(act, hook=None):
				activations_head[layer] = act.detach(); act.requires_grad_(True); return act
			return hook

		def f_mlp(layer: int):
			def hook(act, hook=None):
				activations_mlp[layer] = act.detach(); act.requires_grad_(True); return act
			return hook

		def b_head(layer: int):
			def hook(grad, hook=None):
				gradients_head[layer] = grad.detach(); return grad
			return hook

		def b_mlp(layer: int):
			def hook(grad, hook=None):
				gradients_mlp[layer] = grad.detach(); return grad
			return hook

		f_hooks: List[Tuple[str, callable]] = []
		b_hooks: List[Tuple[str, callable]] = []
		for layer in range(n_layers):
			f_hooks.append((f"blocks.{layer}.attn.hook_result", f_head(layer)))
			b_hooks.append((f"blocks.{layer}.attn.hook_result", b_head(layer)))
			f_hooks.append((f"blocks.{layer}.hook_mlp_out", f_mlp(layer)))
			b_hooks.append((f"blocks.{layer}.hook_mlp_out", b_mlp(layer)))

		return f_hooks, b_hooks, activations_head, activations_mlp, gradients_head, gradients_mlp

	def _compute_scores(self, act_h, act_m, grad_h, grad_m) -> Dict[Component, float]:
		scores: Dict[Component, float] = {}
		for layer, act in act_h.items():
			grad = grad_h.get(layer)
			if grad is None: continue   
			b, p, n_head, h = act.shape
			prod = (act * grad).abs().sum(dim=(0,1,3))  # [n_head]
			for head in range(n_head):
				scores[Component(layer, 'head', head)] = float(prod[head].item())
	
		for layer, act in act_m.items():
			grad = grad_m.get(layer)
			if grad is None: continue
			scores[Component(layer, 'mlp', 0)] = float((act * grad).abs().sum().item())

		return scores

	def extract_circuit(self, prompt: str, target: str) -> Set[Component]:
		device = self.model.cfg.device
		prompt_tokens = self.model.to_tokens(prompt, prepend_bos=True).to(device)
		target_id = self.model.to_tokens(target, prepend_bos=False)[0, 0].item()
		f_hooks, b_hooks, act_h, act_m, grad_h, grad_m = self._register_hooks()
		self.model.zero_grad()
		with self.model.hooks(f_hooks + b_hooks):
			logits = self.model(prompt_tokens)
			last_pos = logits.size(1) - 1
			loss = -logits[0, last_pos, target_id]
			loss.backward()
		scores = self._compute_scores(act_h, act_m, grad_h, grad_m)
		items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

		if self.top_k is not None:
			items = items[:self.top_k]
		return {c for c,_ in items}

	def extract_circuit_ig(self, prompt: str, target: str, baseline_prompt: Optional[str] = None, steps: int = 5) -> Set[Component]:
		"""Integrated gradients style circuit extraction.

		Interpolate embeddings between a baseline (zeros or baseline_prompt) and the real prompt.
		Accumulate gradient*activation scores for heads & MLPs across steps and average.
		Falls back to single-step gradient method when steps < 2.
		"""
		if steps < 2:
			return self.extract_circuit(prompt, target)

		device = self.model.cfg.device
		prompt_tokens = self.model.to_tokens(prompt, prepend_bos=True).to(device)
		with torch.no_grad():
			prompt_embed = self.model.embed(prompt_tokens)  # [B,S,d]
		if baseline_prompt is None:
			baseline_embed = torch.zeros_like(prompt_embed)
		else:
			baseline_tokens = self.model.to_tokens(baseline_prompt, prepend_bos=True).to(device)
			with torch.no_grad():
				baseline_embed = self.model.embed(baseline_tokens)
		diff_embed = prompt_embed - baseline_embed
		target_id = self.model.to_tokens(target, prepend_bos=False)[0, 0].item()

		embed_name = "hook_embed"  # standard name in transformer-lens
		agg: Dict[Component, float] = {}
		for step in range(1, steps + 1):
			alpha = step / steps
			interp_embed = (baseline_embed + alpha * diff_embed).detach().requires_grad_(False)
			f_hooks, b_hooks, act_h, act_m, grad_h, grad_m = self._register_hooks()
			def replace_embedding(act, hook=None):  # act: original embedding
				return interp_embed
			self.model.zero_grad()
			with self.model.hooks([(embed_name, replace_embedding)] + f_hooks + b_hooks):
				logits = self.model(prompt_tokens)
				last_pos = logits.size(1) - 1
				loss = -logits[0, last_pos, target_id]
				loss.backward()
			scores = self._compute_scores(act_h, act_m, grad_h, grad_m)
			for c, v in scores.items():
				agg[c] = agg.get(c, 0.0) + v
			# cleanup references to reduce memory
			del loss, logits, scores
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		# Average
		for c in agg:
			agg[c] /= steps
		items = sorted(agg.items(), key=lambda x: x[1], reverse=True)
		if self.top_k is not None:
			items = items[: self.top_k]
		return {c for c, _ in items}

def compute_shared_circuit(circuits: List[Set[Component]]) -> Set[Component]:
	if not circuits: 
		return set()
	shared = set(circuits[0])
	for c in circuits[1:]:
		shared.intersection_update(c)
	return shared

__all__ = ["Component", "CircuitExtractor", "compute_shared_circuit"]
