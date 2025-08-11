from __future__ import annotations

from typing import Iterable, List, Tuple, Any

try:
	import torch  # type: ignore
except Exception:  # noqa: BLE001
	torch = None  # type: ignore

from .dataset import ArithmeticExample  # type: ignore
from .circuit_extraction import Component  # type: ignore


def _is_tensorlike(x) -> bool:
	return hasattr(x, "shape") or hasattr(x, "size")


def _predict_next_token_mock(model: Any, prompt: str) -> int:
	if hasattr(model, "predictions") and isinstance(model.predictions, dict):
		return int(model.predictions.get(prompt, 0))
	return 0


def _extract_gold_ids(model: Any, prompt: str, target: str, device, verbose: bool = False) -> List[int]:
	"""
	Derive gold target token ids robustly even when BPE merges across the prompt/target boundary.

	We tokenize:
	  A = tokens(prompt)
	  B = tokens(prompt + target)
	Then take the longest common prefix length between A and B; the remainder of B are the gold target ids.
	This avoids failures where simple slicing by len(A) yields an empty remainder because the final
	prompt token in A merges with the first answer characters in B.
	"""
	prompt_tok = model.to_tokens(prompt, prepend_bos=True)
	full_tok = model.to_tokens(prompt + target, prepend_bos=True)
	if hasattr(prompt_tok, "to"): prompt_tok = prompt_tok.to(device)
	if hasattr(full_tok, "to"): full_tok = full_tok.to(device)
	p_ids = prompt_tok[0].tolist()
	f_ids = full_tok[0].tolist()
	lcp = 0
	for a,b in zip(p_ids, f_ids):
		if a == b:
			lcp += 1
		else:
			break
	if lcp == len(f_ids):
		# Entire full sequence identical -> fall back to standalone target tokenization
		alt = model.to_tokens(target, prepend_bos=False)
		if hasattr(alt, "to"): alt = alt.to(device)
		fallback_ids = alt[0].tolist() if alt.ndim == 2 else alt.tolist()
		if verbose:
			print(f"[WARN] No divergent token boundary; fallback standalone target tokenization target='{target}' ids={fallback_ids}")
		return [int(x) for x in fallback_ids]
	gold_ids = f_ids[lcp:]
	if verbose and not gold_ids:
		print(f"[WARN] Derived empty gold ids unexpectedly; prompt_len={len(p_ids)} full_len={len(f_ids)}")
	return [int(x) for x in gold_ids]


def _boolean_token_id_groups(model) -> Tuple[set, set]:
	"""Collect single-token ids for (true,false) across common spacing/casing variants."""
	variants_true = [" true", "true", " True", "True"]
	variants_false = [" false", "false", " False", "False"]
	def collect(variants):
		out = set()
		for v in variants:
			try:
				toks = model.to_tokens(v, prepend_bos=False)
				seq = toks[0].tolist() if hasattr(toks, "tolist") else toks
				if isinstance(seq[0], list): seq = seq[0]
				if len(seq) == 1:
					out.add(int(seq[0]))
			except Exception:
				pass
		return out
	return collect(variants_true), collect(variants_false)


def _classify_boolean(logits_last: Any, model, verbose: bool = False) -> Tuple[str, dict]:
	"""Classify between true/false given final-position logits."""
	true_ids, false_ids = _boolean_token_id_groups(model)
	id_logits = {}
	if torch is not None and hasattr(logits_last, "index_select"):
		for tid in true_ids:
			id_logits[f"true:{tid}"] = float(logits_last[tid].item())
		for fid in false_ids:
			id_logits[f"false:{fid}"] = float(logits_last[fid].item())
		true_score = max((logits_last[tid].item() for tid in true_ids), default=float("-inf"))
		false_score = max((logits_last[fid].item() for fid in false_ids), default=float("-inf"))
	else:
		# Fallback (unlikely in real path)
		true_score = false_score = float("-inf")
	label = "true" if true_score >= false_score else "false"
	if verbose:
		print(f"[BOOL-CLASSIFY] true_score={true_score:.3f} false_score={false_score:.3f} label={label} true_ids={sorted(true_ids)} false_ids={sorted(false_ids)}")
	return label, id_logits


def evaluate_accuracy(model: Any, dataset: Iterable[ArithmeticExample], verbose: bool = False) -> float:
	correct = 0
	total = 0
	for ex in dataset:
		prompt_tokens = model.to_tokens(ex.prompt, prepend_bos=True)
		target_tokens = model.to_tokens(ex.target, prepend_bos=False)
		if not _is_tensorlike(target_tokens):
			pred_id = _predict_next_token_mock(model, ex.prompt)
			gold_id = model.to_single_token(ex.target) if hasattr(model, "to_single_token") else pred_id
			ok = pred_id == gold_id
			if verbose:
				print(f"[GREEDY][MOCK] prompt='{ex.prompt}' target='{ex.target}' pred_id={pred_id} gold_id={gold_id} correct={ok}")
			correct += int(ok)
			total += 1
			continue
		if torch is None:
			raise RuntimeError("PyTorch required for real model evaluation.")
		device = getattr(model.cfg, "device", "cpu")
		if hasattr(prompt_tokens, "to"):
			prompt_tokens = prompt_tokens.to(device)
		# --- Boolean classification shortcut ---
		if ex.target.lower() in ("true", "false"):
			logits = model(prompt_tokens)
			logits_last = logits[0, -1]
			pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
			ok = (pred_label == ex.target.lower())
			if verbose:
				print(f"[GREEDY-BOOL] prompt='{ex.prompt}' target='{ex.target}' pred='{pred_label}' correct={ok}")
			correct += int(ok)
			total += 1
			continue
		# --- Standard autoregressive path ---
		gold_ids = _extract_gold_ids(model, ex.prompt, ex.target, device, verbose=verbose)
		current = prompt_tokens
		generated: List[int] = []
		all_match = True
		for gid in gold_ids:
			logits = model(current)
			pred_id = int(logits[0, -1].argmax(dim=-1).item())
			generated.append(pred_id)
			next_tok = torch.tensor([[pred_id]], device=current.device)
			current = torch.cat([current, next_tok], dim=1)
			if pred_id != gid:
				all_match = False
		if verbose:
			try:
				gold_decoded = model.to_string(torch.tensor([gold_ids], device=current.device))
				pred_decoded = model.to_string(torch.tensor([generated], device=current.device))
			except Exception:
				gold_decoded = "".join(str(t) for t in gold_ids)
				pred_decoded = "".join(str(t) for t in generated)
			print(
				f"[GREEDY] prompt='{ex.prompt}' target='{ex.target}' "
				f"gold={gold_ids} gen={generated} correct={all_match} "
				f"gold_text='{gold_decoded}' pred_text='{pred_decoded}'"
			)
		correct += int(all_match)
		total += 1
	return correct / total if total else 0.0


def evaluate_accuracy_with_knockout(model: Any, dataset: Iterable[ArithmeticExample], removed: Iterable[Component], verbose: bool = False) -> float:
	"""Greedy exact-sequence accuracy with specified components zeroed.

	Parameters
	----------
	model : Any
	dataset : Iterable[ArithmeticExample]
	removed : Iterable[Component]
		Components to zero.
	verbose : bool
		If True, print per-example predictions (mirrors evaluate_accuracy).
	"""
	# Build zeroing hooks (real model path only)
	hooks: List[Tuple[str, callable]] = []
	for comp in removed:
		if comp.kind == "head":
			name = f"blocks.{comp.layer}.attn.hook_result"
			head_idx = comp.index
			def make_zero_head_hook(idx: int):
				def hook(act, hook=None):
					act = act.clone()
					act[:, :, idx, :] = 0
					return act
				return hook
			hooks.append((name, make_zero_head_hook(head_idx)))
		elif comp.kind == "mlp":
			name = f"blocks.{comp.layer}.hook_mlp_out"
			def zero_mlp_hook(act, hook=None):
				return torch.zeros_like(act)
			hooks.append((name, zero_mlp_hook))
		else:
			raise ValueError(f"Unknown component type: {comp.kind}")

	correct = 0
	total = 0
	device = getattr(getattr(model, "cfg", object()), "device", "cpu")
	with model.hooks(hooks):
		for ex in dataset:
			prompt_tokens = model.to_tokens(ex.prompt, prepend_bos=True)
			target_tokens = model.to_tokens(ex.target, prepend_bos=False)
			if not _is_tensorlike(target_tokens):
				pred_id = _predict_next_token_mock(model, ex.prompt)
				gold_id = model.to_single_token(ex.target) if hasattr(model, "to_single_token") else pred_id
				ok = pred_id == gold_id
				if verbose:
					print(f"[KNOCKOUT][MOCK] prompt='{ex.prompt}' target='{ex.target}' pred_id={pred_id} gold_id={gold_id} correct={ok}")
				correct += int(ok)
				total += 1
				continue
			if torch is None:
				raise RuntimeError("PyTorch required for knockout on real model.")
			if hasattr(prompt_tokens, "to"):
				prompt_tokens = prompt_tokens.to(device)
			# --- Boolean classification shortcut (knockout) ---
			if ex.target.lower() in ("true", "false"):
				logits = model(prompt_tokens)
				logits_last = logits[0, -1]
				pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
				ok = (pred_label == ex.target.lower())
				if verbose:
					print(f"[KNOCKOUT-BOOL] prompt='{ex.prompt}' target='{ex.target}' pred='{pred_label}' correct={ok}")
				correct += int(ok)
				total += 1
				continue
			# --- Standard autoregressive path ---
			gold_ids = _extract_gold_ids(model, ex.prompt, ex.target, device, verbose=verbose)
			current = prompt_tokens
			generated: List[int] = []
			all_match = True
			for gid in gold_ids:
				logits = model(current)
				pred_id = int(logits[0, -1].argmax(dim=-1).item())
				generated.append(pred_id)
				next_tok = torch.tensor([[pred_id]], device=current.device)
				current = torch.cat([current, next_tok], dim=1)
				if pred_id != gid:
					all_match = False
			if verbose:
				try:
					gold_decoded = model.to_string(torch.tensor([gold_ids], device=current.device))
					pred_decoded = model.to_string(torch.tensor([generated], device=current.device))
				except Exception:
					gold_decoded = "".join(str(t) for t in gold_ids)
					pred_decoded = "".join(str(t) for t in generated)
				print(
					f"[KNOCKOUT] prompt='{ex.prompt}' target='{ex.target}' "
					f"gold={gold_ids} gen={generated} correct={all_match} "
					f"gold_text='{gold_decoded}' pred_text='{pred_decoded}'"
				)
			correct += int(all_match)
			total += 1
	return correct / total if total else 0.0

__all__ = ["evaluate_accuracy", "evaluate_accuracy_with_knockout"]
