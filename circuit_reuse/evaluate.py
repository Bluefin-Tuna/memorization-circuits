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


# Simple caches to avoid repeated tokenization per example
_BOOL_CACHE = {}
_MC_CACHE = {}

def _boolean_token_id_groups(model) -> Tuple[set, set]:
	"""Collect single-token ids for (true,false) across common spacing/casing variants (cached per model)."""
	cache_key = id(model)
	if cache_key in _BOOL_CACHE:
		return _BOOL_CACHE[cache_key]
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
	res = (collect(variants_true), collect(variants_false))
	_BOOL_CACHE[cache_key] = res
	return res


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


def _mc_letter_token_id_groups(model) -> dict:
	"""Collect possible single-token ids for letters A-D across spacing variants (cached per model)."""
	cache_key = id(model)
	if cache_key in _MC_CACHE:
		return _MC_CACHE[cache_key]
	letters = "ABCD"
	variants = lambda L: [L, f" {L}", f"\n{L}"]
	out = {}
	for L in letters:
		ids = set()
		for v in variants(L):
			try:
				toks = model.to_tokens(v, prepend_bos=False)
				seq = toks[0].tolist() if hasattr(toks, "tolist") else toks
				if isinstance(seq[0], list): seq = seq[0]
				if len(seq) == 1:
					ids.add(int(seq[0]))
			except Exception:
				pass
		out[L] = ids
	_MC_CACHE[cache_key] = out
	return out


def _classify_multiple_choice(logits_last: Any, model, verbose: bool = False) -> str:
	"""Select letter A-D with highest logit (max over variant token ids)."""
	groups = _mc_letter_token_id_groups(model)
	best_letter = "A"
	best_score = float("-inf")
	for L, id_set in groups.items():
		if not id_set:
			continue
		score = max(float(logits_last[tid].item()) for tid in id_set)
		if score > best_score:
			best_score = score
			best_letter = L
	if verbose:
		print(f"[MC-CLASSIFY] scores=" + ", ".join(
			f"{L}:{'none' if not ids else max(logits_last[i].item() for i in ids):.3f}" for L, ids in groups.items()
		))
	return best_letter


def evaluate_accuracy(model: Any, dataset: Iterable[ArithmeticExample], task: str, verbose: bool = False) -> float:
	# Ensure inference mode (no gradients)
	if hasattr(model, "eval"): model.eval()
	correct = 0
	total = 0
	if torch is not None:
		# Use a single no_grad scope for full evaluation
		with torch.no_grad():
			for ex in dataset:
				prompt_tokens = model.to_tokens(ex.prompt, prepend_bos=True)
				target_tokens = model.to_tokens(ex.target, prepend_bos=False)
				if not _is_tensorlike(target_tokens):
					# mock path unchanged
					pred_id = _predict_next_token_mock(model, ex.prompt)
					gold_id = model.to_single_token(ex.target) if hasattr(model, "to_single_token") else pred_id
					ok = pred_id == gold_id
					if verbose:
						print(f"[GREEDY][MOCK] prompt='{ex.prompt}' target='{ex.target}' pred_id={pred_id} gold_id={gold_id} correct={ok}")
					correct += int(ok); total += 1; continue
				if torch is None:
					raise RuntimeError("PyTorch required for real model evaluation.")
				device = getattr(model.cfg, "device", "cpu")
				if hasattr(prompt_tokens, "to"):
					prompt_tokens = prompt_tokens.to(device)

				# --- Task-specific branches ---
				if task == "boolean":
					logits = model(prompt_tokens)
					logits_last = logits[0, -1]
					pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
					ok = (pred_label == ex.target.lower())
					if verbose:
						print(f"[BOOLEAN] target='{ex.target}' pred='{pred_label}' correct={ok}")
					correct += int(ok); total += 1; continue

				if task == "mmlu" or task.startswith("mib_"):
					logits = model(prompt_tokens)
					logits_last = logits[0, -1]
					pred_letter = _classify_multiple_choice(logits_last, model, verbose=verbose)
					ok = (pred_letter == ex.target)
					if verbose:
						print(f"[MC] target='{ex.target}' pred='{pred_letter}' correct={ok}")
					correct += int(ok); total += 1; continue

				# --- Default autoregressive path (e.g., addition) ---
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
					print(f"[AR] target='{ex.target}' gold={gold_ids} gen={generated} correct={all_match} gold_text='{gold_decoded}' pred_text='{pred_decoded}'")
				correct += int(all_match); total += 1
		return correct / total if total else 0.0
	# Fallback path (torch None): original loop (unchanged)
	for ex in dataset:
		prompt_tokens = model.to_tokens(ex.prompt, prepend_bos=True)
		target_tokens = model.to_tokens(ex.target, prepend_bos=False)
		if not _is_tensorlike(target_tokens):
			# mock path unchanged
			pred_id = _predict_next_token_mock(model, ex.prompt)
			gold_id = model.to_single_token(ex.target) if hasattr(model, "to_single_token") else pred_id
			ok = pred_id == gold_id
			if verbose:
				print(f"[GREEDY][MOCK] prompt='{ex.prompt}' target='{ex.target}' pred_id={pred_id} gold_id={gold_id} correct={ok}")
			correct += int(ok); total += 1; continue
		# --- Task-specific branches ---
		if task == "boolean":
			logits = model(prompt_tokens)
			logits_last = logits[0, -1]
			pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
			ok = (pred_label == ex.target.lower())
			if verbose:
				print(f"[BOOLEAN] target='{ex.target}' pred='{pred_label}' correct={ok}")
			correct += int(ok); total += 1; continue

		if task == "mmlu" or task.startswith("mib_"):
			logits = model(prompt_tokens)
			logits_last = logits[0, -1]
			pred_letter = _classify_multiple_choice(logits_last, model, verbose=verbose)
			ok = (pred_letter == ex.target)
			if verbose:
				print(f"[MC] target='{ex.target}' pred='{pred_letter}' correct={ok}")
			correct += int(ok); total += 1; continue

		# --- Default autoregressive path (e.g., addition) ---
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
			print(f"[AR] target='{ex.target}' gold={gold_ids} gen={generated} correct={all_match} gold_text='{gold_decoded}' pred_text='{pred_decoded}'")
		correct += int(all_match); total += 1
	return correct / total if total else 0.0


def evaluate_accuracy_with_ablation(model: Any, dataset: Iterable[ArithmeticExample], task: str, removed: Iterable[Component], verbose: bool = False) -> float:
	"""Evaluate accuracy under component ablation for a specified task."""
	if hasattr(model, "eval"): model.eval()
	hooks: List[Tuple[str, callable]] = []
	for comp in removed:
		if comp.kind == "head":
			name = f"blocks.{comp.layer}.attn.hook_result"
			head_idx = comp.index
			def make_zero_head_hook(idx: int):
				def hook(act, hook=None):
					act[:, :, idx, :].zero_()
					return act
				return hook
			hooks.append((name, make_zero_head_hook(head_idx)))
		elif comp.kind == "mlp":
			name = f"blocks.{comp.layer}.hook_mlp_out"
			def zero_mlp_hook(act, hook=None):
				return act.zero_()
			hooks.append((name, zero_mlp_hook))
		else:
			raise ValueError(f"Unknown component type: {comp.kind}")

	correct = 0
	total = 0
	device = getattr(getattr(model, "cfg", object()), "device", "cpu")
	if torch is not None:
		with torch.no_grad():
			with model.hooks(hooks):
				for ex in dataset:
					prompt_tokens = model.to_tokens(ex.prompt, prepend_bos=True)
					target_tokens = model.to_tokens(ex.target, prepend_bos=False)
					if not _is_tensorlike(target_tokens):
						pred_id = _predict_next_token_mock(model, ex.prompt)
						gold_id = model.to_single_token(ex.target) if hasattr(model, "to_single_token") else pred_id
						ok = pred_id == gold_id
						if verbose:
							print(f"[ABLATION][MOCK] prompt='{ex.prompt}' target='{ex.target}' pred_id={pred_id} gold_id={gold_id} correct={ok}")
						correct += int(ok); total += 1; continue
					if hasattr(prompt_tokens, "to"):
						prompt_tokens = prompt_tokens.to(device)
					if task == "boolean":
						logits = model(prompt_tokens)
						logits_last = logits[0, -1]
						pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
						ok = (pred_label == ex.target.lower())
						if verbose:
							print(f"[ABLATION-BOOLEAN] target='{ex.target}' pred='{pred_label}' correct={ok}")
						correct += int(ok); total += 1; continue
					if task == "mmlu" or task.startswith("mib_"):
						logits = model(prompt_tokens)
						logits_last = logits[0, -1]
						pred_letter = _classify_multiple_choice(logits_last, model, verbose=verbose)
						ok = (pred_letter == ex.target)
						if verbose:
							print(f"[ABLATION-MC] target='{ex.target}' pred='{pred_letter}' correct={ok}")
						correct += int(ok); total += 1; continue
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
						print(f"[ABLATION-AR] target='{ex.target}' gold={gold_ids} gen={generated} correct={all_match}")
					correct += int(all_match); total += 1
		return correct / total if total else 0.0
	# Non-torch fallback unchanged (would remain zero if reached)
	return correct / total if total else 0.0

__all__ = ["evaluate_accuracy", "evaluate_accuracy_with_ablation"]