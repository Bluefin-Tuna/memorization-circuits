from __future__ import annotations

from typing import Iterable, List, Tuple, Any, Dict, Optional
import torch
from .dataset import Example
from .circuit_extraction import Component
from contextlib import nullcontext


def _extract_gold_ids(model: Any, prompt: str, target: str, device, verbose: bool = False) -> List[int]:
    """Return token ids for the continuation target, relative to the prompt."""
    prompt_tok = model.to_tokens(prompt, prepend_bos=True).to(device)
    full_tok = model.to_tokens(prompt + target, prepend_bos=True).to(device)
    p_ids = prompt_tok[0].tolist()
    f_ids = full_tok[0].tolist()
    lcp = 0
    for a, b in zip(p_ids, f_ids):
        if a == b:
            lcp += 1
        else:
            break
    if lcp == len(f_ids):
        alt = model.to_tokens(target, prepend_bos=False).to(device)
        fallback_ids = alt[0].tolist() if alt.ndim == 2 else alt.tolist()
        if verbose:
            print(f"[WARN] No divergent boundary; fallback target ids={fallback_ids} for target='{target}'")
        return [int(x) for x in fallback_ids]
    gold_ids = f_ids[lcp:]
    if verbose and not gold_ids:
        print(f"[WARN] Empty gold ids; prompt_len={len(p_ids)} full_len={len(f_ids)}")
    return [int(x) for x in gold_ids]


_BOOL_CACHE = {}


def _boolean_token_id_groups(model) -> Tuple[set, set]:
    cache_key = id(model)
    if cache_key in _BOOL_CACHE:
        return _BOOL_CACHE[cache_key]
    variants_true = [" true", "true", " True", "True"]
    variants_false = [" false", "false", " False", "False"]

    def collect(variants):
        out = set()
        for v in variants:
            toks = model.to_tokens(v, prepend_bos=False)
            ids = toks[0].tolist()
            if len(ids) == 1:
                out.add(int(ids[0]))
        return out

    res = (collect(variants_true), collect(variants_false))
    _BOOL_CACHE[cache_key] = res
    return res


def _classify_boolean(logits_last: Any, model, verbose: bool = False) -> Tuple[str, dict]:
    true_ids, false_ids = _boolean_token_id_groups(model)
    id_logits = {f"true:{tid}": float(logits_last[tid].item()) for tid in true_ids}
    id_logits.update({f"false:{fid}": float(logits_last[fid].item()) for fid in false_ids})
    true_score = max((logits_last[tid].item() for tid in true_ids), default=float("-inf"))
    false_score = max((logits_last[fid].item() for fid in false_ids), default=float("-inf"))
    label = "true" if true_score >= false_score else "false"
    if verbose:
        print(f"[BOOL] true={true_score:.3f} false={false_score:.3f} -> {label}")
    return label, id_logits


def _score_first_token(logits_last, model, label: str) -> float:
    """Score a label by the best logit among plausible first-token variants."""
    ids = set()
    for v in (label, f" {label}", f"\n{label}", f": {label}", f":\n{label}"):
        toks = model.to_tokens(v, prepend_bos=False)
        t_ids = toks[0].tolist()
        if len(t_ids) >= 1:
            ids.add(int(t_ids[0]))
    return max((float(logits_last[i].item()) for i in ids), default=float("-inf"))


def _classify_from_labels(logits_last, model, labels: List[str]) -> str:
    scores = [(_score_first_token(logits_last, model, L), L) for L in labels]
    return max(scores, key=lambda x: x[0])[1]


def _classify_ioi(logits_last, model, names: List[str]) -> int:
    """Return index of the predicted name among the two candidates."""
    scores = [(_score_first_token(logits_last, model, n), i) for i, n in enumerate(names)]
    return max(scores, key=lambda x: x[0])[1]


def evaluate_accuracy(model: Any, dataset: Iterable[Example], task: str, verbose: bool = False) -> Tuple[int, int]:
    model.eval()
    correct, total = 0, 0
    device = model.cfg.device
    with torch.inference_mode():
        for ex in dataset:
            prompt, target = ex.prompt, ex.target
            logits = model(model.to_tokens(prompt, prepend_bos=True).to(device))
            logits_last = logits[0, -1]

            if task == "boolean":
                pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
                if pred_label == target:
                    correct += 1

            elif task == "ioi":
                labels = ex.labels or [ex.target, ex.corrupted_target]
                pred_idx = _classify_ioi(logits_last, model, labels)
                gold_idx = ex.answer_idx if ex.answer_idx is not None else 0
                if pred_idx == gold_idx:
                    correct += 1

            elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                labels = ex.labels or ["A", "B", "C", "D"]
                pred_label = _classify_from_labels(logits_last, model, labels)
                if pred_label == target:
                    correct += 1

            else:
                gold_ids = _extract_gold_ids(model, prompt, target, device=device, verbose=verbose)
                pred_id = int(logits_last.argmax().item())
                # Compare only to the first continuation token for deterministic scoring
                if gold_ids and pred_id == gold_ids[0]:
                    correct += 1

            total += 1
    return correct, total


def evaluate_accuracy_with_ablation(
    model: Any, dataset: Iterable[Example], task: str, removed: Iterable[Component], verbose: bool = False
) -> Tuple[int, int]:
    model.eval()
    hooks: List[Tuple[str, callable]] = []
    for comp in removed:
        if comp.kind == "head":
            def hook_head(act, hook=None, head_index=comp.index):
                act[:, :, head_index, :] = 0.0
                return act
            hooks.append((f"blocks.{comp.layer}.attn.hook_result", hook_head))
        elif comp.kind == "mlp":
            def hook_mlp(act, hook=None):
                act[:, :, :] = 0.0
                return act
            hooks.append((f"blocks.{comp.layer}.hook_mlp_out", hook_mlp))

    correct, total = 0, 0
    device = model.cfg.device
    with torch.inference_mode(), model.hooks(hooks):
        for ex in dataset:
            logits = model(model.to_tokens(ex.prompt, prepend_bos=True).to(device))
            logits_last = logits[0, -1]

            if task == "boolean":
                pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
                if pred_label == ex.target:
                    correct += 1

            elif task == "ioi":
                labels = ex.labels or [ex.target, ex.corrupted_target]
                pred_idx = _classify_ioi(logits_last, model, labels)
                gold_idx = ex.answer_idx if ex.answer_idx is not None else 0
                if pred_idx == gold_idx:
                    correct += 1

            elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                labels = ex.labels or ["A", "B", "C", "D"]
                pred_label = _classify_from_labels(logits_last, model, labels)
                if pred_label == ex.target:
                    correct += 1

            else:
                gold_ids = _extract_gold_ids(model, ex.prompt, ex.target, device=device, verbose=verbose)
                pred_id = int(logits_last.argmax().item())
                if gold_ids and pred_id == gold_ids[0]:
                    correct += 1

            total += 1
    return correct, total


def evaluate_predictions(
    model: Any,
    dataset: Iterable[Example],
    task: str,
    removed: Iterable[Component] | None = None,
    verbose: bool = False,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Evaluate and also return per-example predictions.

    Returns:
      correct, total, per_example list with:
        {"prompt": str, "target": str, "pred": str | int, "is_correct": bool}
    """
    model.eval()
    hooks: List[Tuple[str, callable]] = []

    if removed:
        for comp in removed:
            if comp.kind == "head":
                def hook_head(act, hook=None, head_index=comp.index):
                    if act.dim() == 4:
                        act[:, :, head_index, :] = 0.0
                    return act
                hooks.append((f"blocks.{comp.layer}.attn.hook_result", hook_head))
            elif comp.kind == "mlp":
                def hook_mlp(act, hook=None):
                    act[:, :, :] = 0.0
                    return act
                hooks.append((f"blocks.{comp.layer}.hook_mlp_out", hook_mlp))

    per_ex: List[Dict[str, Any]] = []
    correct, total = 0, 0
    device = model.cfg.device
    ctx = model.hooks(fwd_hooks=hooks) if hooks else nullcontext()

    with ctx:
        with torch.inference_mode():
            for ex in dataset:
                logits = model(model.to_tokens(ex.prompt, prepend_bos=True).to(device))
                logits_last = logits[0, -1]

                if task == "boolean":
                    pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
                    gold = ex.target
                    ok = (pred_label == gold)
                    per_ex.append({"prompt": ex.prompt, "target": gold, "pred": pred_label, "is_correct": bool(ok)})

                elif task == "ioi":
                    labels = ex.labels or [ex.target, ex.corrupted_target]
                    pred_idx = _classify_ioi(logits_last, model, labels)
                    gold_idx = ex.answer_idx if ex.answer_idx is not None else 0
                    ok = (pred_idx == gold_idx)
                    pred_name = labels[pred_idx] if 0 <= pred_idx < len(labels) else str(pred_idx)
                    gold_name = labels[gold_idx] if 0 <= gold_idx < len(labels) else str(gold_idx)
                    per_ex.append({"prompt": ex.prompt, "target": gold_name, "pred": pred_name, "is_correct": bool(ok)})

                elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                    labels = ex.labels or ["A", "B", "C", "D"]
                    pred_label = _classify_from_labels(logits_last, model, labels)
                    gold = ex.target
                    ok = (pred_label == gold)
                    per_ex.append({"prompt": ex.prompt, "target": gold, "pred": pred_label, "is_correct": bool(ok)})

                else:
                    gold_ids = _extract_gold_ids(model, ex.prompt, ex.target, device=device, verbose=verbose)
                    pred_id = int(logits_last.argmax().item())
                    ok = (gold_ids and pred_id == gold_ids[0])
                    per_ex.append({"prompt": ex.prompt, "target": ex.target, "pred_token_id": pred_id, "is_correct": bool(ok)})

                correct += int(per_ex[-1]["is_correct"])
                total += 1

    return correct, total, per_ex


__all__ = ["evaluate_accuracy", "evaluate_accuracy_with_ablation", "evaluate_predictions"]
