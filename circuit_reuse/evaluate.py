from __future__ import annotations

from typing import Iterable, List, Tuple, Any, Dict
import torch
from .dataset import Example
from .circuit_extraction import Component
from contextlib import nullcontext


def _extract_gold_ids(model: Any, prompt: str, target: str, device, verbose: bool = False) -> List[int]:
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
            print(f"[WARN] No divergent token boundary; fallback standalone target tokenization target='{target}' ids={fallback_ids}")
        return [int(x) for x in fallback_ids]
    gold_ids = f_ids[lcp:]
    if verbose and not gold_ids:
        print(f"[WARN] Derived empty gold ids unexpectedly; prompt_len={len(p_ids)} full_len={len(f_ids)}")
    return [int(x) for x in gold_ids]


_BOOL_CACHE = {}
_MC_CACHE = {}


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
        print(f"[BOOL-CLASSIFY] true_score={true_score:.3f} false_score={false_score:.3f} label={label}")
    return label, id_logits


def _mc_letter_token_id_groups(model) -> dict:
    cache_key = id(model)
    if cache_key in _MC_CACHE:
        return _MC_CACHE[cache_key]
    out = {}
    # Use all uppercase letters to handle various MCQA formats (e.g., up to 10 choices, random letters)
    for i in range(26):
        L = chr(ord("A") + i)
        ids = set()
        for variant in (L, f" {L}", f"\n{L}", f": {L}", f":\n{L}"):
            toks = model.to_tokens(variant, prepend_bos=False)
            t_ids = toks[0].tolist()
            if len(t_ids) == 1:
                ids.add(int(t_ids[0]))
        if ids:  # Only add letters that tokenize to a single token
            out[L] = ids
    _MC_CACHE[cache_key] = out
    return out


def _classify_multiple_choice(logits_last: Any, model, verbose: bool = False) -> str:
    groups = _mc_letter_token_id_groups(model)
    best_letter, best_score = "A", float("-inf")
    for L, id_set in groups.items():
        if not id_set:
            continue
        score = max(float(logits_last[tid].item()) for tid in id_set)
        if score > best_score:
            best_score, best_letter = score, L
    if verbose:
        print(f"[MC-CLASSIFY] scores=" + ", ".join(f"{L}:{'none' if not ids else max(logits_last[i].item() for i in ids):.3f}" for L, ids in groups.items()))
    return best_letter


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
            elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                pred_label = _classify_multiple_choice(logits_last, model, verbose=verbose)
                if pred_label == target:
                    correct += 1
            else:
                gold_ids = _extract_gold_ids(model, prompt, target, device=device, verbose=verbose)
                if gold_ids and int(logits_last.argmax().item()) in gold_ids:
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
            prompt, target = ex.prompt, ex.target
            logits = model(model.to_tokens(prompt, prepend_bos=True).to(device))
            logits_last = logits[0, -1]
            if task == "boolean":
                pred_label, _ = _classify_boolean(logits_last, model, verbose=verbose)
                if pred_label == target:
                    correct += 1
            elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                pred_label = _classify_multiple_choice(logits_last, model, verbose=verbose)
                if pred_label == target:
                    correct += 1
            else:
                gold_ids = _extract_gold_ids(model, prompt, target, device=device, verbose=verbose)
                if gold_ids and int(logits_last.argmax().item()) in gold_ids:
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
        {"prompt": str, "target": str, "pred": str, "is_correct": bool}
      For generative tasks (e.g., addition), "pred" is the best-token id rendered if available.
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
                    correct += int(ok)
                elif task in ("mmlu", "mcqa", "arc_easy", "arc_challenge"):
                    pred_label = _classify_multiple_choice(logits_last, model, verbose=verbose)
                    gold = ex.target
                    ok = (pred_label == gold)
                    per_ex.append({"prompt": ex.prompt, "target": gold, "pred": pred_label, "is_correct": bool(ok)})
                    correct += int(ok)
                else:
                    gold_ids = _extract_gold_ids(model, ex.prompt, ex.target, device=device, verbose=verbose)
                    pred_id = int(logits_last.argmax().item())
                    ok = (gold_ids and pred_id in gold_ids)
                    per_ex.append({"prompt": ex.prompt, "target": ex.target, "pred_token_id": pred_id, "is_correct": bool(ok)})
                    correct += int(ok)
                total += 1

    return correct, total, per_ex


__all__ = ["evaluate_accuracy", "evaluate_accuracy_with_ablation", "evaluate_predictions"]