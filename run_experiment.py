import argparse
import time
from typing import List, Dict, Any, Tuple
import json
import math
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
import random
import hashlib

import torch
from models.olmo_adapter import load_model_any

from circuit_reuse.dataset import get_dataset, Example
from circuit_reuse.circuit_extraction import (
    CircuitExtractor,
    compute_shared_circuit,
    select_shared_by_proportion,
    circuit_identifiability_score,
    Component,
)
from circuit_reuse.evaluate import (
    evaluate_accuracy,
    evaluate_accuracy_with_ablation,
    evaluate_predictions,
)


def _default_run_name():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _prepare_run_dir(output_dir: str, run_name: str | None):
    """
    If output_dir already ends with run_name, use it as-is.
    Otherwise, append run_name (or a fresh timestamp when run_name is None).
    """
    base = Path(output_dir)
    if run_name and base.name == run_name:
        run_dir = base
    elif run_name and run_name.strip():
        run_dir = base / run_name
    else:
        run_dir = base / _default_run_name()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circuit reuse experiment (single-run) with identifiability score and saved artifacts.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")
    parser.add_argument("--hf-revision", type=str, default=None, help="HF revision tag or commit (e.g., some-step-tag).")
    parser.add_argument("--task", type=str, required=True, help="Task name.")
    parser.add_argument("--num_examples", type=int, required=True, help="Number of examples to generate/use.")
    parser.add_argument("--digits", type=int, default=None, help="Digit count (only for addition).")
    parser.add_argument("--top_k", type=int, required=True, help="Size of the shared circuit K (new semantics).")
    parser.add_argument("--example_k", type=int, default=0, help="Per-example top-k used to form example circuits. 0 means keep all scored components.")
    parser.add_argument("--method", type=str, default="eap", choices=["eap", "gradient"], help="Attribution method.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "float16", "float32"], help="Load dtype.")
    parser.add_argument("--log-mem", action="store_true", help="Print CUDA memory after the run.")
    parser.add_argument("--amp", action="store_true", help="Use autocast (mixed precision) during extraction.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Holdout fraction for validation.")
    parser.add_argument("--perm-trials", type=int, default=5000, help="Permutation test trials for shared vs control.")
    return parser.parse_args()


def _enumerate_all_components(model) -> List[Component]:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    comps: List[Component] = []
    for layer in range(n_layers):
        for h in range(n_heads):
            comps.append(Component(layer=layer, kind="head", index=h))
        comps.append(Component(layer=layer, kind="mlp", index=0))
    return comps


def _permutation_test(shared_flags: List[int], control_flags: List[int], trials: int = 5000, rng: random.Random | None = None) -> Dict[str, Any]:
    """
    shared_flags/control_flags are 0/1 correct indicators aligned per example.

    Ccomputes the observed difference in means: diff = mean(control) - mean(shared).
    Under the null, we swap each paired observation and recompute the diff.
    P-value is the fraction of permuted diffs whose absolute value >= |observed|.
    """
    assert len(shared_flags) == len(control_flags)
    n = len(shared_flags)
    if n == 0:
        return {"p_value": 1.0, "obs_diff": 0.0, "trials": 0}

    rng = rng or random.Random(12345)
    obs = (sum(control_flags) / n) - (sum(shared_flags) / n)

    exceed = 0
    for _ in range(max(0, trials)):
        s = shared_flags[:]
        c = control_flags[:]
        # Paired random swap
        for i in range(n):
            if rng.random() < 0.5:
                s[i], c[i] = c[i], s[i]
        diff = (sum(c) / n) - (sum(s) / n)
        if abs(diff) >= abs(obs):
            exceed += 1

    p = (exceed + 1) / (trials + 1) if trials > 0 else 1.0
    return {"p_value": float(p), "obs_diff": float(obs), "trials": int(trials)}


def _save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _run_single_combination(
    model, model_name: str, task: str, num_examples: int, digits: int | None,
    k_shared: int, example_k: int, device: str, debug: bool, run_dir: Path, amp: bool,
    val_fraction: float, method: str, hf_revision: str | None, perm_trials: int,
):
    dataset = get_dataset(task, num_examples=num_examples, digits=digits if digits is not None else 0)
    print(f"[{model_name}/{task}] Generated {len(dataset)} examples for method '{method}'.")

    # Extract attributions for training examples only
    examples = list(dataset)
    random.shuffle(examples)
    n = len(examples)
    vf = max(0.0, min(0.9, val_fraction))
    val_count = int(round(vf * n)) if vf > 0 else 0
    if val_count >= n and n > 1:
        val_count = n - 1
    train_examples = examples[: n - val_count]
    val_examples = examples[n - val_count :]
    if debug:
        print(f"[SPLIT] total={n} train={len(train_examples)} val={len(val_examples)} (val_fraction={vf:.2f})")

    # Per-example top-k (example_k) controls the sets used to compute proportions
    extractor = CircuitExtractor(model, top_k=(None if example_k <= 0 else example_k), method=method)

    combo_key = f"{model_name}|{hf_revision or 'none'}|{task}|{method}|n{num_examples}|d{digits}|K{k_shared}|ek{example_k}"
    start = time.time()

    example_sets: List[set] = []
    example_scores: List[Dict[Component, float]] = []
    try:
        example_sets, example_scores = extractor.extract_circuits_from_examples(
            examples=train_examples,
            task_name=task,
            amp=amp,
            device=device
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] Skipping attribution for {combo_key}: {e}")
    except Exception as e:
        print(f"[ERROR] Circuit extraction failed for {combo_key}: {e}")

    # Select top K shared components by proportion and mean score
    selected_shared, prop_map = select_shared_by_proportion(example_sets, example_scores, k_shared=k_shared)
    cis = circuit_identifiability_score(prop_map)
    end = time.time()

    # Evaluate baseline and ablations on train/val
    baseline_train_correct, baseline_train_total = evaluate_accuracy(model, train_examples, task=task, verbose=debug)
    ablation_train_correct, ablation_train_total = evaluate_accuracy_with_ablation(model, train_examples, task=task, removed=selected_shared, verbose=debug)
    if val_examples:
        baseline_val_correct, baseline_val_total = evaluate_accuracy(model, val_examples, task=task, verbose=debug)
        ablation_val_correct, ablation_val_total = evaluate_accuracy_with_ablation(model, val_examples, task=task, removed=selected_shared, verbose=debug)
    else:
        baseline_val_correct = baseline_val_total = ablation_val_correct = ablation_val_total = 0

    baseline_train_acc = baseline_train_correct / baseline_train_total if baseline_train_total > 0 else 0.0
    ablation_train_acc = ablation_train_correct / ablation_train_total if ablation_train_total > 0 else 0.0
    baseline_val_acc = baseline_val_correct / baseline_val_total if baseline_val_total > 0 else float("nan")
    ablation_val_acc = ablation_val_correct / ablation_val_total if ablation_val_total > 0 else float("nan")

    # Random control ablation with same size
    all_components = _enumerate_all_components(model)
    k = min(len(selected_shared), len(all_components))
    rng_seed = int(hashlib.md5(combo_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(rng_seed)
    control_removed = rng.sample(all_components, k) if k > 0 else []
    print(f"[{task}] Shared K={len(selected_shared)}; Control uses {len(control_removed)}/{len(all_components)} random comps (seed={rng_seed}).")

    control_train_correct, control_train_total = evaluate_accuracy_with_ablation(model, train_examples, task=task, removed=control_removed, verbose=debug)
    control_train_acc = control_train_correct / control_train_total if control_train_total > 0 else 0.0
    if val_examples:
        control_val_correct, control_val_total = evaluate_accuracy_with_ablation(model, val_examples, task=task, removed=control_removed, verbose=debug)
        control_val_acc = control_val_correct / control_val_total if control_val_total > 0 else 0.0
    else:
        control_val_correct = control_val_total = 0
        control_val_acc = float("nan")

    # Knockout diff: (baseline - shared) / (baseline - control)
    def _safe_div(a: float, b: float) -> float:
        denom = b if abs(b) > 1e-12 else (1e-12 if b >= 0 else -1e-12)
        return float(a / denom)

    knockout_train = _safe_div(baseline_train_acc - ablation_train_acc, baseline_train_acc - control_train_acc)
    knockout_val = _safe_div(baseline_val_acc - ablation_val_acc, baseline_val_acc - control_val_acc) if not math.isnan(baseline_val_acc) else float("nan")

    # Per-example predictions for permutation test (train split)
    _, _, preds_shared = evaluate_predictions(model, train_examples, task=task, removed=selected_shared, verbose=debug)
    _, _, preds_control = evaluate_predictions(model, train_examples, task=task, removed=control_removed, verbose=debug)
    shared_flags = [int(r["is_correct"]) for r in preds_shared]
    control_flags = [int(r["is_correct"]) for r in preds_control]
    perm = _permutation_test(shared_flags, control_flags, trials=perm_trials, rng=random.Random(rng_seed + 1))

    # Save artifacts
    # 1) metrics.json (summary)
    metrics = {
        "model_name": model_name,
        "hf_revision": hf_revision,
        "task": task,
        "num_examples": len(dataset),
        "digits": digits if task == "addition" else None,
        "method": method,
        "top_k": k_shared,                     # shared K (new semantics)
        "example_k": (None if example_k <= 0 else example_k),
        "val_fraction": vf,
        "extraction_seconds": end - start,

        "circuit_identifiability_score": cis,
        "shared_circuit_size": len(selected_shared),
        "shared_circuit_components": [str(c) for c in sorted(selected_shared, key=lambda c: (c.layer, c.kind, c.index))],
        "shared_component_proportions": {str(c): float(p) for c, p in sorted(prop_map.items(), key=lambda x: (x[1], x[0].layer, x[0].kind, x[0].index), reverse=True)},

        "baseline_train_accuracy": baseline_train_acc,
        "baseline_train_correct": baseline_train_correct,
        "baseline_train_total": baseline_train_total,
        "ablation_train_accuracy": ablation_train_acc,
        "ablation_train_correct": ablation_train_correct,
        "ablation_train_total": ablation_train_total,
        "control_train_accuracy": control_train_acc,
        "control_train_correct": control_train_correct,
        "control_train_total": control_train_total,

        "baseline_val_accuracy": baseline_val_acc,
        "baseline_val_correct": baseline_val_correct,
        "baseline_val_total": baseline_val_total,
        "ablation_val_accuracy": ablation_val_acc,
        "ablation_val_correct": ablation_val_correct,
        "ablation_val_total": ablation_val_total,
        "control_val_accuracy": control_val_acc,
        "control_val_correct": control_val_correct,
        "control_val_total": control_val_total,

        "accuracy_drop_train": baseline_train_acc - ablation_train_acc,
        "accuracy_drop_val": (baseline_val_acc - ablation_val_acc) if not math.isnan(baseline_val_acc) else float("nan"),
        "control_accuracy_drop_train": baseline_train_acc - control_train_acc,
        "control_accuracy_drop_val": (baseline_val_acc - control_val_acc) if not math.isnan(baseline_val_acc) else float("nan"),

        "knockout_diff_train": knockout_train,
        "knockout_diff_val": knockout_val,

        "control_removed_components": [str(c) for c in sorted(control_removed, key=lambda c: (c.layer, c.kind, c.index))],
        "control_rng_seed": rng_seed,
        "control_total_component_count": len(all_components),

        "perm_obs_diff": perm["obs_diff"],
        "perm_trials": perm["trials"],
        "perm_p_value": perm["p_value"],
    }

    combo_dir = run_dir
    combo_dir.mkdir(parents=True, exist_ok=True)

    with (combo_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"[METRICS] {combo_dir/'metrics.json'}")

    # 2) per-example predictions for baseline, shared, control on train split
    _, _, preds_base = evaluate_predictions(model, train_examples, task=task, removed=None, verbose=debug)
    preds_rows = []
    for i in range(len(train_examples)):
        row = {
            "index": i,
            "prompt": train_examples[i].prompt,
            "target": train_examples[i].target,
            "baseline_pred": preds_base[i].get("pred", preds_base[i].get("pred_token_id")),
            "baseline_correct": bool(preds_base[i]["is_correct"]),
            "shared_pred": preds_shared[i].get("pred", preds_shared[i].get("pred_token_id")),
            "shared_correct": bool(preds_shared[i]["is_correct"]),
            "control_pred": preds_control[i].get("pred", preds_control[i].get("pred_token_id")),
            "control_correct": bool(preds_control[i]["is_correct"]),
        }
        preds_rows.append(row)
    _save_jsonl(combo_dir / "predictions_train.jsonl", preds_rows)
    print(f"[ARTIFACT] predictions_train.jsonl saved.")

    # 3) per-example attribution rankings and scores (train examples)
    attrib_rows = []
    for i, (sset, sc) in enumerate(zip(example_sets, example_scores)):
        ranked = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        attrib_rows.append({
            "index": i,
            "components": [
                {"layer": c.layer, "kind": c.kind, "index": c.index, "score": float(score)}
                for (c, score) in ranked
            ],
            "example_k_used": (None if example_k <= 0 else example_k),
        })
    _save_jsonl(combo_dir / "attributions_train.jsonl", attrib_rows)
    print(f"[ARTIFACT] attributions_train.jsonl saved.")

    print(f"[DONE] {combo_dir}")


def main() -> None:
    args = parse_args()
    base_run_dir = _prepare_run_dir(args.output_dir, args.run_name)
    if args.log_mem:
        print(f"Model: {args.model_name}, Task: {args.task}")

    try:
        print(f"[MODEL LOAD] Loading model {args.model_name} (dtype={args.dtype}) on {args.device}...")
        dtype_map = {"bf16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = None if args.dtype == "auto" else dtype_map[args.dtype]
        model = load_model_any(args.model_name, device=args.device, torch_dtype=torch_dtype, revision=args.hf_revision)
        model.eval()
        if args.log_mem and torch.cuda.is_available():
            print(f"[MEM] post-load allocated={torch.cuda.memory_allocated()/1e9:.2f}GiB reserved={torch.cuda.memory_reserved()/1e9:.2f}GiB")

        digits = args.digits if args.task == "addition" else None
        combo_name = f"{args.model_name.replace('/', '_')}__{args.hf_revision or 'main'}__{args.task}__{args.method}__n{args.num_examples}__d{digits if args.task=='addition' else 'na'}__K{args.top_k}__ek{args.example_k or 0}"
        run_dir = base_run_dir / combo_name
        print(f"\n[RUN] {combo_name}")

        _run_single_combination(
            model=model,
            model_name=args.model_name,
            task=args.task,
            num_examples=args.num_examples,
            digits=digits,
            k_shared=args.top_k,
            example_k=args.example_k,
            device=args.device,
            debug=args.debug,
            run_dir=run_dir,
            amp=args.amp,
            val_fraction=args.val_fraction,
            method=args.method,
            hf_revision=args.hf_revision,
            perm_trials=args.perm_trials,
        )

    except Exception as e:
        print(f"[FATAL] {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            model.reset_hooks()
        except Exception:
            pass
        del model
        torch.cuda.empty_cache()
        print(f"[ALL DONE] Results root: {base_run_dir}")


if __name__ == "__main__":
    main()
