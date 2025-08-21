import argparse
import time
from typing import List, Dict, Any, Tuple, Iterable
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


def _parse_int_list(s: str | None) -> List[int]:
    if s is None:
        return []
    if isinstance(s, list):  # already parsed
        return [int(x) for x in s]
    return [int(x.strip()) for x in str(s).replace(";", ",").split(",") if x is not None and str(x).strip() != ""]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circuit reuse experiment (single-run) with reuse@p metrics and saved artifacts.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")
    parser.add_argument("--hf-revision", type=str, default=None, help="HF revision tag or commit (e.g., some-step-tag).")
    parser.add_argument("--task", type=str, required=True, help="Task name.")
    parser.add_argument("--num_examples", type=int, required=True, help="Number of examples to generate/use.")
    parser.add_argument("--digits", type=int, default=None, help="Digit count (only for addition).")
    parser.add_argument(
        "--top_k_list",
        type=str,
        required=True,
        help="Comma-separated list of per-example top-K values to evaluate (e.g., '5,10,25,50').",
    )
    parser.add_argument(
        "--reuse-thresholds",
        type=str,
        default="95,96,97,98,99,100",
        help="Comma-separated list of reuse thresholds p as percentages (e.g., '95,99,100' for reuse@95..).",
    )
    parser.add_argument("--method", type=str, default="eap", choices=["eap", "gradient"], help="Attribution method.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "float16", "float32"], help="Load dtype.")
    parser.add_argument("--log-mem", action="store_true", help="Print CUDA memory after the run.")
    parser.add_argument("--amp", action="store_true", help="Use autocast (mixed precision) during extraction.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Holdout fraction for validation.")
    parser.add_argument("--perm-trials", type=int, default=5000, help="Trials for paired permutation test between shared vs control ablations.")
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


def _build_topk_example_sets(per_example_scores: List[Dict[Component, float]], k: int) -> List[set]:
    sets: List[set] = []
    for sc in per_example_scores:
        ranked = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        sets.append({c for c, _ in ranked[:k]})
    return sets


def _count_components(example_sets: List[set]) -> Dict[Component, int]:
    counts: Dict[Component, int] = {}
    for s in example_sets:
        for c in s:
            counts[c] = counts.get(c, 0) + 1
    return counts


def _safe_div(a: float, b: float) -> float:
    denom = b if abs(b) > 1e-12 else (1e-12 if b >= 0 else -1e-12)
    return float(a / denom)


def _run_single_combination(
    model, model_name: str, task: str, num_examples: int, digits: int | None,
    top_k_list: List[int], reuse_thresholds: List[int], device: str, debug: bool, run_dir: Path, amp: bool,
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

    # Compute full per-example attribution scores once (keep all), we will slice to top-K later
    extractor = CircuitExtractor(model, top_k=None, method=method)

    combo_key_root = f"{model_name}|{hf_revision or 'none'}|{task}|{method}|n{num_examples}|d{digits}"
    start = time.time()

    example_sets_all: List[set] = []
    example_scores: List[Dict[Component, float]] = []
    try:
        example_sets_all, example_scores = extractor.extract_circuits_from_examples(
            examples=train_examples,
            task_name=task,
            amp=amp,
            device=device,
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] Skipping attribution for {combo_key_root}: {e}")
    except Exception as e:
        print(f"[ERROR] Circuit extraction failed for {combo_key_root}: {e}")
    end = time.time()

    # Evaluate baseline once
    baseline_train_correct, baseline_train_total = evaluate_accuracy(model, train_examples, task=task, verbose=debug)
    baseline_train_acc = baseline_train_correct / baseline_train_total if baseline_train_total > 0 else 0.0
    if val_examples:
        baseline_val_correct, baseline_val_total = evaluate_accuracy(model, val_examples, task=task, verbose=debug)
        baseline_val_acc = baseline_val_correct / baseline_val_total if baseline_val_total > 0 else float("nan")
    else:
        baseline_val_correct = baseline_val_total = 0
        baseline_val_acc = float("nan")

    # Prepare metrics structure
    metrics: Dict[str, Any] = {
        "version": 2,
        "model_name": model_name,
        "hf_revision": hf_revision,
        "task": task,
        "num_examples": len(dataset),
        "digits": digits if task == "addition" else None,
        "method": method,
        "top_k_list": list(sorted(set(int(k) for k in top_k_list))),
        "reuse_thresholds": list(sorted(set(int(p) for p in reuse_thresholds))),
        "val_fraction": vf,
    "perm_trials": int(perm_trials),
        "extraction_seconds": end - start,
        "baseline_train_accuracy": baseline_train_acc,
        "baseline_train_correct": baseline_train_correct,
        "baseline_train_total": baseline_train_total,
        "baseline_val_accuracy": baseline_val_acc,
        "baseline_val_correct": baseline_val_correct,
        "baseline_val_total": baseline_val_total,
        "by_k": {},
    }

    all_components = _enumerate_all_components(model)

    # Loop over K and thresholds
    for K in metrics["top_k_list"]:
        if K <= 0:
            continue
        sets_k = _build_topk_example_sets(example_scores, K)
        counts = _count_components(sets_k)
        n_ex = len(sets_k)

        per_thresh: Dict[str, Any] = {}
        for p in metrics["reuse_thresholds"]:
            thr = max(0, min(100, int(p)))
            need = int(math.ceil(thr / 100.0 * n_ex))
            shared = [c for c, cnt in counts.items() if cnt >= need]
            shared_size = len(shared)
            # Reuse percent is capped at 100
            reuse_percent = float(min(shared_size, K) / max(1, K) * 100.0)

            # Evaluate ablations and collect per-example correctness for permutation tests
            rng_seed = int(hashlib.md5(f"{combo_key_root}|K{K}|p{thr}".encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(rng_seed)
            control_size = min(shared_size, len(all_components))
            control_removed = rng.sample(all_components, control_size) if control_size > 0 else []

            if shared_size > 0:
                ablation_train_correct, ablation_train_total, ablation_train_preds = evaluate_predictions(
                    model, train_examples, task=task, removed=shared, verbose=debug
                )
            else:
                ablation_train_correct, ablation_train_total = baseline_train_correct, baseline_train_total
                ablation_train_preds = [{"is_correct": True}] * len(train_examples)
            if control_size > 0:
                control_train_correct, control_train_total, control_train_preds = evaluate_predictions(
                    model, train_examples, task=task, removed=control_removed, verbose=debug
                )
            else:
                control_train_correct, control_train_total = baseline_train_correct, baseline_train_total
                control_train_preds = [{"is_correct": True}] * len(train_examples)

            ablation_train_acc = ablation_train_correct / ablation_train_total if ablation_train_total > 0 else 0.0
            control_train_acc = control_train_correct / control_train_total if control_train_total > 0 else 0.0

            if val_examples:
                if shared_size > 0:
                    ablation_val_correct, ablation_val_total, ablation_val_preds = evaluate_predictions(
                        model, val_examples, task=task, removed=shared, verbose=debug
                    )
                else:
                    ablation_val_correct, ablation_val_total = baseline_val_correct, baseline_val_total
                    ablation_val_preds = [{"is_correct": True}] * len(val_examples)
                if control_size > 0:
                    control_val_correct, control_val_total, control_val_preds = evaluate_predictions(
                        model, val_examples, task=task, removed=control_removed, verbose=debug
                    )
                else:
                    control_val_correct, control_val_total = baseline_val_correct, baseline_val_total
                    control_val_preds = [{"is_correct": True}] * len(val_examples)
                ablation_val_acc = ablation_val_correct / ablation_val_total if ablation_val_total > 0 else float("nan")
                control_val_acc = control_val_correct / control_val_total if control_val_total > 0 else float("nan")
            else:
                ablation_val_acc = control_val_acc = float("nan")
                ablation_val_preds = control_val_preds = []

            # Permutation tests (paired) for train/val
            shared_flags_train = [1 if r.get("is_correct") else 0 for r in ablation_train_preds]
            control_flags_train = [1 if r.get("is_correct") else 0 for r in control_train_preds]
            perm_train = _permutation_test(shared_flags_train, control_flags_train, trials=int(perm_trials), rng=random.Random(rng_seed + 1)) if len(shared_flags_train) == len(control_flags_train) else {"p_value": 1.0, "obs_diff": 0.0, "trials": 0}

            if val_examples:
                shared_flags_val = [1 if r.get("is_correct") else 0 for r in ablation_val_preds]
                control_flags_val = [1 if r.get("is_correct") else 0 for r in control_val_preds]
                perm_val = _permutation_test(shared_flags_val, control_flags_val, trials=int(perm_trials), rng=random.Random(rng_seed + 2)) if len(shared_flags_val) == len(control_flags_val) else {"p_value": 1.0, "obs_diff": 0.0, "trials": 0}
            else:
                perm_val = {"p_value": float("nan"), "obs_diff": float("nan"), "trials": 0}

            thresh_entry = {
                "threshold": thr,
                "shared_circuit_size": shared_size,
                "reuse_percent": reuse_percent,
                "shared_components": [str(c) for c in sorted(shared, key=lambda c: (c.layer, c.kind, c.index))],
                "rng_seed": rng_seed,
                "train": {
                    "ablation_accuracy": ablation_train_acc,
                    "control_accuracy": control_train_acc,
                    "accuracy_drop_ablation": baseline_train_acc - ablation_train_acc,
                    "accuracy_drop_control": baseline_train_acc - control_train_acc,
                    "knockout_diff": _safe_div(baseline_train_acc - ablation_train_acc, baseline_train_acc - control_train_acc),
                    "permutation": perm_train,
                },
                "val": {
                    "ablation_accuracy": ablation_val_acc,
                    "control_accuracy": control_val_acc,
                    "accuracy_drop_ablation": (baseline_val_acc - ablation_val_acc) if not math.isnan(baseline_val_acc) else float("nan"),
                    "accuracy_drop_control": (baseline_val_acc - control_val_acc) if not math.isnan(baseline_val_acc) else float("nan"),
                    "knockout_diff": _safe_div(baseline_val_acc - ablation_val_acc, baseline_val_acc - control_val_acc) if not math.isnan(baseline_val_acc) else float("nan"),
                    "permutation": perm_val,
                },
            }
            per_thresh[str(thr)] = thresh_entry

        metrics["by_k"][str(K)] = {
            "thresholds": per_thresh,
        }

    combo_dir = run_dir
    combo_dir.mkdir(parents=True, exist_ok=True)

    with (combo_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"[METRICS] {combo_dir/'metrics.json'}")

    # Save per-example attribution rankings once
    attrib_rows = []
    for i, sc in enumerate(example_scores):
        ranked = sorted(sc.items(), key=lambda x: x[1], reverse=True)
        attrib_rows.append({
            "index": i,
            "components": [
                {"layer": c.layer, "kind": c.kind, "index": c.index, "score": float(score)}
                for (c, score) in ranked
            ],
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
        top_k_list = _parse_int_list(args.top_k_list)
        reuse_thresholds = _parse_int_list(args.reuse_thresholds)
        combo_name = (
            f"{args.model_name.replace('/', '_')}__{args.hf_revision or 'main'}__{args.task}__{args.method}__"
            f"n{args.num_examples}__d{digits if args.task=='addition' else 'na'}__Ks{','.join(map(str, top_k_list))}__reuse{','.join(map(str, reuse_thresholds))}"
        )
        run_dir = base_run_dir / combo_name
        print(f"\n[RUN] {combo_name}")

        _run_single_combination(
            model=model,
            model_name=args.model_name,
            task=args.task,
            num_examples=args.num_examples,
            digits=digits,
            top_k_list=top_k_list,
            reuse_thresholds=reuse_thresholds,
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
