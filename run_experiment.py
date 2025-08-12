#!/usr/bin/env python3
# Minimal CLI to run circuit reuse experiments.

import argparse
import time
from typing import List
import json
import math
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import torch
from transformer_lens import HookedTransformer

from circuit_reuse.dataset import AdditionDataset, get_dataset
from circuit_reuse.circuit_extraction import CircuitExtractor, compute_shared_circuit
from circuit_reuse.evaluate import evaluate_accuracy, evaluate_accuracy_with_ablation

def _default_run_name():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _prepare_run_dir(output_dir: str, run_name: str | None):
    base = Path(output_dir)
    if run_name is None or run_name.strip() == "":
        run_name = _default_run_name()
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def parse_args() -> argparse.Namespace:
    """Parse CLI args (modern multi-run only)."""
    parser = argparse.ArgumentParser(description="Circuit reuse experiment (multi-run only)")
    # ONLY list-based args now
    parser.add_argument("--model_names", nargs="+", type=str, required=True,
                        help="List of model names.")
    parser.add_argument("--tasks", nargs="+", type=str, required=True,
                        help="List of tasks.")
    parser.add_argument("--num_examples_list", nargs="+", type=int, required=True,
                        help="List of num_examples values.")
    parser.add_argument("--digits_list", nargs="+", type=int, required=True,
                        help="List of digit counts (used only for addition).")
    parser.add_argument("--top_ks", nargs="+", type=int, required=True,
                        help="List of top_k values.")
    parser.add_argument("--methods", nargs="+", type=str, choices=["gradient", "ig"], required=True,
                        help="List of attribution methods.")
    parser.add_argument("--steps", type=int, default=5,
                        help="Interpolation steps for integrated gradients.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "bf16", "float16", "float32"],
                        help="Model/load dtype to reduce memory (bf16 recommended).")
    parser.add_argument("--log-mem", action="store_true",
                        help="Print CUDA memory after each combo.")
    parser.add_argument("--amp", action="store_true",
                        help="Use autocast (mixed precision) during extraction.")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of dataset held out for validation evaluation (NOT used for circuit extraction). 0 disables validation.")
    return parser.parse_args()

def _run_single_combination(
    model: HookedTransformer,
    model_name: str,
    task: str,
    num_examples: int,
    digits: int | None,
    top_k: int,
    method: str,
    steps: int,
    device: str,
    debug: bool,
    run_dir: Path,
    amp: bool,
    val_fraction: float,
):
    """Execute one (model, task, num_examples, digits, top_k, method) combo and write outputs into run_dir."""
    # DATASET
    if task == "addition":
        dataset = AdditionDataset(num_examples=num_examples, digits=digits if digits is not None else 2)
        print(f"[{model_name}/{task}] Generated {len(dataset)} examples (digits={digits}).")
    else:
        dataset = get_dataset(task, num_examples=num_examples, digits=digits if digits is not None else 0)
        print(f"[{model_name}/{task}] Loaded {len(dataset)} examples.")

    extractor = CircuitExtractor(model, top_k=top_k)

    # Train/validation split (validation only for evaluation)
    examples = list(dataset)
    n = len(examples)
    vf = max(0.0, min(0.9, val_fraction))  # clamp
    val_count = int(round(vf * n)) if vf > 0 else 0
    if val_count >= n and n > 1:
        val_count = n - 1
    train_examples = examples[: n - val_count]
    val_examples = examples[n - val_count:]
    if debug:
        print(f"[SPLIT] total={n} train={len(train_examples)} val={len(val_examples)} (val_fraction={vf:.2f})")

    circuits: List[set] = []
    start = time.time()
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if amp and device.startswith("cuda")
        else nullcontext()
    )
    for idx, example in enumerate(train_examples):
        with autocast_ctx:
            if method == "gradient":
                comp_set = extractor.extract_circuit(example.prompt, example.target)
            else:
                comp_set = extractor.extract_circuit_ig(example.prompt, example.target, steps=steps)
        circuits.append(comp_set)
        if debug:
            print(f"[EXTRACT {task}] idx={idx} comps={len(comp_set)}")
        if (idx + 1) % 10 == 0 or (idx + 1) == len(train_examples):
            print(f"[{task}] {idx + 1}/{len(train_examples)} train examples (last circuit size={len(comp_set)})")
        # free intermediate grads right away
        model.zero_grad(set_to_none=True)
    end = time.time()

    shared = compute_shared_circuit(circuits)
    print(f"[{task}] Shared circuit size={len(shared)} (top_k per example={top_k}).")

    baseline_train_acc = evaluate_accuracy(model, train_examples, task=task, verbose=debug)
    ablation_train_acc = evaluate_accuracy_with_ablation(model, train_examples, task=task, removed=shared, verbose=debug)
    if val_examples:
        baseline_val_acc = evaluate_accuracy(model, val_examples, task=task, verbose=debug)
        ablation_val_acc = evaluate_accuracy_with_ablation(model, val_examples, task=task, removed=shared, verbose=debug)
    else:
        baseline_val_acc = float('nan')
        ablation_val_acc = float('nan')

    metrics = {
        "model_name": model_name,
        "task": task,
        "num_examples": len(dataset),
        "digits": digits if task == "addition" else None,
        "top_k": top_k,
        "method": method,
        "ig_steps": steps if method == "ig" else None,
        "baseline_train_accuracy": baseline_train_acc,
        "ablation_train_accuracy": ablation_train_acc,
        "baseline_val_accuracy": baseline_val_acc,
        "ablation_val_accuracy": ablation_val_acc,
        "accuracy_drop_train": baseline_train_acc - ablation_train_acc,
        "accuracy_drop_val": (baseline_val_acc - ablation_val_acc) if not math.isnan(baseline_val_acc) else float('nan'),
        "accuracy_drop_train_val": baseline_train_acc - ablation_val_acc if not math.isnan(ablation_val_acc) else float('nan'),
        "val_fraction": vf,
        "shared_circuit_size": len(shared),
        "shared_circuit_components": [
            str(c) for c in sorted(shared, key=lambda c: (c.layer, c.kind, c.index))
        ],
        "extraction_seconds": end - start,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (run_dir / "metrics.json").open("w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write metrics.json in {run_dir}: {e}")

    try:
        config = {
            "model_name": model_name,
            "task": task,
            "num_examples": num_examples,
            "digits": digits,
            "top_k": top_k,
            "method": method,
            "steps": steps,
            "device": device,
            "debug": debug,
        }
        with (run_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"[WARN] Failed to write config.json in {run_dir}: {e}")

    print(f"[DONE] {run_dir}")
    return metrics

def main() -> None:
    """Run experiment end-to-end (single or multi-run)."""
    args = parse_args()

    base_run_dir = _prepare_run_dir(args.output_dir, args.run_name)

    print(f"Models: {args.model_names}")
    print(f"Tasks: {args.tasks}")
    print(f"Methods: {args.methods}")
    print(f"top_ks: {args.top_ks}")
    print(f"digits_list: {args.digits_list}")
    print(f"num_examples_list: {args.num_examples_list}")

    dtype_map = {
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    def _print_mem(prefix: str):
        if not torch.cuda.is_available():
            return
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[MEM] {prefix} allocated={alloc:.2f}GiB reserved={reserved:.2f}GiB")

    total_runs = 0
    for task in args.tasks:
        if task == "addition":
            total_runs += (len(args.model_names) * len(args.methods) *
                           len(args.top_ks) * len(args.digits_list) * len(args.num_examples_list))
        else:
            total_runs += (len(args.model_names) * len(args.methods) *
                           len(args.top_ks) * 1 * len(args.num_examples_list))
    run_counter = 0

    for model_name in args.model_names:
        print(f"[MODEL LOAD] Loading model {model_name} (dtype={args.dtype}) on {args.device}...")
        load_kwargs = {"trust_remote_code": True}
        if args.dtype != "auto":
            load_kwargs["dtype"] = dtype_map[args.dtype]
        model: HookedTransformer = HookedTransformer.from_pretrained(model_name, **load_kwargs)
        model.to(args.device).eval()
        _print_mem("post-load")

        for task in args.tasks:
            digits_iter = args.digits_list if task == "addition" else [None]
            for method in args.methods:
                for top_k in args.top_ks:
                    for digits in digits_iter:
                        for num_examples in args.num_examples_list:
                            run_counter += 1
                            combo_name = (
                                f"{model_name}__{task}"
                                f"__n{num_examples}"
                                f"__d{digits if task=='addition' else 'na'}"
                                f"__k{top_k}__{method}"
                            )
                            run_dir = base_run_dir / combo_name
                            print(f"\n[RUN {run_counter}/{total_runs}] {combo_name}")
                            try:
                                _run_single_combination(
                                    model=model,
                                    model_name=model_name,
                                    task=task,
                                    num_examples=num_examples,
                                    digits=digits,
                                    top_k=top_k,
                                    method=method,
                                    steps=args.steps,
                                    device=args.device,
                                    debug=args.debug,
                                    run_dir=run_dir,
                                    amp=args.amp,
                                    val_fraction=args.val_fraction,
                                )
                            except torch.cuda.OutOfMemoryError as oom:
                                print(f"[OOM] Skipping {combo_name}: {oom}")
                            except Exception as e:
                                print(f"[ERROR] {combo_name} failed: {e}")
                            finally:
                                # Release hooks/caches if present
                                if hasattr(model, "reset_hooks"):
                                    try: model.reset_hooks()
                                    except Exception: pass
                                if hasattr(model, "clear_contexts"):
                                    try: model.clear_contexts()
                                    except Exception: pass
                                model.zero_grad(set_to_none=True)
                                torch.cuda.empty_cache()
                                if args.log_mem:
                                    _print_mem(combo_name)

        del model
        torch.cuda.empty_cache()
        if args.log_mem:
            _print_mem("after-model-del")

    print(f"[ALL DONE] Completed {run_counter}/{total_runs} runs. Results root: {base_run_dir.resolve()}")

if __name__ == "__main__":
    main()