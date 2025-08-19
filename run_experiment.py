import argparse
import time
from typing import List
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
from circuit_reuse.circuit_extraction import CircuitExtractor, compute_shared_circuit, Component
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
    parser = argparse.ArgumentParser(description="Circuit reuse experiment (single-run)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")
    parser.add_argument("--task", type=str, required=True, help="Task name.")
    parser.add_argument("--num_examples", type=int, required=True, help="Number of examples to generate/use.")
    parser.add_argument("--digits", type=int, default=None, help="Digit count (only used for addition task).")
    parser.add_argument("--top_k", type=int, required=True, help="Top-k components per example for extraction.")
    parser.add_argument("--method", type=str, default="eap", choices=["eap", "gradient"], help="Attribution method to use.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "float16", "float32"], help="Model/load dtype to reduce memory (bf16 recommended).")
    parser.add_argument("--log-mem", action="store_true", help="Print CUDA memory after the run.")
    parser.add_argument("--amp", action="store_true", help="Use autocast (mixed precision) during extraction.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of dataset held out for validation evaluation (NOT used for circuit extraction). 0 disables validation.")
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


def _run_single_combination(
    model, model_name: str, task: str, num_examples: int, digits: int | None,
    top_k: int, device: str, debug: bool, run_dir: Path, amp: bool, val_fraction: float, method: str,
):
    dataset = get_dataset(task, num_examples=num_examples, digits=digits if digits is not None else 0)
    print(f"[{model_name}/{task}] Generated {len(dataset)} examples for method '{method}'.")

    extractor = CircuitExtractor(model, top_k=top_k, method=method)

    examples = list(dataset)
    random.shuffle(examples)
    n = len(examples)
    vf = max(0.0, min(0.9, val_fraction))
    val_count = int(round(vf * n)) if vf > 0 else 0
    if val_count >= n and n > 1: val_count = n - 1
    train_examples = examples[: n - val_count]
    val_examples = examples[n - val_count :]
    if debug:
        print(f"[SPLIT] total={n} train={len(train_examples)} val={len(val_examples)} (val_fraction={vf:.2f})")

    combo_key = f"{model_name}|{task}|{method}|n{num_examples}|d{digits}|k{top_k}"
    start = time.time()
    circuits: List[set] = []
    
    try:
        circuits = extractor.extract_circuits_from_examples(
            examples=train_examples,
            task_name=task,
            amp=amp,
            device=device
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"[OOM] Skipping {combo_key}: CUDA out of memory. {e}")
        # We can still proceed with an empty circuit list to generate a partial result file.
    except Exception as e:
        print(f"[ERROR] Circuit extraction failed for {combo_key}: {e}")

    end = time.time()

    shared = compute_shared_circuit(circuits)
    print(f"[{task}] Shared circuit size={len(shared)} (top_k per example={top_k}).")

    baseline_train_correct, baseline_train_total = evaluate_accuracy(model, train_examples, task=task, verbose=debug)
    baseline_train_acc = baseline_train_correct / baseline_train_total if baseline_train_total > 0 else 0.0
    ablation_train_correct, ablation_train_total = evaluate_accuracy_with_ablation(model, train_examples, task=task, removed=shared, verbose=debug)
    ablation_train_acc = ablation_train_correct / ablation_train_total if ablation_train_total > 0 else 0.0

    if val_examples:
        baseline_val_correct, baseline_val_total = evaluate_accuracy(model, val_examples, task=task, verbose=debug)
        baseline_val_acc = baseline_val_correct / baseline_val_total if baseline_val_total > 0 else 0.0
        ablation_val_correct, ablation_val_total = evaluate_accuracy_with_ablation(model, val_examples, task=task, removed=shared, verbose=debug)
        ablation_val_acc = ablation_val_correct / ablation_val_total if ablation_val_total > 0 else 0.0
    else:
        baseline_val_correct, baseline_val_total, ablation_val_correct, ablation_val_total = 0, 0, 0, 0
        baseline_val_acc, ablation_val_acc = float('nan'), float('nan')

    all_components = _enumerate_all_components(model)
    k = min(len(shared), len(all_components))
    rng_seed = int(hashlib.md5(combo_key.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(rng_seed)
    control_removed = rng.sample(all_components, k) if k > 0 else []
    print(f"[{task}] Control ablation uses {len(control_removed)}/{len(all_components)} random components (seed={rng_seed}).")

    control_train_correct, control_train_total = evaluate_accuracy_with_ablation(model, train_examples, task=task, removed=control_removed, verbose=debug)
    control_train_acc = control_train_correct / control_train_total if control_train_total > 0 else 0.0
    if val_examples:
        control_val_correct, control_val_total = evaluate_accuracy_with_ablation(model, val_examples, task=task, removed=control_removed, verbose=debug)
        control_val_acc = control_val_correct / control_val_total if control_val_total > 0 else 0.0
    else:
        control_val_correct, control_val_total = 0, 0
        control_val_acc = float('nan')

    metrics = {
        "model_name": model_name, "task": task, "num_examples": len(dataset),
        "digits": digits if task == "addition" else None, "top_k": top_k, "method": method,
        "baseline_train_accuracy": baseline_train_acc, "baseline_train_correct": baseline_train_correct,
        "baseline_train_total": baseline_train_total, "ablation_train_accuracy": ablation_train_acc,
        "ablation_train_correct": ablation_train_correct, "ablation_train_total": ablation_train_total,
        "baseline_val_accuracy": baseline_val_acc, "baseline_val_correct": baseline_val_correct,
        "baseline_val_total": baseline_val_total, "ablation_val_accuracy": ablation_val_acc,
        "ablation_val_correct": ablation_val_correct, "ablation_val_total": ablation_val_total,
        "accuracy_drop_train": baseline_train_acc - ablation_train_acc,
        "accuracy_drop_val": (baseline_val_acc - ablation_val_acc) if not math.isnan(baseline_val_acc) else float('nan'),
        "accuracy_drop_train_val": baseline_train_acc - ablation_val_acc if not math.isnan(ablation_val_acc) else float('nan'),
        "val_fraction": vf, "shared_circuit_size": len(shared),
        "shared_circuit_components": [str(c) for c in sorted(shared, key=lambda c: (c.layer, c.kind, c.index))],
        "extraction_seconds": end - start, "control_train_accuracy": control_train_acc,
        "control_train_correct": control_train_correct, "control_train_total": control_train_total,
        "control_val_accuracy": control_val_acc, "control_val_correct": control_val_correct,
        "control_val_total": control_val_total, "control_accuracy_drop_train": baseline_train_acc - control_train_acc,
        "control_accuracy_drop_val": (baseline_val_acc - control_val_acc) if not math.isnan(baseline_val_acc) else float('nan'),
        "control_accuracy_drop_train_val": baseline_train_acc - control_val_acc if not math.isnan(control_val_acc) else float('nan'),
        "control_removed_components": [str(c) for c in sorted(control_removed, key=lambda c: (c.layer, c.kind, c.index))],
        "control_rng_seed": rng_seed, "control_total_component_count": len(all_components),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"[DONE] {run_dir}")


def main() -> None:
    args = parse_args()
    base_run_dir = _prepare_run_dir(args.output_dir, args.run_name)
    if args.log_mem:
        print(f"Model: {args.model_name}, Task: {args.task}")

    dtype_map = {"bf16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    
    total_runs = 1
    run_count = 0

    model_name = args.model_name
    model = None
    try:
        print(f"[MODEL LOAD] Loading model {model_name} (dtype={args.dtype}) on {args.device}...")
        torch_dtype = None if args.dtype == "auto" else dtype_map[args.dtype]
        model = load_model_any(model_name, device=args.device, torch_dtype=torch_dtype)
        model.eval()
        if args.log_mem:
            print(f"[MEM] post-load allocated={torch.cuda.memory_allocated()/1e9:.2f}GiB reserved={torch.cuda.memory_reserved()/1e9:.2f}GiB")

        task = args.task
        digits = args.digits if task == "addition" else None
        top_k = args.top_k
        num_examples = args.num_examples

        run_count += 1
        combo_name = f"{model_name.replace('/', '_')}__{task}__{args.method}__n{num_examples}__d{digits if task=='addition' else 'na'}__k{top_k}"
        run_dir = base_run_dir / combo_name
        print(f"\n[RUN {run_count}/{total_runs}] {combo_name}")

        try:
            _run_single_combination(
                model=model, model_name=model_name, task=task, num_examples=num_examples,
                digits=digits, top_k=top_k, device=args.device, debug=args.debug,
                run_dir=run_dir, amp=args.amp, val_fraction=args.val_fraction, method=args.method
            )
        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM] Skipping {combo_name}: {e}")
            else:
                print(f"[ERROR] {combo_name} failed: {e}")
                import traceback
                traceback.print_exc()
        finally:
            if args.log_mem:
                print(f"[MEM] {combo_name} allocated={torch.cuda.memory_allocated()/1e9:.2f}GiB reserved={torch.cuda.memory_reserved()/1e9:.2f}GiB")
            model.reset_hooks()

    except Exception as e:
        print(f"[FATAL] Failed to process model {model_name}: {e}")
    finally:
        # Clean up model and cache after all tasks for it are done
        if model is not None:
            del model
        torch.cuda.empty_cache()
        if args.log_mem:
            print(f"[MEM] after-model-del allocated={torch.cuda.memory_allocated()/1e9:.2f}GiB reserved={torch.cuda.memory_reserved()/1e9:.2f}GiB")
    
    print(f"[ALL DONE] Completed {run_count}/{total_runs} runs. Results root: {base_run_dir}")


if __name__ == "__main__":
    main()