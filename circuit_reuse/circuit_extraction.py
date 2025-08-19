from __future__ import annotations
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from transformer_lens import HookedTransformer
from contextlib import nullcontext
import gc
from collections import defaultdict

from .graph import Graph, attribute_single_example, make_hooks
from .dataset import Example


@dataclass(frozen=True)
class Component:
    layer: int
    kind: str  # "head" or "mlp"
    index: int

    def __hash__(self) -> int:
        return hash((self.layer, self.kind, self.index))

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.kind}[layer={self.layer}, index={self.index}]"


def _component_key(c: Component) -> Tuple[int, str, int]:
    return (c.layer, c.kind, c.index)


class CircuitExtractor:
    """
    Extract per-example attribution scores for components.

    Note on semantics:
      - top_k controls how many components we keep per-example (for the old pipeline).
        For the new "circuit identifiability score" pipeline, set top_k=None to keep
        all scored components per example and decide the shared K separately downstream.
    """
    def __init__(self, model: HookedTransformer, top_k: Optional[int] = None, method: str = "eap") -> None:
        self.model = model
        self.top_k = top_k
        self.method = method
        self.graph = Graph.from_model(model)
        # Enable hooks needed for all methods
        self.model.cfg.use_split_qkv_input = True
        self.model.cfg.use_attn_result = True
        self.model.cfg.use_hook_mlp_in = True

    def _get_metric_fn(self, positions: torch.Tensor, target_ids: torch.Tensor):
        def metric(logits: torch.Tensor, corrupted_logits: torch.Tensor, input_lengths: torch.Tensor, label: Any) -> torch.Tensor:
            logprobs = logits.log_softmax(dim=-1)
            selected = logprobs[0, positions, :].gather(dim=1, index=target_ids.view(-1, 1))
            return selected.sum()
        return metric

    def _scores_to_components(self, scores: torch.Tensor) -> Dict[Component, float]:
        from .graph import InputNode, MLPNode, AttentionNode
        component_scores: Dict[Component, float] = {}
        per_component_scores = scores.abs().sum(dim=1)

        for fwd_idx, score in enumerate(per_component_scores.tolist()):
            node = self.graph.idx_to_forward_node.get(fwd_idx)
            if node is None or isinstance(node, InputNode):
                continue

            comp = None
            if hasattr(node, "layer"):
                if node.__class__.__name__ == "MLPNode":
                    comp = Component(layer=node.layer, kind="mlp", index=0)
                elif node.__class__.__name__ == "AttentionNode":
                    comp = Component(layer=node.layer, kind="head", index=node.head)
            if comp:
                component_scores[comp] = float(score)
        return component_scores

    def _prepare_eap_inputs(self, example: Example):
        prompt_tok = self.model.to_tokens(example.prompt, prepend_bos=True)
        clean_full_tok = self.model.to_tokens(example.prompt + example.target, prepend_bos=True)
        corrupted_full_tok = self.model.to_tokens(example.corrupted_prompt + example.corrupted_target, prepend_bos=True)

        device = self.model.cfg.device
        p_ids, f_ids = prompt_tok.tolist()[0], clean_full_tok.tolist()[0]
        lcp = 0
        while lcp < len(p_ids) and lcp < len(f_ids) and p_ids[lcp] == f_ids[lcp]:
            lcp += 1

        gold_ids_list = f_ids[lcp:] if lcp < len(f_ids) else self.model.to_tokens(example.target, prepend_bos=False).tolist()[0]
        target_ids = torch.tensor(gold_ids_list, device=device, dtype=torch.long)
        prompt_len = prompt_tok.shape[1]
        positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(gold_ids_list), device=device, dtype=torch.long)

        max_len = max(clean_full_tok.shape[1], corrupted_full_tok.shape[1])
        pad_token = self.model.tokenizer.pad_token_id if self.model.tokenizer.pad_token_id is not None else self.model.tokenizer.eos_token_id

        clean_tokens = F.pad(clean_full_tok, (0, max_len - clean_full_tok.shape[1]), "constant", pad_token).to(device)
        corrupted_tokens = F.pad(corrupted_full_tok, (0, max_len - corrupted_full_tok.shape[1]), "constant", pad_token).to(device)

        metric = self._get_metric_fn(positions=positions, target_ids=target_ids)
        return clean_tokens, corrupted_tokens, metric, max_len

    def _prepare_gradient_inputs(self, example: Example):
        prompt_tok = self.model.to_tokens(example.prompt, prepend_bos=True)
        clean_full_tok = self.model.to_tokens(example.prompt + example.target, prepend_bos=True)

        device = self.model.cfg.device
        p_ids, f_ids = prompt_tok.tolist()[0], clean_full_tok.tolist()[0]
        lcp = 0
        while lcp < len(p_ids) and lcp < len(f_ids) and p_ids[lcp] == f_ids[lcp]:
            lcp += 1

        gold_ids_list = f_ids[lcp:] if lcp < len(f_ids) else self.model.to_tokens(example.target, prepend_bos=False).tolist()[0]
        target_ids = torch.tensor(gold_ids_list, device=device, dtype=torch.long)
        prompt_len = prompt_tok.shape[1]
        positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(gold_ids_list), device=device, dtype=torch.long)
        metric = self._get_metric_fn(positions=positions, target_ids=target_ids)
        return clean_full_tok.to(device), metric, clean_full_tok.shape[1]

    def extract_circuits_from_examples(
        self, examples: List[Example], task_name: str, amp: bool, device: str
    ) -> Tuple[List[Set[Component]], List[Dict[Component, float]]]:
        """
        Returns:
          - per-example circuit sets (possibly truncated to self.top_k if set)
          - per-example full score dicts {Component -> score} (no truncation)
        """
        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp and device.startswith("cuda") else nullcontext())

        circuits: List[Set[Component]] = []
        per_example_scores: List[Dict[Component, float]] = []

        if self.method == "gradient":
            max_seq_len = 0
            all_tokenized = [self._prepare_gradient_inputs(ex) for ex in examples]
            for clean_tokens, _, _ in all_tokenized:
                max_seq_len = max(max_seq_len, clean_tokens.shape[1])

            activations_buffer = torch.zeros(
                (1, max_seq_len, self.graph.n_forward, self.model.cfg.d_model),
                device=self.model.cfg.device, dtype=self.model.cfg.dtype,
            )

            for idx, (clean_tokens, metric, seq_len) in enumerate(all_tokenized):
                with autocast_ctx:
                    scores = torch.zeros((self.graph.n_forward, self.graph.n_backward), device=self.model.cfg.device, dtype=self.model.cfg.dtype)
                    activations_buffer.zero_()

                    _, fwd_hooks_clean, bwd_hooks = make_hooks(self.model, self.graph, activations_buffer, scores)

                    with self.model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
                        logits = self.model(clean_tokens)
                        metric_value = metric(logits, None, None, None)
                        metric_value.backward()

                    self.model.zero_grad(set_to_none=True)
                    self.model.reset_hooks()

                component_scores = self._scores_to_components(scores.cpu())
                per_example_scores.append(component_scores)
                items = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
                comp_set = {c for c, _ in (items[:self.top_k] if self.top_k is not None else items)}
                circuits.append(comp_set)

                if (idx + 1) % 10 == 0 or (idx + 1) == len(examples):
                    print(f"[{task_name}] ({self.method}) {idx + 1}/{len(examples)} examples processed (last circuit size={len(comp_set)})")

            del activations_buffer
            gc.collect()
            torch.cuda.empty_cache()
            return circuits, per_example_scores

        elif self.method == "eap":
            max_seq_len = 0
            all_tokenized = [self._prepare_eap_inputs(ex) for ex in examples]
            for clean_tokens, corrupted_tokens, _, _ in all_tokenized:
                max_seq_len = max(max_seq_len, clean_tokens.shape[1], corrupted_tokens.shape[1])

            activation_difference = torch.zeros(
                (1, max_seq_len, self.graph.n_forward, self.model.cfg.d_model),
                device=self.model.cfg.device, dtype=self.model.cfg.dtype,
            )

            for idx, (clean_tokens, corrupted_tokens, metric, seq_len) in enumerate(all_tokenized):
                with autocast_ctx:
                    scores = attribute_single_example(
                        model=self.model, graph=self.graph, metric=metric,
                        clean_tokens=clean_tokens, corrupted_tokens=corrupted_tokens,
                        activation_difference=activation_difference,
                    )

                component_scores = self._scores_to_components(scores)
                per_example_scores.append(component_scores)
                items = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
                comp_set = {c for c, _ in (items[:self.top_k] if self.top_k is not None else items)}
                circuits.append(comp_set)

                if (idx + 1) % 10 == 0 or (idx + 1) == len(all_tokenized):
                    print(f"[{task_name}] ({self.method}) {idx + 1}/{len(all_tokenized)} train examples processed (last circuit size={len(comp_set)})")

            del activation_difference
            gc.collect()
            torch.cuda.empty_cache()
            return circuits, per_example_scores

        else:
            raise ValueError(f"Unsupported method: {self.method}")


def compute_shared_circuit(circuits: List[Set[Component]]) -> Set[Component]:
    """
    Legacy: strict intersection across all examples.
    """
    if not circuits:
        return set()
    shared = set(circuits[0])
    for c in circuits[1:]:
        shared.intersection_update(c)
    return shared


def select_shared_by_proportion(
    example_sets: List[Set[Component]],
    example_scores: List[Dict[Component, float]],
    k_shared: int,
) -> Tuple[Set[Component], Dict[Component, float]]:
    """
    New selection for the "circuit identifiability score".

    For each component c, compute:
      - prop(c): fraction of examples whose per-example circuit contains c.
      - mean_score(c): mean attribution score across examples where c appears.

    We then choose the top-k_shared components sorted by:
      1) prop(c) descending,
      2) mean_score(c) descending.

    Returns:
      selected set, and a dict of prop(c) for all selected components.
    """
    n = max(1, len(example_sets))
    counts: Dict[Component, int] = defaultdict(int)
    sum_scores: Dict[Component, float] = defaultdict(float)

    for s, sc in zip(example_sets, example_scores):
        for c in s:
            counts[c] += 1
            sum_scores[c] += float(sc.get(c, 0.0))

    candidates = []
    for c, cnt in counts.items():
        prop = cnt / n
        mean_sc = sum_scores[c] / max(1, cnt)
        candidates.append((c, prop, mean_sc))

    candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)
    chosen = [c for c, _, _ in candidates[:max(0, k_shared)]]
    prop_map = {c: p for c, p, _ in candidates[:max(0, k_shared)]}
    return set(chosen), prop_map


def circuit_identifiability_score(prop_map: Dict[Component, float]) -> float:
    """
    Average of selected component proportions.
    """
    if not prop_map:
        return 0.0
    return float(sum(prop_map.values()) / len(prop_map))


__all__ = [
    "Component",
    "CircuitExtractor",
    "compute_shared_circuit",        # legacy
    "select_shared_by_proportion",   # new
    "circuit_identifiability_score", # new
]
