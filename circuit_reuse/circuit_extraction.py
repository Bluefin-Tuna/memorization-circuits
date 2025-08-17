from __future__ import annotations
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from transformer_lens import HookedTransformer

from .eap import Graph, attribute
from .dataset import Example


@dataclass(frozen=True)
class Component:
    layer: int
    kind: str  # 'head' or 'mlp'
    index: int

    def __hash__(self) -> int:
        return hash((self.layer, self.kind, self.index))

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.kind}[layer={self.layer}, index={self.index}]"


class CircuitExtractor:
    def __init__(self, model: HookedTransformer, top_k: Optional[int] = 5) -> None:
        self.model = model
        self.top_k = top_k
        self.graph = Graph.from_model(model)
        # Enable hooks needed for EAP
        self.model.cfg.use_split_qkv_input = True
        self.model.cfg.use_attn_result = True
        self.model.cfg.use_hook_mlp_in = True

    def _get_metric_fn(self, positions: torch.Tensor, target_ids: torch.Tensor):
        def metric(logits: torch.Tensor, corrupted_logits: torch.Tensor, input_lengths: torch.Tensor, label: Any) -> torch.Tensor:
            # Sum log-probabilities of all target tokens at their respective positions (teacher forcing)
            # logits: [batch, pos, vocab]
            # positions: [m], target_ids: [m]
            logprobs = logits.log_softmax(dim=-1)
            # Batch size is 1 in our EAP dataloader; index accordingly
            selected = logprobs[0, positions, :].gather(dim=1, index=target_ids.view(-1, 1))
            return selected.sum()
        return metric

    def _scores_to_components(self, scores: torch.Tensor) -> Dict[Component, float]:
        from .eap import InputNode, MLPNode, AttentionNode
        component_scores: Dict[Component, float] = {}
        # scores has shape (n_forward, n_backward). Sum over backward dim to get score per source component.
        per_component_scores = scores.abs().sum(dim=1)

        for fwd_idx, score in enumerate(per_component_scores.tolist()):
            node = self.graph.idx_to_forward_node.get(fwd_idx)
            if node is None or isinstance(node, InputNode):
                continue

            comp = None
            if isinstance(node, MLPNode):
                comp = Component(layer=node.layer, kind="mlp", index=0)
            elif isinstance(node, AttentionNode):
                comp = Component(layer=node.layer, kind="head", index=node.head)

            if comp:
                component_scores[comp] = score
        return component_scores

    def extract_circuit(self, example: Example) -> Set[Component]:
        # Tokenize prompt and full prompt+target for principled multi-token metric
        prompt_tok = self.model.to_tokens(example.prompt, prepend_bos=True)
        clean_full_tok = self.model.to_tokens(example.prompt + example.target, prepend_bos=True)
        corrupted_full_tok = self.model.to_tokens(example.corrupted_prompt + example.corrupted_target, prepend_bos=True)

        # Derive target token IDs after the prompt boundary
        device = self.model.cfg.device
        # Local derivation to avoid circular import: get tokens in full vs prompt and take the suffix
        p_ids = prompt_tok.to(device)[0].tolist()
        f_ids = clean_full_tok.to(device)[0].tolist()
        lcp = 0
        for a, b in zip(p_ids, f_ids):
            if a == b:
                lcp += 1
            else:
                break
        gold_ids_list = f_ids[lcp:] if lcp < len(f_ids) else self.model.to_tokens(example.target, prepend_bos=False).to(device)[0].tolist()
        target_ids = torch.tensor(gold_ids_list, device=device, dtype=torch.long)
        prompt_len = prompt_tok.shape[1]
        positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(gold_ids_list), device=device, dtype=torch.long)

        # Pad clean/corrupted sequences to the same length
        max_len = max(clean_full_tok.shape[1], corrupted_full_tok.shape[1])
        pad_token = self.model.tokenizer.pad_token_id if self.model.tokenizer.pad_token_id is not None else self.model.tokenizer.eos_token_id

        clean_padding = (max_len - clean_full_tok.shape[1], 0)
        clean_tokens = F.pad(clean_full_tok, clean_padding, "constant", pad_token)
        clean_attention_mask = (clean_tokens != pad_token).long()
        # Not used by metric anymore, but keep shape for API compatibility
        clean_input_lengths = torch.tensor([clean_full_tok.shape[1]], device=clean_tokens.device)

        corrupted_padding = (max_len - corrupted_full_tok.shape[1], 0)
        corrupted_tokens = F.pad(corrupted_full_tok, corrupted_padding, "constant", pad_token)
        corrupted_attention_mask = (corrupted_tokens != pad_token).long()

        # Build metric for multi-token targets
        metric = self._get_metric_fn(positions=positions, target_ids=target_ids)

        dataloader = [
            (
                (clean_tokens.to(device), clean_attention_mask.to(device), clean_input_lengths.to(device), max_len),
                (corrupted_tokens.to(device), corrupted_attention_mask.to(device), None, None),
                (positions, target_ids),
            )
        ]

        scores = attribute(
            model=self.model,
            graph=self.graph,
            dataloader=dataloader,
            metric=metric,
            method="EAP",
            quiet=True,
        )

        component_scores = self._scores_to_components(scores)
        items = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)

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