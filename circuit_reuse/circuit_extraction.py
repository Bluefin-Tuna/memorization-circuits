from __future__ import annotations
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from transformer_lens import HookedTransformer
from contextlib import nullcontext
import gc

from .graph import Graph, attribute_single_example, make_hooks
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
    def __init__(self, model: HookedTransformer, top_k: Optional[int] = 5, method: str = "eap") -> None:
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
            # Sum log-probabilities of all target tokens at their respective positions (teacher forcing)
            # logits: [batch, pos, vocab]
            # positions: [m], target_ids: [m]
            logprobs = logits.log_softmax(dim=-1)
            # Batch size is 1; index accordingly
            selected = logprobs[0, positions, :].gather(dim=1, index=target_ids.view(-1, 1))
            return selected.sum()
        return metric

    def _scores_to_components(self, scores: torch.Tensor) -> Dict[Component, float]:
        from .graph import InputNode, MLPNode, AttentionNode
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

    def _prepare_eap_inputs(self, example: Example):
        """Prepares and tokenizes a single example for EAP."""
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

        clean_padding = (0, max_len - clean_full_tok.shape[1])
        clean_tokens = F.pad(clean_full_tok, clean_padding, "constant", pad_token).to(device)
        
        corrupted_padding = (0, max_len - corrupted_full_tok.shape[1])
        corrupted_tokens = F.pad(corrupted_full_tok, corrupted_padding, "constant", pad_token).to(device)

        metric = self._get_metric_fn(positions=positions, target_ids=target_ids)
        
        return clean_tokens, corrupted_tokens, metric, max_len

    def _prepare_gradient_inputs(self, example: Example):
        """Prepares and tokenizes a single example for the gradient method."""
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
        
    def extract_circuits_from_examples(self, examples: List[Example], task_name: str, amp: bool, device: str) -> List[Set[Component]]:
        """
        Extracts circuits for a list of examples using the method specified in the constructor.
        """
        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp and device.startswith("cuda") else nullcontext())
        
        if self.method == "gradient":
            circuits = []
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
                items = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
                comp_set = {c for c, _ in (items[:self.top_k] if self.top_k is not None else items)}
                circuits.append(comp_set)

                if (idx + 1) % 10 == 0 or (idx + 1) == len(examples):
                    print(f"[{task_name}] ({self.method}) {idx + 1}/{len(examples)} examples processed (last circuit size={len(comp_set)})")

            del activations_buffer
            gc.collect()
            torch.cuda.empty_cache()
            return circuits

        elif self.method == "eap":
            circuits = []
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
                items = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
                comp_set = {c for c, _ in (items[:self.top_k] if self.top_k is not None else items)}
                circuits.append(comp_set)

                if (idx + 1) % 10 == 0 or (idx + 1) == len(examples):
                    print(f"[{task_name}] ({self.method}) {idx + 1}/{len(examples)} train examples processed (last circuit size={len(comp_set)})")

            del activation_difference
            gc.collect()
            torch.cuda.empty_cache()
            return circuits
        
        else:
            raise ValueError(f"Unsupported method: {self.method}")


def compute_shared_circuit(circuits: List[Set[Component]]) -> Set[Component]:
    if not circuits:
        return set()
    shared = set(circuits[0])
    for c in circuits[1:]:
        shared.intersection_update(c)
    return shared


__all__ = ["Component", "CircuitExtractor", "compute_shared_circuit"]