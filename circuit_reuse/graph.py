"""This implementation is taken from existing work on Edge Attribution Patching.
@inproceedings{
    syed2023attribution,
    title={Attribution Patching Outperforms Automated Circuit Discovery},
    author={Aaquib Syed and Can Rager and Arthur Conmy},
    booktitle={NeurIPS Workshop on Attributing Model Behavior at Scale},
    year={2023},
    url={https://openreview.net/forum?id=tiLbFR4bJW}
}
"""

from __future__ import annotations
from typing import Callable, List, Union, Optional, Literal, Dict, Set, Tuple
from functools import partial
import torch
from torch import Tensor
from transformer_lens import HookedTransformer
from einops import einsum


class Node:
    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set["Node"]
    children: Set["Node"]
    in_graph: bool
    qkv_inputs: Optional[List[str]]

    def __init__(
        self,
        name: str,
        layer: int,
        in_hook: str,
        out_hook: str,
        index: Tuple,
        qkv_inputs: Optional[List[str]] = None,
    ):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook
        self.index = index
        self.in_graph = True
        self.parents = set()
        self.children = set()
        self.qkv_inputs = qkv_inputs

    def __repr__(self):
        return f"Node({self.name})"

    def __hash__(self):
        return hash(self.name)


class LogitNode(Node):
    def __init__(self, n_layers: int):
        super().__init__(
            "logits", n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", "", slice(None)
        )


class MLPNode(Node):
    def __init__(self, layer: int):
        super().__init__(
            f"m{layer}", layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", slice(None)
        )


class AttentionNode(Node):
    head: int
    kv_head: int

    def __init__(self, layer: int, head: int, cfg: Dict):
        self.head = head
        # Map attention head index to its corresponding KV head index.
        # Some models (e.g., GPT-2) have no separate KV heads; treat n_kv_heads == n_heads.
        n_heads = int(cfg["n_heads"])
        n_kv = int(cfg.get("n_kv_heads") or n_heads)
        ratio = max(1, n_heads // max(1, n_kv))
        self.kv_head = head // ratio
        super().__init__(
            f"a{layer}.h{head}",
            layer,
            f"blocks.{layer}.hook_attn_in",
            f"blocks.{layer}.attn.hook_result",
            (slice(None), slice(None), head),
            [f"blocks.{layer}.hook_{letter}_input" for letter in "qkv"],
        )


class InputNode(Node):
    def __init__(self):
        super().__init__("input", 0, "", "hook_embed", slice(None))


class Graph:
    nodes: Dict[str, Node]
    edges: Dict[str, "Edge"]
    n_forward: int
    n_backward: int
    cfg: Dict
    idx_to_forward_node: Dict[int, Node]

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0
        self.idx_to_forward_node = {}

    def prev_index(self, node: Node) -> int:
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
        elif isinstance(node, MLPNode):
            return 1 + node.layer * (self.cfg["n_heads"] + 1) + self.cfg["n_heads"]
        elif isinstance(node, AttentionNode):
            return 1 + node.layer * (self.cfg["n_heads"] + 1)
        raise ValueError(f"Invalid node type: {type(node)}")

    def forward_index(self, node: Node) -> int:
        if isinstance(node, InputNode):
            return 0
        if isinstance(node, MLPNode):
            return 1 + node.layer * (self.cfg["n_heads"] + 1) + self.cfg["n_heads"]
        if isinstance(node, AttentionNode):
            return 1 + node.layer * (self.cfg["n_heads"] + 1) + node.head
        raise ValueError(f"Node has no forward index: {node}")

    def backward_index(self, node: Node, qkv: Optional[str]) -> int:
        if isinstance(node, InputNode):
            raise ValueError("InputNode has no backward index")
        total_heads_per_layer = self.cfg["n_heads"] + 2 * self.cfg["n_kv_heads"] + 1
        if isinstance(node, LogitNode):
            return self.n_backward - 1
        if isinstance(node, MLPNode):
            return node.layer * total_heads_per_layer + self.cfg["n_heads"] + 2 * self.cfg["n_kv_heads"]
        if isinstance(node, AttentionNode):
            layer_offset = node.layer * total_heads_per_layer
            if qkv is None: qkv = "v" # Default to 'v' if not provided
            if qkv == "q":
                return layer_offset + node.head
            elif qkv == "k":
                return layer_offset + self.cfg["n_heads"] + node.kv_head
            elif qkv == "v":
                return layer_offset + self.cfg["n_heads"] + self.cfg["n_kv_heads"] + node.kv_head
        raise ValueError(f"Invalid node type or qkv: {type(node)}, {qkv}")

    @classmethod
    def from_model(cls, model: HookedTransformer) -> "Graph":
        graph = Graph()
        cfg = model.cfg
        nkv_heads = getattr(cfg, "n_key_value_heads", None)
        nkv_heads = int(nkv_heads) if nkv_heads is not None else int(cfg.n_heads)
        if nkv_heads <= 0:
            nkv_heads = int(cfg.n_heads)
        graph.cfg = {"n_layers": cfg.n_layers, "n_heads": cfg.n_heads, "n_kv_heads": nkv_heads}

        nodes: List[Node] = [InputNode()]
        for layer in range(cfg.n_layers):
            nodes.extend([AttentionNode(layer, h, graph.cfg) for h in range(cfg.n_heads)])
            nodes.append(MLPNode(layer))
        nodes.append(LogitNode(cfg.n_layers))

        for node in nodes:
            graph.nodes[node.name] = node
            if not isinstance(node, LogitNode):
                fwd_idx = graph.forward_index(node)
                graph.idx_to_forward_node[fwd_idx] = node

        for parent_node in nodes:
            if isinstance(parent_node, LogitNode): continue
            for child_node in nodes:
                is_causal = (parent_node.layer < child_node.layer) or \
                            (isinstance(parent_node, AttentionNode) and isinstance(child_node, MLPNode) and parent_node.layer == child_node.layer) or \
                            isinstance(child_node, LogitNode)
                if not is_causal: continue
                if isinstance(child_node, AttentionNode):
                    for letter in "qkv":
                        graph.edges[Edge(parent_node, child_node, qkv=letter).name] = True
                elif not isinstance(child_node, InputNode):
                    graph.edges[Edge(parent_node, child_node).name] = True

        graph.n_forward = 1 + cfg.n_layers * (cfg.n_heads + 1)
        graph.n_backward = (cfg.n_layers * (cfg.n_heads + 2 * nkv_heads + 1)) + 1
        return graph


class Edge:
    def __init__(self, parent: Node, child: Node, qkv: Optional[Literal["q", "k", "v"]] = None):
        self.parent = parent
        self.child = child
        self.qkv = qkv
        self.name = f"{parent.name}->{child.name}" + (f"<{qkv}>" if qkv else "")


def make_hooks(model: HookedTransformer, graph: Graph, activation_difference: Tensor, scores: Tensor):
    
    def activation_hook(index, activations, hook, add: bool = True, head_index: Optional[int] = None):
        seq_len = activations.shape[1]
        acts = activations.detach()
        if head_index is not None and acts.ndim == 4:
            acts = acts[:, :, head_index, :]
        
        # Write into the slice of the pre-allocated tensor
        if add:
            activation_difference[:, :seq_len, index, :] += acts
        else:
            activation_difference[:, :seq_len, index, :] -= acts

    def gradient_hook(prev_index: int, bwd_index_slice: slice, gradients: Tensor, hook):
        seq_len = gradients.shape[1]
        grads = gradients.detach()
        if grads.ndim == 4:
            grads = grads.sum(dim=2)
        if grads.ndim == 3:
            grads = grads.unsqueeze(2)
        
        # Use the corresponding slice of the activation_difference tensor
        act_diff_slice = activation_difference[:, :seq_len, :prev_index]
        
        s = einsum(
            act_diff_slice, grads,
            "batch pos fwd hidden, batch pos bwd hidden -> fwd bwd",
        )
        scores[:prev_index, bwd_index_slice] += s

    fwd_hooks_clean, fwd_hooks_corrupted, bwd_hooks = [], [], []
    processed_attn_layers = set()
    for name, node in graph.nodes.items():
        if not isinstance(node, LogitNode):
            fwd_idx = graph.forward_index(node)
            head_idx = node.head if isinstance(node, AttentionNode) else None
            fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_idx, add=True, head_index=head_idx)))
            fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_idx, add=False, head_index=head_idx)))

        if not isinstance(node, InputNode):
            prev_idx = graph.prev_index(node)
            if isinstance(node, AttentionNode):
                if node.layer in processed_attn_layers: continue
                processed_attn_layers.add(node.layer)
            bwd_idx = graph.backward_index(node, None)
            bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_idx, slice(bwd_idx, bwd_idx + 1))))
            
    return fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks


def attribute_single_example(
    model: HookedTransformer,
    graph: Graph,
    metric: Callable,
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    activation_difference: Tensor,
) -> Tensor:
    """
    Runs EAP for a single example, reusing a pre-allocated activation_difference tensor.
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=model.cfg.device, dtype=model.cfg.dtype)
    
    # IMPORTANT: Reset the reusable tensor before processing the new example
    activation_difference.zero_()

    fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks = make_hooks(
        model, graph, activation_difference, scores
    )

    with torch.no_grad():
        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            model(corrupted_tokens)

    with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
        logits = model(clean_tokens)
        metric_value = metric(logits, None, None, None)
        metric_value.backward()
    
    model.zero_grad(set_to_none=True)
    model.reset_hooks()

    return scores.cpu()