from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


Hook = Tuple[str, Callable[[torch.Tensor, Any], Optional[torch.Tensor]]]


class HFHookedOLMo:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        torch_dtype: Optional[torch.dtype] = None,
        revision: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()

        cfg = self.model.config
        n_layers = int(getattr(cfg, "num_hidden_layers"))
        n_heads = int(getattr(cfg, "num_attention_heads"))
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads) or n_heads)
        d_model = int(getattr(cfg, "hidden_size"))

        self.cfg = SimpleNamespace(
            n_layers=n_layers,
            n_heads=n_heads,
            n_key_value_heads=n_kv_heads,
            d_model=d_model,
            device=device,
            dtype=next(self.model.parameters()).dtype,
            use_split_qkv_input=True,
            use_attn_result=True,
            use_hook_mlp_in=True,
        )

        self._active_fwd: Dict[str, List[Callable]] = {}
        self._active_bwd: Dict[str, List[Callable]] = {}
        self._persist_handles: List[Any] = []
        self._wire_persistent_hooks()

    def to(self, device: str) -> "HFHookedOLMo":
        self.model.to(device)
        self.cfg.device = device
        return self

    def eval(self) -> "HFHookedOLMo":
        self.model.eval()
        return self

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=tokens.to(self.cfg.device))
        return out.logits

    def reset_hooks(self) -> None:
        self._active_fwd.clear()
        self._active_bwd.clear()

    def to_tokens(self, text: str, prepend_bos: bool = False) -> torch.Tensor:
        if prepend_bos and self.tokenizer.bos_token_id is not None:
            ids = [self.tokenizer.bos_token_id]
            ids += self.tokenizer(text, add_special_tokens=False)["input_ids"]
            return torch.tensor([ids], device=self.cfg.device, dtype=torch.long)
        toks = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return toks["input_ids"].to(self.cfg.device)

    @contextmanager
    def hooks(
        self,
        fwd_hooks: Optional[List[Hook]] = None,
        bwd_hooks: Optional[List[Hook]] = None,
    ):
        if fwd_hooks:
            for name, fn in fwd_hooks:
                self._active_fwd.setdefault(name, []).append(fn)
        if bwd_hooks:
            for name, fn in bwd_hooks:
                self._active_bwd.setdefault(name, []).append(fn)
        try:
            yield self
        finally:
            self.reset_hooks()

    # ----------------- internal helpers -----------------

    def _emit_fwd(self, name: str, tensor: torch.Tensor) -> None:
        for fn in self._active_fwd.get(name, []):
            try:
                fn(tensor, None)
            except TypeError:
                fn(tensor)

    def _attach_bwd(self, name: str, tensor: torch.Tensor) -> None:
        if name not in self._active_bwd:
            return
        if tensor is None or tensor.grad_fn is None:
            return

        def _on_grad(grad: torch.Tensor):
            for fn in self._active_bwd.get(name, []):
                try:
                    fn(grad, None)
                except TypeError:
                    fn(grad)
            return grad

        tensor.register_hook(_on_grad)

    def _register_pre_hook(self, module: nn.Module, fn, with_kwargs=True):
        """
        Try new API with with_kwargs=True; fall back if not supported.
        """
        try:
            return module.register_forward_pre_hook(fn, with_kwargs=with_kwargs)
        except TypeError:
            # Older PyTorch: hook signature must be (mod, args)
            def shim(mod, args):
                return fn(mod, args, {})  # pass empty kwargs
            return module.register_forward_pre_hook(shim)

    def _wire_persistent_hooks(self) -> None:
        model = self.model

        # hook_embed
        emb = model.get_input_embeddings()

        def _embed_fwd(_mod, _inp, out):
            self._emit_fwd("hook_embed", out)

        self._persist_handles.append(emb.register_forward_hook(_embed_fwd))

        # layers
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise RuntimeError("Expected Olmo2Model at .model with .layers ModuleList")
        layers: nn.ModuleList = model.model.layers
        if not isinstance(layers, nn.ModuleList):
            raise RuntimeError("Expected model.model.layers to be an nn.ModuleList")

        n_layers = self.cfg.n_layers
        n_heads = self.cfg.n_heads
        d_model = self.cfg.d_model
        d_head = d_model // n_heads
        last_idx = n_layers - 1

        for li in range(n_layers):
            block = layers[li]
            if not hasattr(block, "self_attn"):
                raise RuntimeError(f"Layer {li} has no .self_attn")
            attn = block.self_attn

            # FIX: support kwargs-only calls for attention
            def _attn_pre(mod, args, kwargs, _li=li):
                # input hidden states may come as positional or kwarg
                x = None
                if isinstance(args, tuple) and len(args) > 0:
                    x = args[0]
                elif isinstance(kwargs, dict):
                    x = kwargs.get("hidden_states", None)
                if x is None:
                    return
                name = f"blocks.{_li}.hook_attn_in"
                self._emit_fwd(name, x)
                self._attach_bwd(name, x)

            self._persist_handles.append(self._register_pre_hook(attn, _attn_pre, with_kwargs=True))

            # attn.hook_result via o_proj input
            if not hasattr(attn, "o_proj"):
                raise RuntimeError(f"Layer {li} self_attn has no .o_proj")
            o_proj = attn.o_proj

            def _o_proj_pre(mod, inputs, _kw, _li=li, _n_heads=n_heads, _d_head=d_head):
                x = inputs[0] if isinstance(inputs, tuple) and len(inputs) > 0 else None
                if x is None:
                    return
                if x.dim() == 3 and x.size(-1) == _n_heads * _d_head:
                    x4 = x.view(x.size(0), x.size(1), _n_heads, _d_head)
                    self._emit_fwd(f"blocks.{_li}.attn.hook_result", x4)
                else:
                    self._emit_fwd(f"blocks.{_li}.attn.hook_result", x)

            self._persist_handles.append(self._register_pre_hook(o_proj, _o_proj_pre, with_kwargs=True))

            # mlp in/out (handle kwargs just in case)
            if not hasattr(block, "mlp"):
                raise RuntimeError(f"Layer {li} has no .mlp")
            mlp = block.mlp

            def _mlp_pre(mod, args, kwargs, _li=li):
                x = None
                if isinstance(args, tuple) and len(args) > 0:
                    x = args[0]
                elif isinstance(kwargs, dict):
                    x = kwargs.get("hidden_states", None)
                if x is None:
                    return
                name = f"blocks.{_li}.hook_mlp_in"
                self._emit_fwd(name, x)
                self._attach_bwd(name, x)

            def _mlp_fwd(mod, inputs, out, _li=li):
                self._emit_fwd(f"blocks.{_li}.hook_mlp_out", out)

            self._persist_handles.append(self._register_pre_hook(mlp, _mlp_pre, with_kwargs=True))
            self._persist_handles.append(mlp.register_forward_hook(_mlp_fwd))

            if li == last_idx:
                def _block_fwd(mod, inputs, out, _li=li):
                    name = f"blocks.{_li}.hook_resid_post"
                    self._emit_fwd(name, out)
                    self._attach_bwd(name, out)
                self._persist_handles.append(block.register_forward_hook(_block_fwd))


def load_model_any(
    model_name: str,
    device: str,
    torch_dtype: Optional[torch.dtype] = None,
    revision: Optional[str] = None,
):
    try:
        from transformer_lens import HookedTransformer
        m = HookedTransformer.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch_dtype
        ).to(device).eval()
        return m
    except Exception:
        return HFHookedOLMo(model_name, device=device, torch_dtype=torch_dtype, revision=revision)
