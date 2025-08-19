import torch
from models.olmo_adapter import HFHookedOLMo  # the adapter we added

MODEL_ID = "allenai/OLMo-2-0425-1B-early-training"
REVISION = "stage1-step20000-tokens42B"  # pick any tag from the model card

m = HFHookedOLMo(MODEL_ID, device="cpu", torch_dtype=None, revision=REVISION)

ids = m.to_tokens("Hello world", prepend_bos=True)

buf = {}
def grab(label):
    def f(x, hook=None, **_):
        buf[label] = x.detach().cpu()
    return f

with m.hooks(fwd_hooks=[
    ("hook_embed", grab("embed")),
    ("blocks.0.hook_attn_in", grab("attn_in_0")),
    ("blocks.0.attn.hook_result", grab("attn_res_0")),
    ("blocks.0.hook_mlp_in", grab("mlp_in_0")),
    ("blocks.0.hook_mlp_out", grab("mlp_out_0")),
    (f"blocks.{m.cfg.n_layers-1}.hook_resid_post", grab("resid_post_last")),
]):
    with torch.no_grad():
        logits = m(ids)

print("logits", tuple(logits.shape))
print("attn_res_0", tuple(buf["attn_res_0"].shape))
