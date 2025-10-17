from __future__ import annotations

from typing import Optional
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
)


def load_vlm_model(
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    *,
    dtype: Optional[str] = None,
    revision: Optional[str] = None,
):
    # Resolve device: if CUDA requested but unavailable, fallback to CPU
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    torch_dtype = None
    if dtype is not None and dtype != "auto":
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map[dtype]

    processor = AutoProcessor.from_pretrained(model_name, revision=revision)

    # Prefer modern multi-modal class when supported; else fallback
    try:
        if hasattr(processor, "image_processor"):
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=torch_dtype,
                device_map=None,
            )
        else:
            raise ValueError("not multi-modal")
    except Exception:
        # fallback path for older checkpoints
        if hasattr(processor, "image_processor"):
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=torch_dtype,
                device_map=None,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=torch_dtype,
                device_map=None,
            )

    model = model.to(device)
    model.eval()
    return model, processor


__all__ = ["load_vlm_model"]