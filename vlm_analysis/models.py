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
    load_in_8bit: bool = False,
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

    # 8-bit quantization config
    load_kwargs = {
        "revision": revision,
        "torch_dtype": torch_dtype,
    }

    if load_in_8bit:
        # Use device_map="auto" for 8-bit quantization
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = None

    processor = AutoProcessor.from_pretrained(model_name, revision=revision)

    # Set padding side to left for decoder-only models (required for correct generation)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    # Some processors have tokenizer as a direct attribute
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"

    # Prefer modern multi-modal class when supported; else fallback
    try:
        if hasattr(processor, "image_processor"):
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **load_kwargs,
            )
        else:
            raise ValueError("not multi-modal")
    except Exception:
        # fallback path for older checkpoints
        if hasattr(processor, "image_processor"):
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                **load_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs,
            )

    # Only move to device if not using 8-bit (device_map handles it)
    if not load_in_8bit:
        model = model.to(device)
    model.eval()
    return model, processor


__all__ = ["load_vlm_model"]