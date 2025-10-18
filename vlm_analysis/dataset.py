from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from datasets import Image as HFImage
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class VLMExample:
    """Single VLMBias dataset item with image, question, ground truth, and expected bias."""

    id: str
    image: Optional[object]
    image_path: str
    question: str
    ground_truth: str
    expected_bias: str
    metadata: Optional[dict] = None


def load_vlmbias(
    split: str = "main",
    domain: Optional[str] = None,
    *,
    return_images: bool = True,
    num_examples: Optional[int] = None,
):
    """Load VLMBias dataset, optionally filtered by domain."""
    ds = load_dataset("anvo25/vlms-are-biased", split=split)

    if domain is not None:
        # Disable image decoding during filtering to avoid PIL errors on truncated files
        if "image" in ds.features:
            ds = ds.cast_column("image", HFImage(decode=False))
        ds = ds.filter(lambda x: str(x.get("topic", "")).lower() == domain.lower())
        # Re-enable image decoding for downstream use
        if "image" in ds.features:
            ds = ds.cast_column("image", HFImage(decode=True))

    examples = []
    for i, row in enumerate(ds):
        if num_examples is not None and i >= num_examples:
            break

        # Lazily and defensively load images: tolerate truncated/corrupt files
        if return_images:
            try:
                img = row.get("image", None)
            except Exception:
                img = None
        else:
            img = None
        ex = VLMExample(
            id=str(row.get("ID", i)),
            image=img,
            image_path=str(row.get("image_path", "")),
            question=str(row.get("prompt", "")),
            ground_truth=str(row.get("ground_truth", "")),
            expected_bias=str(row.get("expected_bias", "")),
            metadata={
                k: row[k]
                for k in ["topic", "sub_topic", "with_title", "type_of_question"]
                if k in row
            },
        )
        examples.append(ex)

    return examples


__all__ = ["VLMExample", "load_vlmbias"]