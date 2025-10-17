from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from datasets import Image as HFImage
from PIL import ImageFile

# Some images in the dataset are slightly truncated; allow PIL to load them instead of erroring
ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class VLMExample:
    """Container for a single VLMBias item.

    Attributes
    ----------
    id: str
        Unique identifier for the example.
    image: any
        The PIL image or NumPy array representing the picture.  This field
        may be ``None`` if ``return_images`` is set to False when
        loading the dataset; in that case only the ``image_path`` is
        populated and downstream code can lazily load the image.
    image_path: str
        Path to the image within the dataset.  Useful when deferring
        image loading or logging.
    question: str
        The question posed to the model (e.g., ``"How many stripes does the logo have?"``).
    ground_truth: str
        The correct answer according to the modified image.
    expected_bias: str
        The answer that matches the bias encoded in the unmodified object
        (e.g., ``"3"`` for a four-stripe Adidas logo).  Used when
        computing the bias-aligned error rate.
    metadata: Optional[dict]
        Additional metadata provided by the dataset such as topic or
        subtopic.  Not used directly in evaluation but available for
        filtering.
    """

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
    """
    Load a subset of the VLMBias dataset.

    Parameters
    ----------
    split : str, default ``"main"``
        Which split of the dataset to load.  The dataset card defines
        several splits such as ``"main"`` and ``"identification"``.  Use
        ``"main"`` for the primary experiments.
    domain : str, optional
        Restrict the loaded examples to a particular domain (topic).
        Domains correspond to high-level categories such as "Logos",
        "Chess", "Board Games", etc.  If ``None`` all domains are
        returned.
    return_images : bool, default True
        Whether to return the decoded PIL images for each example.  When
        False, the ``image`` attribute of each :class:`VLMExample` will
        be ``None`` and only ``image_path`` is populated.  Loading
        images eagerly can be expensive; setting this flag to False is
        useful when only file paths are needed (e.g., when the
        underlying model accepts image paths directly).
    num_examples : int, optional
        If provided, truncate the dataset to the first ``num_examples``
        items after filtering.  This is primarily intended for quick
        debugging or testing.

    Returns
    -------
    List[VLMExample]
        A list of :class:`VLMExample` instances containing the
        requested subset of the data.

    Notes
    -----
    This function will import the ``datasets`` library at call time.
    Raising an ImportError is deferred until dataset loading is
    requested.  If the dataset cannot be downloaded due to network
    restrictions the caller should handle the resulting exception.
    """
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