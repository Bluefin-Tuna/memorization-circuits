from __future__ import annotations

import re
import torch
from PIL import Image
from tqdm import tqdm

from .dataset import VLMExample


def _normalise_answer(ans):
    return ans.strip().lower()


def _extract_answer(text):
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[1]
    
    square_matches = re.findall(r"\[([^\]]+)\]", text)
    if square_matches:
        return square_matches[-1]
    
    curly_matches = re.findall(r"\{([^\}]+)\}", text)
    if curly_matches:
        return curly_matches[-1]
    
    parts = text.split()
    return parts[0] if parts else ""


def evaluate_dataset(
    model,
    processor,
    examples: list[VLMExample],
    *,
    max_new_tokens: int = 5,
    batch_size: int = 8,
):
    """
    Evaluate dataset with batched processing for speed.

    Args:
        model: The VLM model
        processor: The model's processor
        examples: List of VLMExample objects
        max_new_tokens: Maximum new tokens to generate
        batch_size: Number of examples to process in parallel
    """
    correct = 0
    total = 0
    bias_aligned_errors = 0
    total_errors = 0
    details = []

    def build_batch_inputs(batch_examples):
        """Build inputs for a batch of examples."""
        images = []
        texts = []

        for ex in batch_examples:
            img_or_path = ex.image if ex.image is not None else ex.image_path
            images.append(img_or_path)
            texts.append(ex.question)

        # Try to use chat template if available
        try:
            if hasattr(processor, "apply_chat_template") and hasattr(processor, "chat_template") and processor.chat_template is not None:
                # Build messages for batch
                formatted_texts = []
                for img, text in zip(images, texts):
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": text},
                            ],
                        }
                    ]
                    formatted_texts.append(processor.apply_chat_template(messages, add_generation_prompt=True))

                inputs = processor(text=formatted_texts, images=images, return_tensors="pt", padding=True)
            else:
                raise AttributeError("No chat template")
        except (ValueError, AttributeError, RuntimeError):
            # Fallback for processors without chat template
            sig = processor.__call__.__code__.co_varnames
            image_key = "images" if "images" in sig else "pixel_values"
            inputs = processor(
                **{image_key: images},
                text=texts,
                return_tensors="pt",
                padding=True,
            )

        return {k: v.to(model.device) for k, v in inputs.items()}

    # Process examples in batches with progress bar
    pbar = tqdm(total=len(examples), desc="Evaluating", unit="ex")

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]

        try:
            inputs = build_batch_inputs(batch)

            # Generate for batch
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

            # Decode batch outputs
            if hasattr(processor, "batch_decode"):
                pred_strs = processor.batch_decode(outputs, skip_special_tokens=True)
            elif hasattr(getattr(processor, "tokenizer", None), "batch_decode"):
                pred_strs = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            else:
                tok = getattr(model, "tokenizer", None)
                pred_strs = tok.batch_decode(outputs, skip_special_tokens=True) if tok is not None else [""] * len(batch)

            # Process results for each example in batch
            for ex, pred_str in zip(batch, pred_strs):
                answer = _extract_answer(pred_str)

                norm_pred = _normalise_answer(answer)
                norm_gt = _normalise_answer(ex.ground_truth)
                norm_bias = _normalise_answer(ex.expected_bias)

                is_correct = norm_pred == norm_gt
                if is_correct:
                    correct += 1
                else:
                    total_errors += 1
                    if norm_pred == norm_bias:
                        bias_aligned_errors += 1
                total += 1

                details.append(
                    {
                        "id": ex.id,
                        "question": ex.question,
                        "ground_truth": ex.ground_truth,
                        "expected_bias": ex.expected_bias,
                        "prediction_raw": pred_str,
                        "prediction": answer,
                        "prediction_normalised": norm_pred,
                        "is_correct": is_correct,
                        "is_bias_aligned": (norm_pred == norm_bias) and not is_correct,
                    }
                )

            # Update progress bar
            pbar.update(len(batch))

        except Exception as e:
            # If batch fails, fall back to processing one at a time for this batch
            print(f"Warning: Batch processing failed ({e}), falling back to sequential processing for this batch")
            for ex in batch:
                try:
                    single_batch = [ex]
                    inputs = build_batch_inputs(single_batch)

                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

                    if hasattr(processor, "batch_decode"):
                        pred_str = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    elif hasattr(getattr(processor, "tokenizer", None), "batch_decode"):
                        pred_str = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    else:
                        tok = getattr(model, "tokenizer", None)
                        pred_str = tok.batch_decode(outputs, skip_special_tokens=True)[0] if tok is not None else ""

                    answer = _extract_answer(pred_str)

                    norm_pred = _normalise_answer(answer)
                    norm_gt = _normalise_answer(ex.ground_truth)
                    norm_bias = _normalise_answer(ex.expected_bias)

                    is_correct = norm_pred == norm_gt
                    if is_correct:
                        correct += 1
                    else:
                        total_errors += 1
                        if norm_pred == norm_bias:
                            bias_aligned_errors += 1
                    total += 1

                    details.append(
                        {
                            "id": ex.id,
                            "question": ex.question,
                            "ground_truth": ex.ground_truth,
                            "expected_bias": ex.expected_bias,
                            "prediction_raw": pred_str,
                            "prediction": answer,
                            "prediction_normalised": norm_pred,
                            "is_correct": is_correct,
                            "is_bias_aligned": (norm_pred == norm_bias) and not is_correct,
                        }
                    )
                    # Update progress bar for successful sequential processing
                    pbar.update(1)
                except Exception as ex_error:
                    print(f"Error processing example {ex.id}: {ex_error}")
                    # Still update progress even on error
                    pbar.update(1)
                    continue

    # Close progress bar
    pbar.close()

    accuracy = correct / total if total > 0 else 0.0
    baer = (bias_aligned_errors / total_errors) if total_errors > 0 else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "total_errors": total_errors,
        "bias_aligned_errors": bias_aligned_errors,
        "baer": baer,
        "details": details,
    }


__all__ = ["evaluate_dataset"]