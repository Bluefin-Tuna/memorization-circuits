from __future__ import annotations

import re
from PIL import Image

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
):
    correct = 0
    total = 0
    bias_aligned_errors = 0
    total_errors = 0
    details = []

    def build_inputs(ex: VLMExample):
        img_or_path = ex.image if ex.image is not None else ex.image_path
        if hasattr(processor, "apply_chat_template") and hasattr(processor, "chat_template") and processor.chat_template is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_or_path},
                        {"type": "text", "text": ex.question},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[img_or_path], return_tensors="pt", padding=True)
        else:
            sig = processor.__call__.__code__.co_varnames
            image_key = "images" if "images" in sig else "pixel_values"
            inputs = processor(
                **{image_key: img_or_path},
                text=ex.question,
                return_tensors="pt",
            )
        return {k: v.to(model.device) for k, v in inputs.items()}

    for ex in examples:
        inputs = build_inputs(ex)

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