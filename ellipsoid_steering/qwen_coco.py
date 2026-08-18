"""Qwen2.5-VL helpers for small deterministic COCO/CHAIR experiments.

Heavy Hugging Face imports are deliberately lazy so geometry-only users do not
need the vision-language dependencies.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from .perturbations import GaussianBlurPerturbation


DEFAULT_PROMPT = "Please describe this image in detail."


def select_coco_images(annotation_file: str | Path, count: int = 200,
                       seed: int = 42) -> list[dict]:
    """Return a stable random subset from a COCO instances JSON file."""
    if count < 1:
        raise ValueError("count must be positive")
    with Path(annotation_file).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = sorted(payload["images"], key=lambda item: int(item["id"]))
    if count > len(images):
        raise ValueError(f"requested {count} images but annotation file has {len(images)}")
    random.Random(seed).shuffle(images)
    return images[:count]


def write_coco_split(annotation_file: str | Path, output: str | Path,
                     fit_count: int = 100, eval_count: int = 100,
                     seed: int = 42) -> dict:
    selected = select_coco_images(annotation_file, fit_count + eval_count, seed)
    split = {
        "seed": seed,
        "annotation_file": str(annotation_file),
        "fit": selected[:fit_count],
        "eval": selected[fit_count:],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split


def load_split(path: str | Path, partition: str) -> list[dict]:
    if partition not in {"fit", "eval"}:
        raise ValueError("partition must be 'fit' or 'eval'")
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)[partition]


class Qwen25VLAdapter:
    """Minimal adapter around Qwen2.5-VL processor and language decoder."""

    def __init__(self, model_name: str, dtype: str = "bfloat16",
                 device_map: str = "auto", min_pixels: int | None = None,
                 max_pixels: int | None = None) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        torch_dtype = getattr(torch, dtype)
        self.model_name = model_name
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype=torch_dtype, device_map=device_map).eval()
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def prepare(self, image: Image.Image, prompt: str = DEFAULT_PROMPT) -> dict:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt")
        return {key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in inputs.items()}

    def text_mask(self, inputs: dict) -> torch.Tensor:
        mask = inputs["attention_mask"].bool()
        input_ids = inputs["input_ids"]
        special_ids = []
        for name in ("image_token_id", "video_token_id"):
            value = getattr(self.model.config, name, None)
            if value is not None:
                special_ids.append(value)
        for token_id in special_ids:
            mask &= input_ids.ne(token_id)
        return mask

    def decode_generated(self, output_ids: torch.Tensor, prompt_length: int) -> str:
        generated = output_ids[:, prompt_length:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0].strip()


def iter_coco_images(image_dir: str | Path, records: Iterable[dict]):
    root = Path(image_dir)
    for record in records:
        path = root / record["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            yield int(record["id"]), image.convert("RGB")


def make_masked_image(image: Image.Image, blur_radius: float = 12.0) -> Image.Image:
    return GaussianBlurPerturbation(blur_radius)(image)


def save_chair_captions(path: str | Path, captions: list[dict]) -> None:
    """Save the input schema consumed by the original CHAIR scorer."""
    normalized = [{"image_id": int(item["image_id"]),
                   "caption": str(item["caption"])} for item in captions]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
