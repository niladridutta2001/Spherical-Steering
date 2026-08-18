"""Small-scale Qwen2.5-VL COCO collection and CHAIR caption generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ellipsoid_steering.activation_collector import ActivationCollector
from ellipsoid_steering.cache import load_geometry
from ellipsoid_steering.config import EllipsoidSteeringConfig
from ellipsoid_steering.controller import EllipsoidSteeringController, SteeringContext
from ellipsoid_steering.qwen_coco import (
    DEFAULT_PROMPT, Qwen25VLAdapter, iter_coco_images, load_split,
    make_masked_image, save_chair_captions, write_coco_split)
from ellipsoid_steering.utils import resolve_transformer_layers


def add_model_args(parser):
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)


def build_adapter(args):
    return Qwen25VLAdapter(args.model_name, args.dtype, min_pixels=args.min_pixels,
                           max_pixels=args.max_pixels)


def command_split(args):
    split = write_coco_split(args.annotations, args.output, args.fit_count,
                             args.eval_count, args.seed)
    print(f"Saved {len(split['fit'])} fit and {len(split['eval'])} eval images to {args.output}")


def command_collect(args):
    adapter = build_adapter(args)
    layers = resolve_transformer_layers(adapter.model)
    collector = ActivationCollector(adapter.model, args.layers, layers, offload_to_cpu=True)
    records = load_split(args.split, "fit")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)

    for index, (image_id, image) in enumerate(tqdm(
            iter_coco_images(args.image_dir, records), total=len(records))):
        original = adapter.prepare(image, args.prompt)
        masked = adapter.prepare(make_masked_image(image, args.blur_radius), args.prompt)
        text_mask = adapter.text_mask(original).cpu()
        original_values = collector.collect(original, token_mask=text_mask)
        masked_values = collector.collect(masked, token_mask=text_mask)
        activation_chunk = {
            "activations": original_values,
            "example_ids": torch.tensor([image_id]),
        }
        paired_chunk = {
            "grounded": original_values,
            "hallucinated": masked_values,
            "example_ids": torch.full((next(iter(original_values.values())).shape[0],), image_id),
        }
        torch.save(activation_chunk, output / f"grounded_{index:05d}.pt")
        torch.save(paired_chunk, output / f"paired_{index:05d}.pt")
    print(f"Saved {len(records)} grounded and paired chunks to {output}")


def _generate_one(adapter, inputs, max_new_tokens):
    with torch.inference_mode():
        output = adapter.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False, use_cache=True)
    return adapter.decode_generated(output, inputs["input_ids"].shape[1])


def command_generate(args):
    adapter = build_adapter(args)
    records = load_split(args.split, args.partition)
    controller = None
    if args.geometry:
        covariances, modes, _ = load_geometry(
            args.geometry, expected_model=args.model_name,
            expected_layers=args.layers)
        config = EllipsoidSteeringConfig(
            target_layers=args.layers, routing_mode="instance", position_aware=False,
            steering_gain=args.steering_gain, steering_threshold=args.steering_threshold,
            beta_max=args.beta_max, uniform_beta=args.beta,
            spectral_gamma=args.spectral_gamma,
            preserve_mahalanobis_radius=not args.no_preserve_radius)
        controller = EllipsoidSteeringController(
            adapter.model, covariances, modes, config,
            layers=resolve_transformer_layers(adapter.model))

    captions = []
    for image_id, image in tqdm(iter_coco_images(args.image_dir, records), total=len(records)):
        inputs = adapter.prepare(image, args.prompt)
        if controller is None:
            caption = _generate_one(adapter, inputs, args.max_new_tokens)
        else:
            masked_inputs = adapter.prepare(
                make_masked_image(image, args.blur_radius), args.prompt)
            masked_activations = controller.routing_pass(masked_inputs)
            context = SteeringContext(masked_activations)
            with controller.enabled(context):
                caption = _generate_one(adapter, inputs, args.max_new_tokens)
        captions.append({"image_id": image_id, "caption": caption})
    save_chair_captions(args.output, captions)
    print(json.dumps({"captions": len(captions), "output": args.output,
                      "steered": controller is not None}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("make-split")
    split.add_argument("--annotations", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--fit-count", type=int, default=100)
    split.add_argument("--eval-count", type=int, default=100)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=command_split)

    collect = commands.add_parser("collect")
    add_model_args(collect)
    collect.add_argument("--split", required=True)
    collect.add_argument("--image-dir", required=True)
    collect.add_argument("--output-dir", required=True)
    collect.add_argument("--layers", type=int, nargs="+", required=True)
    collect.add_argument("--prompt", default=DEFAULT_PROMPT)
    collect.add_argument("--blur-radius", type=float, default=12.0)
    collect.set_defaults(func=command_collect)

    generate = commands.add_parser("generate")
    add_model_args(generate)
    generate.add_argument("--split", required=True)
    generate.add_argument("--partition", choices=("fit", "eval"), default="eval")
    generate.add_argument("--image-dir", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--geometry")
    generate.add_argument("--layers", type=int, nargs="+", default=[19])
    generate.add_argument("--prompt", default=DEFAULT_PROMPT)
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--blur-radius", type=float, default=12.0)
    generate.add_argument("--steering-gain", type=float, default=1.0)
    generate.add_argument("--steering-threshold", type=float, default=0.05)
    generate.add_argument("--beta-max", type=float, default=0.5)
    generate.add_argument("--beta", type=float)
    generate.add_argument("--spectral-gamma", type=float, default=1e-2)
    generate.add_argument("--no-preserve-radius", action="store_true")
    generate.set_defaults(func=command_generate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
