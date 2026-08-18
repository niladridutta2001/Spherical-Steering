"""Run two-pass LVLM steering through a user-provided model adapter factory."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ellipsoid_steering.cache import load_geometry
from ellipsoid_steering.config import EllipsoidSteeringConfig
from ellipsoid_steering.controller import EllipsoidSteeringController, SteeringContext


def load_symbol(spec: str):
    module, symbol = spec.split(":", 1)
    return getattr(importlib.import_module(module), symbol)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Factory returns model, original_inputs, masked_inputs, generate_kwargs")
    parser.add_argument("--factory", required=True, help="Python `module:function` adapter factory")
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-layers", type=int, nargs="+", required=True)
    parser.add_argument("--routing-mode", choices=("teacher_forced", "instance", "position"),
                        default="instance")
    parser.add_argument("--routing-temperature", type=float, default=0.1)
    parser.add_argument("--steering-gain", type=float, default=1.0)
    parser.add_argument("--steering-threshold", type=float, default=0.05)
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--spectral-gamma", type=float, default=1e-2)
    parser.add_argument("--hard-routing", action="store_true")
    parser.add_argument("--hard-projection", action="store_true")
    parser.add_argument("--no-preserve-radius", action="store_true")
    parser.add_argument("--debug-invariants", action="store_true")
    parser.add_argument("--factory-args", default="{}", help="JSON passed to factory")
    args = parser.parse_args()

    bundle = load_symbol(args.factory)(json.loads(args.factory_args))
    model = bundle["model"]
    covariances, modes, metadata = load_geometry(
        args.geometry, expected_model=bundle.get("model_name"),
        expected_layers=args.target_layers,
        expected_hidden_size=bundle.get("hidden_size"))
    if modes is None:
        raise ValueError("geometry artifact has no hallucination modes")
    config = EllipsoidSteeringConfig(
        target_layers=args.target_layers, routing_mode=args.routing_mode,
        position_aware=args.routing_mode != "instance",
        routing_temperature=args.routing_temperature, steering_gain=args.steering_gain,
        steering_threshold=args.steering_threshold, beta_max=args.beta_max,
        spectral_gamma=args.spectral_gamma, hard_routing=args.hard_routing,
        spectral_weighting="hard" if args.hard_projection else "soft",
        preserve_mahalanobis_radius=not args.no_preserve_radius,
        debug_invariants=args.debug_invariants)
    controller = EllipsoidSteeringController(model, covariances, modes, config,
                                             layers=bundle.get("layers"))
    masked = controller.routing_pass(bundle["masked_inputs"])
    context = SteeringContext(masked, bundle.get("steering_masks"))
    with controller.enabled(context):
        with torch.inference_mode():
            output = model.generate(**bundle["original_inputs"],
                                    **bundle.get("generate_kwargs", {}))
    output_ids = output.sequences if hasattr(output, "sequences") else output
    payload = {"output_ids": output_ids.cpu(), "metadata": metadata,
               "diagnostics": {layer: [d.summary() for d in values]
                               for layer, values in context.diagnostics.items()}}
    torch.save(payload, args.output)
    print(json.dumps(payload["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
