"""Fit per-layer grounded low-rank covariances from chunked activation tensors."""

import argparse
import glob
import importlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ellipsoid_steering.cache import save_geometry
from ellipsoid_steering.activation_collector import ActivationCollector
from ellipsoid_steering.covariance import StreamingCovarianceEstimator


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--activation-chunks", nargs="+",
                        help="Glob(s) of `.pt` dictionaries: layer -> [N,D] or [B,T,D]")
    source.add_argument("--factory", help="`module:function` returning model and grounded examples")
    parser.add_argument("--factory-args", default="{}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--target-layers", type=int, nargs="+", required=True)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--residual-floor", type=float, default=1e-5)
    parser.add_argument("--max-pca-samples", type=int, default=16384)
    parser.add_argument("--token-policy", default="text")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    estimators = {}
    hidden_size = None

    def consume(values_by_layer, masks=None):
        nonlocal hidden_size
        masks = masks or {}
        for layer in args.target_layers:
            values = values_by_layer[layer] if layer in values_by_layer else values_by_layer[str(layer)]
            if hidden_size is None:
                hidden_size = values.shape[-1]
            if values.shape[-1] != hidden_size:
                raise ValueError("hidden size changed across chunks")
            estimators.setdefault(layer, StreamingCovarianceEstimator(
                hidden_size, args.rank, args.eps, args.residual_floor,
                args.max_pca_samples, args.seed+layer))
            mask = masks.get(layer, masks.get(str(layer))) if isinstance(masks, dict) else None
            estimators[layer].update(values, mask)

    if args.activation_chunks:
        files = sorted({path for pattern in args.activation_chunks for path in glob.glob(pattern)})
        if not files:
            raise FileNotFoundError("no activation chunks matched")
        for path in files:
            chunk = torch.load(path, map_location="cpu", weights_only=True)
            consume(chunk.get("activations", chunk), chunk.get("masks", {}))
    else:
        module, symbol = args.factory.split(":", 1)
        bundle = getattr(importlib.import_module(module), symbol)(json.loads(args.factory_args))
        collector = ActivationCollector(bundle["model"], args.target_layers,
                                        bundle.get("layers"), offload_to_cpu=True)
        for example in bundle["examples"]:
            activations = collector.collect(example["model_inputs"])
            consume(activations, example.get("masks"))
    covariances = {layer: estimator.finalize() for layer, estimator in estimators.items()}
    metadata = {"model_name": args.model_name, "model_revision": args.model_revision,
                "target_layers": args.target_layers, "hidden_size": hidden_size,
                "covariance_rank": args.rank, "dataset_name": args.dataset_name,
                "token_policy": args.token_policy, "version": 1}
    save_geometry(args.output, covariances, None, metadata)
    print(f"Saved {len(covariances)} layer covariances to {args.output}")


if __name__ == "__main__":
    main()
