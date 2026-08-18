"""Cluster whitened hallucination displacements and fit per-mode PCA bases."""

import argparse
import glob
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ellipsoid_steering.cache import load_geometry, save_geometry
from ellipsoid_steering.hallucination_modes import fit_hallucination_modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-chunks", nargs="+", required=True)
    parser.add_argument("--covariances", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--hallucination-rank", type=int, default=16)
    parser.add_argument("--max-samples-per-layer", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    covariances, _, metadata = load_geometry(args.covariances)
    files = sorted({p for pattern in args.delta_chunks for p in glob.glob(pattern)})
    per_layer = {layer: [] for layer in covariances}
    for path in files:
        chunk = torch.load(path, map_location="cpu", weights_only=True)
        layer = int(chunk["layer"])
        if layer in per_layer:
            per_layer[layer].append(chunk["delta_white"])
    modes = {}
    generator = torch.Generator().manual_seed(args.seed)
    for layer, chunks in per_layer.items():
        if not chunks: raise ValueError(f"no displacement chunks for layer {layer}")
        values = torch.cat(chunks)
        if len(values) > args.max_samples_per_layer:
            index = torch.randperm(len(values), generator=generator)[:args.max_samples_per_layer]
            values = values[index]
        modes[layer] = fit_hallucination_modes(
            values, args.num_modes, args.hallucination_rank, args.seed+layer)
    metadata.update(num_modes=args.num_modes, hallucination_rank=args.hallucination_rank)
    save_geometry(args.output, covariances, modes, metadata)
    print(f"Saved hallucination modes to {args.output}")


if __name__ == "__main__":
    main()
