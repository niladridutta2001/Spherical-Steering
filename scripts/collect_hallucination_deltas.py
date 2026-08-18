"""Whiten aligned or local-window grounded/hallucinated displacement chunks."""

import argparse
import glob
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ellipsoid_steering.cache import load_geometry
from ellipsoid_steering.hallucination_modes import construct_whitened_deltas, pool_local_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-chunks", nargs="+", required=True)
    parser.add_argument("--covariances", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("aligned", "local-window"), default="aligned")
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    covariances, _, metadata = load_geometry(args.covariances)
    files = sorted({p for pattern in args.paired_chunks for p in glob.glob(pattern)})
    if not files:
        raise FileNotFoundError("no paired chunks matched")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    buffers = {layer: [] for layer in covariances}
    id_buffers = {layer: [] for layer in covariances}
    position_buffers = {layer: [] for layer in covariances}
    counters = {layer: 0 for layer in covariances}

    def flush(layer: int) -> None:
        if not buffers[layer]: return
        values = torch.cat(buffers[layer])
        example_ids = torch.cat(id_buffers[layer])
        positions = torch.cat(position_buffers[layer])
        for start in range(0, len(values), args.chunk_size):
            path = output/f"layer{layer}_deltas_{counters[layer]:05d}.pt"
            torch.save({"layer": layer, "delta_white": values[start:start+args.chunk_size],
                        "example_ids": example_ids[start:start+args.chunk_size],
                        "token_positions": positions[start:start+args.chunk_size],
                        "metadata": metadata}, path)
            counters[layer] += 1
        buffers[layer].clear(); id_buffers[layer].clear(); position_buffers[layer].clear()

    for path in files:
        chunk = torch.load(path, map_location="cpu", weights_only=True)
        for layer, covariance in covariances.items():
            plus = chunk["grounded"].get(layer, chunk["grounded"].get(str(layer)))
            minus = chunk["hallucinated"].get(layer, chunk["hallucinated"].get(str(layer)))
            if plus is None or minus is None:
                raise ValueError(f"chunk lacks layer {layer}")
            if args.mode == "local-window":
                plus_window = chunk["grounded_window"]
                minus_window = chunk["hallucinated_window"]
                plus = pool_local_window(plus, *plus_window)
                minus = pool_local_window(minus, *minus_window)
            original_shape = plus.shape[:-1]
            delta = construct_whitened_deltas(plus, minus, covariance).reshape(-1, covariance.hidden_size)
            supplied_ids = chunk.get("example_ids")
            if supplied_ids is None:
                batch = original_shape[0] if original_shape else len(delta)
                supplied_ids = torch.arange(batch)
            supplied_ids = torch.as_tensor(supplied_ids).reshape(-1)
            if len(original_shape) >= 2:
                ids = supplied_ids[:, None].expand(original_shape).reshape(-1)
                positions = torch.arange(original_shape[-1])[None, :].expand(original_shape).reshape(-1)
            else:
                ids = supplied_ids[:len(delta)]
                positions = torch.full((len(delta),),
                    int(chunk.get("grounded_window", (0, 1))[0]) if args.mode == "local-window" else 0)
            buffers[layer].append(delta.cpu())
            id_buffers[layer].append(ids.cpu()); position_buffers[layer].append(positions.cpu())
            if sum(len(x) for x in buffers[layer]) >= args.chunk_size:
                flush(layer)
    for layer in buffers: flush(layer)
    print(f"Saved whitened displacement chunks to {output}")


if __name__ == "__main__":
    main()
