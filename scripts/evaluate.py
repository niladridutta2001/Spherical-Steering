"""Run a plug-in downstream evaluator on saved steering outputs."""

import argparse
import importlib
import json
import torch


def load_symbol(spec: str):
    module, symbol = spec.split(":", 1)
    return getattr(importlib.import_module(module), symbol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--evaluator", required=True,
                        help="`module:function` supporting POPE/CHAIR/etc.")
    parser.add_argument("--evaluator-args", default="{}")
    args = parser.parse_args()
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=True)
    metrics = load_symbol(args.evaluator)(predictions, json.loads(args.evaluator_args))
    print(json.dumps(metrics, indent=2, sort_keys=True))
