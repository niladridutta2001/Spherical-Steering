from collections.abc import Sequence

import torch


def resolve_transformer_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    """Resolve common Hugging Face text-decoder layer containers."""
    candidates = (
        "model.layers", "language_model.model.layers", "language_model.layers",
        "model.language_model.layers", "transformer.h", "model.decoder.layers")
    for path in candidates:
        value = model
        try:
            for part in path.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        if isinstance(value, (torch.nn.ModuleList, list, tuple)):
            return value
    raise ValueError("could not resolve transformer layers; pass layers explicitly")


def extract_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_hidden(output, hidden: torch.Tensor):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]
