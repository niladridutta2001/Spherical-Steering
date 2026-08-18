import torch

from .hooks import register_layer_hooks
from .utils import resolve_transformer_layers


class ActivationCollector:
    """Collect detached `[B,T,D]` residual-stream activations at selected layers."""

    def __init__(self, model: torch.nn.Module, target_layers: list[int],
                 layers=None, offload_to_cpu: bool = True) -> None:
        self.model, self.target_layers = model, target_layers
        self.layers = layers if layers is not None else resolve_transformer_layers(model)
        self.offload_to_cpu = offload_to_cpu
        self.activations: dict[int, torch.Tensor] = {}

    def _capture(self, layer: int, hidden: torch.Tensor):
        value = hidden.detach()
        self.activations[layer] = value.cpu() if self.offload_to_cpu else value

    def collect(self, model_inputs: dict, token_mask: torch.Tensor | None = None,
                **forward_kwargs) -> dict[int, torch.Tensor]:
        self.activations = {}
        with register_layer_hooks(self.layers, self.target_layers, self._capture):
            with torch.inference_mode():
                self.model(**model_inputs, **forward_kwargs)
        if token_mask is not None:
            result = {}
            for layer, values in self.activations.items():
                mask = token_mask.to(values.device)
                if mask.shape != values.shape[:-1]:
                    raise ValueError("token mask shape mismatch")
                result[layer] = values[mask]
            return result
        return dict(self.activations)

    def __call__(self, model_inputs: dict, **kwargs):
        return self.collect(model_inputs, **kwargs)
