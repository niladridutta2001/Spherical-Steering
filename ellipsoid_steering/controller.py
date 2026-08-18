from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch

from .activation_collector import ActivationCollector
from .config import EllipsoidSteeringConfig
from .covariance import LowRankCovariance
from .hallucination_modes import HallucinationMode
from .hooks import register_layer_hooks
from .mode_router import route_modes
from .steering import SteeringDiagnostics, steer_activation
from .utils import resolve_transformer_layers
from .whitening import whiten_delta


@dataclass
class SteeringContext:
    """Per-example two-pass routing state."""

    masked_activations: dict[int, torch.Tensor]
    steering_masks: dict[int, torch.Tensor] | torch.Tensor | None = None
    fixed_routing_weights: dict[int, torch.Tensor] | None = None
    diagnostics: dict[int, list[SteeringDiagnostics]] = field(default_factory=dict)


class EllipsoidSteeringController:
    """Generic hook controller; no parameters, gradients, or optimizers are created."""

    def __init__(self, model: torch.nn.Module,
                 covariances: dict[int, LowRankCovariance],
                 hallucination_modes: dict[int, list[HallucinationMode]],
                 config: EllipsoidSteeringConfig, layers=None) -> None:
        self.model, self.covariances, self.modes, self.config = (
            model, covariances, hallucination_modes, config)
        self.layers = layers if layers is not None else resolve_transformer_layers(model)
        shared_cov = {-1} if not config.layer_aware and -1 in covariances else set()
        shared_modes = {-1} if not config.layer_aware and -1 in hallucination_modes else set()
        missing_cov = set(config.target_layers)-set(covariances) if not shared_cov else set()
        missing_modes = set(config.target_layers)-set(hallucination_modes) if not shared_modes else set()
        if missing_cov or missing_modes:
            raise ValueError(f"missing covariances {sorted(missing_cov)} or modes {sorted(missing_modes)}")
        for layer in config.target_layers:
            covariance = covariances.get(layer, covariances.get(-1))
            layer_modes = hallucination_modes.get(layer, hallucination_modes.get(-1))
            if covariance is not None:
                if not layer_modes:
                    raise ValueError(f"layer {layer} has no hallucination modes")
                if any(mode.centroid.numel() != covariance.hidden_size
                       for mode in layer_modes):
                    raise ValueError(f"layer {layer} mode/covariance hidden-size mismatch")

    def _covariance(self, layer: int) -> LowRankCovariance:
        return self.covariances.get(layer, self.covariances.get(-1))

    def _layer_modes(self, layer: int) -> list[HallucinationMode]:
        return self.modes.get(layer, self.modes.get(-1))

    def routing_pass(self, masked_inputs: dict) -> dict[int, torch.Tensor]:
        """Pass A: collect perturbed-image prompt activations without steering."""
        collector = ActivationCollector(self.model, self.config.target_layers,
                                        self.layers, offload_to_cpu=False)
        return collector.collect(masked_inputs)

    def _mask(self, context: SteeringContext, layer: int,
              hidden: torch.Tensor) -> torch.Tensor:
        masks = context.steering_masks
        if masks is None:
            return torch.ones(hidden.shape[:-1], dtype=torch.bool, device=hidden.device)
        mask = masks[layer] if isinstance(masks, dict) else masks
        if mask.shape != hidden.shape[:-1]:
            if self.config.routing_mode == "instance" and mask.shape[0] == hidden.shape[0]:
                # During cached generation, extend the final prompt policy to new tokens.
                if hidden.shape[1] == 1:
                    mask = mask[:, -1:]
                elif hidden.shape[1] < mask.shape[1]:
                    mask = mask[:, :hidden.shape[1]]
                else:
                    fill = mask[:, -1:].expand(hidden.shape[0], hidden.shape[1]-mask.shape[1])
                    mask = torch.cat((mask, fill), dim=1)
            else:
                raise ValueError("steering mask is not aligned with current sequence")
        return mask.to(hidden.device)

    def _routing(self, context: SteeringContext, layer: int,
                 hidden: torch.Tensor) -> torch.Tensor:
        if context.fixed_routing_weights and layer in context.fixed_routing_weights:
            weights = context.fixed_routing_weights[layer].to(hidden.device)
        else:
            masked = context.masked_activations[layer].to(hidden.device, hidden.dtype)
            if self.config.routing_mode != "instance" and masked.shape != hidden.shape:
                raise ValueError("masked/original trajectories are not token aligned; use instance routing")
            length = min(masked.shape[-2], hidden.shape[-2])
            signal = whiten_delta(hidden[..., :length, :]-masked[..., :length, :],
                                   self._covariance(layer))
            weights = route_modes(
                signal, self._layer_modes(layer), self.config.routing_temperature,
                position_aware=self.config.position_aware and self.config.routing_mode != "instance",
                hard=self.config.hard_routing)
            if self.config.routing_mode == "instance":
                if context.fixed_routing_weights is None:
                    context.fixed_routing_weights = {}
                context.fixed_routing_weights[layer] = weights.detach()
        if weights.shape[-2] == 1:
            weights = weights.expand(*hidden.shape[:-2], hidden.shape[-2], weights.shape[-1])
        elif weights.shape[:-1] != hidden.shape[:-1]:
            raise ValueError("routing weights are not aligned with current activation")
        return weights

    def _edit(self, context: SteeringContext, layer: int,
              hidden: torch.Tensor) -> torch.Tensor:
        covariance = self._covariance(layer).to(hidden.device, torch.float32)
        modes = [mode.to(hidden.device, torch.float32) for mode in self._layer_modes(layer)]
        work = hidden.float()
        weights = self._routing(context, layer, work)
        output, diagnostics = steer_activation(
            work, covariance, modes, weights,
            beta=self.config.uniform_beta, steering_mask=self._mask(context, layer, work),
            steering_gain=self.config.steering_gain,
            steering_threshold=self.config.steering_threshold,
            beta_max=self.config.beta_max, gamma=self.config.spectral_gamma,
            preserve_radius=self.config.preserve_mahalanobis_radius,
            hard_projection=self.config.spectral_weighting == "hard",
            debug=self.config.debug_invariants, return_diagnostics=True)
        context.diagnostics.setdefault(layer, []).append(diagnostics)
        return output.to(hidden.dtype)

    @contextmanager
    def enabled(self, context: SteeringContext):
        """Pass B context: edit only target-layer residual streams."""
        def callback(layer, hidden):
            return self._edit(context, layer, hidden)

        with register_layer_hooks(self.layers, self.config.target_layers, callback, edit=True):
            with torch.inference_mode():
                yield context
