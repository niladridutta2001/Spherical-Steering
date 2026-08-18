from dataclasses import dataclass, field

import torch

from .covariance import LowRankCovariance
from .hallucination_modes import HallucinationMode, apply_mode_operator
from .mode_router import routing_entropy
from .whitening import apply_cov_inv_sqrt, apply_cov_sqrt


@dataclass
class SteeringDiagnostics:
    radius_before: torch.Tensor
    radius_after: torch.Tensor
    relative_radius_error: torch.Tensor
    energy_before: torch.Tensor
    energy_after: torch.Tensor
    beta: torch.Tensor
    routing_entropy: torch.Tensor
    dominant_modes: torch.Tensor
    steered_mask: torch.Tensor
    intervention_magnitude: torch.Tensor

    def summary(self) -> dict[str, float | list[int]]:
        selected = self.steered_mask
        counts = torch.bincount(self.dominant_modes.reshape(-1)).cpu().tolist()
        return {
            "mean_radius_before": self.radius_before.mean().item(),
            "mean_radius_after": self.radius_after.mean().item(),
            "max_relative_radius_error": self.relative_radius_error.max().item(),
            "mean_energy_before": self.energy_before.mean().item(),
            "mean_energy_after": self.energy_after.mean().item(),
            "mean_beta": self.beta.mean().item(), "max_beta": self.beta.max().item(),
            "mean_routing_entropy": self.routing_entropy.mean().item(),
            "dominant_mode_counts": counts,
            "fraction_tokens_steered": selected.float().mean().item(),
            "mean_intervention_magnitude": self.intervention_magnitude.mean().item(),
        }


def adaptive_mode_operator(q: torch.Tensor, modes: list[HallucinationMode],
                           routing_weights: torch.Tensor, gamma: float,
                           hard_projection: bool = False) -> torch.Tensor:
    if routing_weights.shape[-1] != len(modes):
        raise ValueError("routing mode count mismatch")
    result = torch.zeros_like(q)
    for k, mode in enumerate(modes):
        result = result + routing_weights[..., k, None]*apply_mode_operator(
            q, mode, gamma, hard_projection)
    return result


def hallucination_energy(q: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
    return (q*component).sum(dim=-1)


def steer_activation(z: torch.Tensor, covariance: LowRankCovariance,
                     modes: list[HallucinationMode], routing_weights: torch.Tensor,
                     beta: torch.Tensor | float | None = None,
                     steering_mask: torch.Tensor | None = None,
                     steering_gain: float = 1.0, steering_threshold: float = 0.05,
                     beta_max: float = 0.5, gamma: float = 1e-2,
                     preserve_radius: bool = True, hard_projection: bool = False,
                     eps: float = 1e-8, debug: bool = False,
                     return_diagnostics: bool = False):
    """Suppress routed hallucination components and optionally preserve radius."""
    if z.shape[-1] != covariance.hidden_size or routing_weights.shape[:-1] != z.shape[:-1]:
        raise ValueError("activation/routing shapes are incompatible")
    if steering_mask is None:
        steering_mask = torch.ones(z.shape[:-1], dtype=torch.bool, device=z.device)
    if steering_mask.shape != z.shape[:-1]:
        raise ValueError("steering_mask must match activation leading dimensions")

    mean = covariance.mean.to(z)
    q = apply_cov_inv_sqrt(z-mean, covariance)
    component = adaptive_mode_operator(q, modes, routing_weights, gamma, hard_projection)
    energy_before = hallucination_energy(q, component)
    radius_sq = q.square().sum(dim=-1)
    normalized_energy = (energy_before/(radius_sq+eps)).clamp_min(0)
    if beta is None:
        beta_value = steering_gain*normalized_energy
    else:
        beta_value = torch.as_tensor(beta, dtype=z.dtype, device=z.device)
        beta_value = torch.broadcast_to(beta_value, z.shape[:-1])
    beta_value = beta_value.clamp(0, beta_max)
    active = steering_mask & (normalized_energy >= steering_threshold) & (beta_value > 0)
    beta_value = torch.where(active, beta_value, torch.zeros_like(beta_value))

    candidate = q-beta_value[..., None]*component
    old_norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    new_norm = torch.linalg.vector_norm(candidate, dim=-1, keepdim=True)
    safe = (old_norm >= eps) & (new_norm >= eps)
    active = active & safe.squeeze(-1)
    beta_value = torch.where(active, beta_value, torch.zeros_like(beta_value))
    if preserve_radius:
        q_new = torch.where(safe, candidate*(old_norm/new_norm.clamp_min(eps)), q)
    else:
        q_new = torch.where(safe, candidate, q)
    z_candidate = mean+apply_cov_sqrt(q_new, covariance)
    z_out = torch.where(active[..., None], z_candidate, z)

    if not torch.isfinite(z_out).all():
        raise FloatingPointError("non-finite steered activation")
    radius_after = torch.linalg.vector_norm(
        apply_cov_inv_sqrt(z_out-mean, covariance), dim=-1)
    radius_before = old_norm.squeeze(-1)
    relative_error = (radius_after-radius_before).abs()/radius_before.clamp_min(eps)
    if debug and preserve_radius and active.any() and relative_error[active].max() >= 1e-3:
        raise AssertionError("Mahalanobis-radius invariant failed")

    if not return_diagnostics:
        return z_out
    q_out = apply_cov_inv_sqrt(z_out-mean, covariance)
    component_after = adaptive_mode_operator(q_out, modes, routing_weights, gamma, hard_projection)
    diagnostics = SteeringDiagnostics(
        radius_before, radius_after, relative_error, energy_before,
        hallucination_energy(q_out, component_after), beta_value,
        routing_entropy(routing_weights), routing_weights.argmax(dim=-1), active,
        torch.linalg.vector_norm(z_out-z, dim=-1)/torch.linalg.vector_norm(z, dim=-1).clamp_min(eps))
    return z_out, diagnostics
