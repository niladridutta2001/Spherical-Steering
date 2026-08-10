"""Mahalanobis ellipsoidal steering in whitened coordinates."""

from functools import partial
import math
import torch
import torch.nn.functional as F

from ellipsoid_geometry import whiten_torch, color_torch


def _validate(mu_T, mu_H, center, geometry, kappa, alpha, beta):
    tensors = [mu_T, mu_H, center]
    if any(not torch.isfinite(x).all() for x in tensors):
        raise ValueError("prototype and center values must be finite")
    if mu_T.shape != center.shape or mu_H.shape != center.shape:
        raise ValueError("prototype and center dimensions must match")
    if not torch.isclose(mu_T.norm(), torch.tensor(1.0, device=mu_T.device), atol=1e-4, rtol=1e-4):
        raise ValueError("truthful prototype must be a unit vector")
    if not all(math.isfinite(float(x)) for x in (kappa, alpha, beta)):
        raise ValueError("steering parameters must be finite")
    if beta >= 1:
        raise ValueError("beta must be < 1")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    if geometry.get("diag_var") is not None:
        if (geometry["diag_var"].shape != center.shape or
                not torch.isfinite(geometry["diag_var"]).all() or
                torch.any(geometry["diag_var"] <= 0)):
            raise ValueError("invalid diagonal covariance artifact")
    else:
        U, lam, residual = geometry.get("basis"), geometry.get("eigvals"), geometry.get("residual_var")
        if U is None or lam is None or residual is None or U.shape[0] != center.numel() or U.shape[1] != lam.numel():
            raise ValueError("invalid low-rank covariance artifact")
        if (not torch.isfinite(lam).all() or not torch.isfinite(residual) or
                torch.any(lam <= 0) or residual <= 0 or not torch.isfinite(U).all()):
            raise ValueError("invalid low-rank covariance values")


def _deterministic_tangent(x):
    indices = torch.argmin(torch.abs(x), dim=-1)
    e = F.one_hot(indices, num_classes=x.shape[-1]).to(x.dtype)
    tangent = e - (e * x).sum(-1, keepdim=True) * x
    return F.normalize(tangent, dim=-1)


def _steer_batch(h, mu_T, mu_H, center, geometry, kappa, alpha, beta):
    x = h.float()
    if not torch.isfinite(x).all():
        raise ValueError("activation values must be finite")
    finite = torch.isfinite(x).all(-1)
    z = whiten_torch(x - center, geometry)
    radius = torch.linalg.vector_norm(z, dim=-1)
    valid = finite & torch.isfinite(radius) & (radius > 1e-12)
    z_hat = z / radius.clamp_min(1e-12).unsqueeze(-1)
    cos_T = (z_hat * mu_T).sum(-1).clamp(-1.0, 1.0)
    cos_H = (z_hat * mu_H).sum(-1).clamp(-1.0, 1.0)
    probs = F.softmax(kappa * torch.stack((cos_T, cos_H), dim=-1), dim=-1)
    delta = probs[..., 1] - probs[..., 0]
    strength = (alpha * (delta - beta) / (1.0 - beta)).clamp(0.0, 1.0)
    trigger = valid & (delta > beta) & (strength > 0)

    theta = torch.acos(cos_T)
    tangent = mu_T - cos_T.unsqueeze(-1) * z_hat
    tangent_norm = torch.linalg.vector_norm(tangent, dim=-1)
    antipodal = (cos_T < -1.0 + 1e-6) | (tangent_norm < 1e-7)
    tangent_unit = tangent / tangent_norm.clamp_min(1e-12).unsqueeze(-1)
    tangent_unit = torch.where(antipodal.unsqueeze(-1), _deterministic_tangent(z_hat), tangent_unit)
    angle = strength * theta
    rotated = torch.cos(angle).unsqueeze(-1) * z_hat + torch.sin(angle).unsqueeze(-1) * tangent_unit
    rotated = F.normalize(rotated, dim=-1)
    candidate = center + color_torch(rotated * radius.unsqueeze(-1), geometry)
    out = torch.where(trigger.unsqueeze(-1), candidate, x)
    return out.to(h.dtype), trigger, radius, torch.linalg.vector_norm(
        whiten_torch(out.float() - center, geometry), dim=-1), theta, torch.acos(
            (F.normalize(whiten_torch(out.float() - center, geometry), dim=-1) * mu_T).sum(-1).clamp(-1, 1))


def ellipsoidal_geometric_logic(h, mu_T, mu_H, center, geometry,
                                 kappa, alpha, beta):
    """Steer one activation while preserving its Mahalanobis radius."""
    _validate(mu_T, mu_H, center, geometry, kappa, alpha, beta)
    out, trigger, *_ = _steer_batch(h.unsqueeze(0), mu_T, mu_H, center,
                                    geometry, kappa, alpha, beta)
    return out[0], bool(trigger.item())


def ellipsoidal_baukit_hook_fn(output, layer_name, mu_T, mu_H, center,
                                geometry, kappa, alpha, beta, stats=None,
                                start_idx=None, end_idx_exclusive=None):
    hidden = output[0] if isinstance(output, tuple) else output
    device = hidden.device
    mu_T, mu_H, center = (x.to(device=device, dtype=torch.float32)
                           for x in (mu_T, mu_H, center))
    geom = {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v)
            for k, v in geometry.items()}
    _validate(mu_T, mu_H, center, geom, kappa, alpha, beta)
    start = hidden.shape[1] - 1 if start_idx is None else max(0, min(start_idx, hidden.shape[1] - 1))
    end = hidden.shape[1] if end_idx_exclusive is None else max(start, min(end_idx_exclusive, hidden.shape[1]))
    selected = hidden[:, start:end, :]
    flat = selected.reshape(-1, selected.shape[-1])
    changed, trigger, before_r, after_r, before_a, after_a = _steer_batch(
        flat, mu_T, mu_H, center, geom, kappa, alpha, beta)
    hidden[:, start:end, :] = changed.reshape_as(selected)
    if stats is not None:
        stats["total"] = stats.get("total", 0) + trigger.numel()
        stats["steered"] = stats.get("steered", 0) + int(trigger.sum().item())
        stats["metric_radius_before_sum"] = stats.get("metric_radius_before_sum", 0.0) + float(before_r.sum())
        stats["metric_radius_after_sum"] = stats.get("metric_radius_after_sum", 0.0) + float(after_r.sum())
        rel = (after_r - before_r).abs() / before_r.clamp_min(1e-12)
        stats["max_relative_metric_radius_error"] = max(stats.get("max_relative_metric_radius_error", 0.0), float(rel.max()))
        stats["angle_before_sum"] = stats.get("angle_before_sum", 0.0) + float(before_a.sum())
        stats["angle_after_sum"] = stats.get("angle_after_sum", 0.0) + float(after_a.sum())
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden


def get_ellipsoidal_intervention(mu_T, mu_H, center, geometry,
                                  kappa=20.0, alpha=0.15, beta=0.1,
                                  stats=None):
    return partial(ellipsoidal_baukit_hook_fn, mu_T=mu_T, mu_H=mu_H,
                   center=center, geometry=geometry, kappa=kappa,
                   alpha=alpha, beta=beta, stats=stats)
