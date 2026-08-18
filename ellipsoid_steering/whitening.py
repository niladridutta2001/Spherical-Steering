import torch

from .covariance import LowRankCovariance


def _check(x: torch.Tensor, cov: LowRankCovariance) -> None:
    if x.shape[-1] != cov.hidden_size:
        raise ValueError("tensor hidden size does not match covariance")


def apply_cov_inv_sqrt(x: torch.Tensor, cov: LowRankCovariance) -> torch.Tensor:
    """Apply the linear covariance inverse square root; never subtracts a mean."""
    _check(x, cov)
    u = cov.basis.to(x)
    lam = cov.eigenvalues.to(x).clamp_min(cov.eps)
    residual = cov.residual_variance.to(x).clamp_min(cov.eps)
    coeff = x @ u
    parallel = coeff @ u.mT
    return (coeff * lam.rsqrt()) @ u.mT + (x-parallel) * residual.rsqrt()


def apply_cov_sqrt(q: torch.Tensor, cov: LowRankCovariance) -> torch.Tensor:
    """Apply the linear covariance square root without constructing `[D,D]`."""
    _check(q, cov)
    u = cov.basis.to(q)
    lam = cov.eigenvalues.to(q).clamp_min(cov.eps)
    residual = cov.residual_variance.to(q).clamp_min(cov.eps)
    coeff = q @ u
    parallel = coeff @ u.mT
    return (coeff * lam.sqrt()) @ u.mT + (q-parallel) * residual.sqrt()


def whiten(x: torch.Tensor, cov: LowRankCovariance) -> torch.Tensor:
    """Whiten centered activations by subtracting the grounded mean once."""
    return apply_cov_inv_sqrt(x-cov.mean.to(x), cov)


def unwhiten(q: torch.Tensor, cov: LowRankCovariance) -> torch.Tensor:
    return apply_cov_sqrt(q, cov) + cov.mean.to(q)


def whiten_delta(delta: torch.Tensor, cov: LowRankCovariance) -> torch.Tensor:
    """Whiten a displacement linearly. The activation mean is not subtracted."""
    return apply_cov_inv_sqrt(delta, cov)


def mahalanobis_norm(x: torch.Tensor, cov: LowRankCovariance,
                     centered: bool = False) -> torch.Tensor:
    q = apply_cov_inv_sqrt(x if centered else x-cov.mean.to(x), cov)
    return torch.linalg.vector_norm(q, dim=-1)
