from dataclasses import dataclass

import numpy as np
import torch

from .covariance import LowRankCovariance
from .whitening import whiten_delta


@dataclass
class HallucinationMode:
    """A low-rank hallucination subspace in grounded-whitened coordinates."""

    centroid: torch.Tensor
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    sample_count: int

    def __post_init__(self) -> None:
        if self.centroid.ndim != 1 or self.basis.ndim != 2:
            raise ValueError("centroid must be [D] and basis [D,Rh]")
        if self.basis.shape[0] != self.centroid.numel():
            raise ValueError("mode hidden sizes disagree")
        if self.eigenvalues.shape != (self.basis.shape[1],):
            raise ValueError("mode eigenvalues do not match basis rank")
        if self.sample_count < 1 or torch.any(self.eigenvalues < 0):
            raise ValueError("invalid mode sample count or eigenvalues")
        if not all(torch.isfinite(x).all() for x in
                   (self.centroid, self.basis, self.eigenvalues)):
            raise ValueError("mode contains non-finite values")

    def to(self, device=None, dtype=None) -> "HallucinationMode":
        return HallucinationMode(self.centroid.to(device=device, dtype=dtype),
                                 self.basis.to(device=device, dtype=dtype),
                                 self.eigenvalues.to(device=device, dtype=dtype),
                                 self.sample_count)

    def state_dict(self) -> dict:
        return {"centroid": self.centroid, "basis": self.basis,
                "eigenvalues": self.eigenvalues, "sample_count": self.sample_count}

    @classmethod
    def from_state_dict(cls, state: dict) -> "HallucinationMode":
        return cls(**state)


def construct_whitened_deltas(z_grounded: torch.Tensor,
                               z_hallucinated: torch.Tensor,
                               covariance: LowRankCovariance) -> torch.Tensor:
    """Return `Sigma^-1/2 (z_minus-z_plus)` without mean subtraction."""
    if z_grounded.shape != z_hallucinated.shape:
        raise ValueError("paired grounded/hallucinated activation shapes differ")
    return whiten_delta(z_hallucinated-z_grounded, covariance)


def pool_local_window(activations: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Pool only a specified local token window, never the entire sequence implicitly."""
    if not 0 <= start < end <= activations.shape[-2]:
        raise ValueError("invalid local window")
    return activations[..., start:end, :].mean(dim=-2)


def fit_hallucination_modes(delta_white: torch.Tensor, num_modes: int,
                            rank: int, seed: int = 0) -> list[HallucinationMode]:
    """K-means followed by per-cluster low-rank PCA in whitened coordinates."""
    from sklearn.cluster import KMeans

    if delta_white.ndim != 2 or len(delta_white) < num_modes:
        raise ValueError("delta_white must be [N,D] with N >= num_modes")
    values = delta_white.detach().float().cpu()
    labels = KMeans(n_clusters=num_modes, random_state=seed, n_init=10).fit_predict(values.numpy())
    modes: list[HallucinationMode] = []
    for label in range(num_modes):
        cluster = values[torch.from_numpy(labels == label)]
        centroid = cluster.mean(dim=0)
        centered = cluster-centroid
        available = min(rank, centered.shape[0]-1, centered.shape[1])
        if available < 1:
            # A singleton still defines a directional centroid mode.
            direction = centroid / centroid.norm().clamp_min(1e-12)
            basis, eig = direction[:, None], torch.zeros(1)
        else:
            _, singular, basis = torch.pca_lowrank(
                centered, q=available, center=False, niter=4)
            eig = singular.square() / max(len(cluster)-1, 1)
        modes.append(HallucinationMode(centroid, basis[:, :available or 1],
                                       eig[:available or 1], len(cluster)))
    return modes


def apply_mode_operator(q: torch.Tensor, mode: HallucinationMode,
                        gamma: float, hard_projection: bool = False) -> torch.Tensor:
    """Apply `V g(Lambda) V^T q` without constructing a dense operator."""
    if gamma <= 0 or q.shape[-1] != mode.centroid.numel():
        raise ValueError("invalid gamma or hidden dimension")
    basis = mode.basis.to(q)
    eig = mode.eigenvalues.to(q)
    weights = torch.ones_like(eig) if hard_projection else eig/(eig+gamma)
    return ((q @ basis)*weights) @ basis.mT
