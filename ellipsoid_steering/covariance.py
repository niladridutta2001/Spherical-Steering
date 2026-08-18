from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class LowRankCovariance:
    """Actual PC variances plus isotropic variance on the orthogonal complement."""

    mean: torch.Tensor
    basis: torch.Tensor
    eigenvalues: torch.Tensor
    residual_variance: torch.Tensor
    eps: float = 1e-5
    sample_count: int = 0

    def __post_init__(self) -> None:
        d = self.mean.numel()
        if self.mean.ndim != 1 or self.basis.ndim != 2 or self.basis.shape[0] != d:
            raise ValueError("mean must be [D] and basis [D,R]")
        if self.eigenvalues.shape != (self.basis.shape[1],):
            raise ValueError("eigenvalues must have one value per basis column")
        if self.basis.shape[1] >= d:
            raise ValueError("low-rank basis rank must be smaller than hidden size")
        if not all(torch.isfinite(x).all() for x in
                   (self.mean, self.basis, self.eigenvalues, self.residual_variance)):
            raise ValueError("covariance contains non-finite values")
        if torch.any(self.eigenvalues <= 0) or self.residual_variance.item() <= 0:
            raise ValueError("all represented variances must be positive")

    @property
    def hidden_size(self) -> int:
        return self.mean.numel()

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    def to(self, device=None, dtype=None) -> "LowRankCovariance":
        return LowRankCovariance(
            self.mean.to(device=device, dtype=dtype),
            self.basis.to(device=device, dtype=dtype),
            self.eigenvalues.to(device=device, dtype=dtype),
            self.residual_variance.to(device=device, dtype=dtype),
            self.eps, self.sample_count)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "basis": self.basis,
                "eigenvalues": self.eigenvalues,
                "residual_variance": self.residual_variance,
                "eps": self.eps, "sample_count": self.sample_count}

    @classmethod
    def from_state_dict(cls, state: dict) -> "LowRankCovariance":
        return cls(**state)


class StreamingCovarianceEstimator:
    """Streaming mean/trace estimator with a bounded uniform reservoir for PCA."""

    def __init__(self, hidden_size: int, rank: int, eps: float = 1e-5,
                 residual_floor: float = 1e-5, max_pca_samples: int = 16384,
                 seed: int = 0) -> None:
        if not 1 <= rank < hidden_size:
            raise ValueError("rank must satisfy 1 <= rank < hidden_size")
        self.hidden_size, self.rank = hidden_size, rank
        self.eps, self.residual_floor = eps, residual_floor
        self.max_pca_samples = max_pca_samples
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.count = 0
        self.mean = torch.zeros(hidden_size, dtype=torch.float64)
        self.m2_trace = torch.zeros((), dtype=torch.float64)
        self._reservoir = torch.empty((max_pca_samples, hidden_size), dtype=torch.float32)
        self._reservoir_count = 0

    def update(self, batch: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Consume `[N,D]` or `[B,T,D]` activations without retaining the batch."""
        if batch.shape[-1] != self.hidden_size:
            raise ValueError("activation hidden size mismatch")
        values = batch.detach().reshape(-1, self.hidden_size).to("cpu", torch.float64)
        if mask is not None:
            flat_mask = mask.detach().reshape(-1).to("cpu", torch.bool)
            if len(flat_mask) != len(values):
                raise ValueError("mask does not match activation leading dimensions")
            values = values[flat_mask]
        if not len(values):
            return
        old_count = self.count
        batch_count = len(values)
        batch_mean = values.mean(dim=0)
        batch_m2 = (values-batch_mean).square().sum()
        total_count = old_count+batch_count
        delta = batch_mean-self.mean
        self.m2_trace += batch_m2 + delta.square().sum()*old_count*batch_count/total_count
        self.mean += delta*batch_count/total_count
        self.count = total_count
        for offset, row in enumerate(values, start=1):
            seen = old_count+offset
            if self._reservoir_count < self.max_pca_samples:
                self._reservoir[self._reservoir_count].copy_(row.float())
                self._reservoir_count += 1
            else:
                index = int(torch.randint(seen, (1,), generator=self.generator))
                if index < self.max_pca_samples:
                    self._reservoir[index].copy_(row.float())

    def finalize(self) -> LowRankCovariance:
        if self.count < 2 or self._reservoir_count <= self.rank:
            raise ValueError("insufficient samples for covariance rank")
        samples = self._reservoir[:self._reservoir_count].double() - self.mean
        _, singular, vh = torch.pca_lowrank(
            samples, q=min(self.rank + 8, min(samples.shape)),
            center=False, niter=4)
        # torch.pca_lowrank returns V in the third output, not V^T.
        basis = vh[:, :self.rank]
        eigenvalues = singular[:self.rank].square() / (self._reservoir_count - 1)
        total_trace = self.m2_trace / (self.count - 1)
        residual = (total_trace - eigenvalues.sum()).clamp_min(0) / (self.hidden_size-self.rank)
        scale = (total_trace / self.hidden_size).clamp_min(self.eps)
        floor = max(self.residual_floor, self.residual_floor * scale.item())
        eigenvalues = eigenvalues.clamp_min(max(self.eps, floor))
        residual = residual.clamp_min(max(self.eps, floor))
        return LowRankCovariance(self.mean.float(), basis.float(), eigenvalues.float(),
                                 residual.float(), self.eps, self.count)


def fit_streaming_covariance(batches: Iterable[torch.Tensor], hidden_size: int,
                             rank: int, **kwargs) -> LowRankCovariance:
    estimator = StreamingCovarianceEstimator(hidden_size, rank, **kwargs)
    for batch in batches:
        estimator.update(batch)
    return estimator.finalize()
