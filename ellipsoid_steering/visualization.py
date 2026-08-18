from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .covariance import LowRankCovariance
from .hallucination_modes import HallucinationMode
from .steering import steer_activation


def synthetic_2d_demo(output_path: str | Path | None = None) -> dict[str, float]:
    """Steer on `diag(9,1)` and optionally save a visualization."""
    cov = LowRankCovariance(
        mean=torch.zeros(2, dtype=torch.float64),
        basis=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        eigenvalues=torch.tensor([9.0], dtype=torch.float64),
        residual_variance=torch.tensor(1.0, dtype=torch.float64), eps=1e-12)
    mode = HallucinationMode(
        centroid=torch.tensor([0.0, 1.0], dtype=torch.float64),
        basis=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        eigenvalues=torch.tensor([4.0], dtype=torch.float64), sample_count=10)
    z = torch.tensor([[3.0, 1.0]], dtype=torch.float64)
    weights = torch.ones(1, 1, dtype=torch.float64)
    out, diag = steer_activation(z, cov, [mode], weights, beta=0.4,
                                 steering_threshold=0, beta_max=0.5,
                                 gamma=1e-2, return_diagnostics=True)
    values = {"radius_error": diag.relative_radius_error.max().item(),
              "energy_before": diag.energy_before.item(),
              "energy_after": diag.energy_after.item(),
              "euclidean_norm_change": (out.norm()-z.norm()).item()}
    if output_path is not None:
        theta = torch.linspace(0, 2*torch.pi, 400)
        ellipse = torch.stack((3*theta.cos(), theta.sin()), dim=1)
        plt.figure(figsize=(5, 5)); plt.plot(ellipse[:, 0], ellipse[:, 1], "k--")
        plt.scatter(*z[0], label="before"); plt.scatter(*out[0], label="after")
        plt.axis("equal"); plt.legend(); plt.title("Ellipsoid-constrained suppression")
        plt.savefig(output_path, dpi=160, bbox_inches="tight"); plt.close()
    return values
