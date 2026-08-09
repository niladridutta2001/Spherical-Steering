"""Fitting and matrix-free operations for Mahalanobis ellipsoid geometry."""

import numpy as np
import torch


GEOMETRIES = ("sphere", "ellipsoid-diag", "ellipsoid-lowrank")


def _validate_inputs(X, y, geometry, covariance_source, shrinkage,
                     variance_floor, cov_rank):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("X must be [N,D] and y must be [N]")
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry: {geometry}")
    if covariance_source not in ("pooled", "global"):
        raise ValueError("covariance_source must be 'pooled' or 'global'")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must satisfy 0 <= shrinkage <= 1")
    if variance_floor <= 0 or not np.isfinite(variance_floor):
        raise ValueError("variance_floor must be finite and > 0")
    if geometry == "ellipsoid-lowrank" and not 1 <= cov_rank < X.shape[1]:
        raise ValueError("cov_rank must satisfy 1 <= cov_rank < hidden_dim")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("training activations must be finite with binary labels")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("both classes are required")


def _floor(values, tau, epsilon):
    return np.maximum(values, max(epsilon * tau, epsilon))


def whiten_numpy(v, geometry):
    v = np.asarray(v)
    if "diag_var" in geometry:
        return v / np.sqrt(geometry["diag_var"])
    U, lam, residual = geometry["basis"], geometry["eigvals"], geometry["residual_var"]
    base = 1.0 / np.sqrt(residual)
    return v * base + (v @ U) @ (((1.0 / np.sqrt(lam)) - base)[:, None] * U.T)


def color_numpy(v, geometry):
    v = np.asarray(v)
    if "diag_var" in geometry:
        return v * np.sqrt(geometry["diag_var"])
    U, lam, residual = geometry["basis"], geometry["eigvals"], geometry["residual_var"]
    base = np.sqrt(residual)
    return v * base + (v @ U) @ ((np.sqrt(lam) - base)[:, None] * U.T)


def whiten_torch(v, geometry):
    if geometry.get("diag_var") is not None:
        return v / torch.sqrt(geometry["diag_var"])
    U, lam, residual = geometry["basis"], geometry["eigvals"], geometry["residual_var"]
    base = torch.rsqrt(residual)
    return v * base + ((v @ U) * (torch.rsqrt(lam) - base)) @ U.T


def color_torch(v, geometry):
    if geometry.get("diag_var") is not None:
        return v * torch.sqrt(geometry["diag_var"])
    U, lam, residual = geometry["basis"], geometry["eigvals"], geometry["residual_var"]
    base = torch.sqrt(residual)
    return v * base + ((v @ U) * (torch.sqrt(lam) - base)) @ U.T


def fit_ellipsoid_geometry(X, y, geometry="ellipsoid-diag",
                           covariance_source="pooled", shrinkage=0.1,
                           variance_floor=1e-5, cov_rank=128,
                           radius_quantiles=(0.5, 0.9, 0.95, 0.99)):
    """Fit geometry solely from the supplied training samples."""
    _validate_inputs(X, y, geometry, covariance_source, shrinkage,
                     variance_floor, cov_rank)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    m_T, m_H = X[y == 1].mean(0), X[y == 0].mean(0)
    center = X.mean(0)
    residuals = X - center if covariance_source == "global" else np.where(
        (y == 1)[:, None], X - m_T, X - m_H)
    denom = max(len(residuals) - (2 if covariance_source == "pooled" else 1), 1)
    result = {"center": center, "m_T": m_T, "m_H": m_H}

    if geometry == "ellipsoid-diag":
        raw = np.sum(residuals * residuals, axis=0) / denom
        tau = float(raw.mean())
        diag = _floor((1.0 - shrinkage) * raw + shrinkage * tau,
                      tau, variance_floor)
        result["diag_var"] = diag
        result["axis_sizes"] = np.sqrt(diag)
    elif geometry == "ellipsoid-lowrank":
        from sklearn.utils.extmath import randomized_svd
        effective_rank = min(cov_rank, min(residuals.shape) - 1)
        if effective_rank < 1:
            raise ValueError("not enough training samples for low-rank covariance")
        _, singular, basis_t = randomized_svd(
            residuals, n_components=effective_rank, random_state=0)
        basis = basis_t.T
        raw_eig = singular * singular / denom
        trace = float(np.sum(residuals * residuals) / denom)
        raw_residual = max((trace - raw_eig.sum()) / (X.shape[1] - effective_rank), 0.0)
        tau = trace / X.shape[1]
        eig = _floor((1.0 - shrinkage) * raw_eig + shrinkage * tau,
                     tau, variance_floor)
        residual = float(_floor(np.array((1.0 - shrinkage) * raw_residual +
                                         shrinkage * tau), tau, variance_floor))
        result.update(basis=basis, eigvals=eig, residual_var=residual,
                      axis_sizes=np.sqrt(eig), cov_rank=effective_rank)
    else:
        raise ValueError("fit_ellipsoid_geometry requires an ellipsoid geometry")

    direction_z = whiten_numpy(m_T - m_H, result)
    norm = np.linalg.norm(direction_z)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("contrastive mean difference has near-zero whitened norm")
    result["mu_T"] = direction_z / norm
    result["mu_H"] = -result["mu_T"]
    radii = np.linalg.norm(whiten_numpy(X - center, result), axis=1)
    quantiles = np.asarray(tuple(radius_quantiles), dtype=np.float64)
    if quantiles.ndim != 1 or len(quantiles) == 0 or np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("radius quantiles must lie in [0,1]")
    result["radius_quantile_levels"] = quantiles
    result["radius_quantiles"] = np.quantile(radii, quantiles)
    return result
