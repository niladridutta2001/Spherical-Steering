"""Weighted fitting and matrix-free powered Mahalanobis geometry."""

import hashlib
import numpy as np
import torch


GEOMETRIES = ("sphere", "ellipsoid-diag", "ellipsoid-lowrank")
CENTER_MODES = ("zero", "global", "class-midpoint")


def effective_weight_denominator(weights):
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if total <= 0 or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("sample weights must be finite, nonnegative, and have positive sum")
    denom = total - np.dot(weights, weights) / total
    if denom <= 0:
        raise ValueError("effective weighted covariance denominator is nonpositive")
    return float(denom)


def weighted_mean(X, weights):
    X, weights = np.asarray(X, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    return np.sum(X * weights[:, None], axis=0) / weights.sum()


def class_statistics(X, y, weights=None, center_mode="global"):
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    weights = np.ones(len(X), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    m_T = weighted_mean(X[y == 1], weights[y == 1])
    m_H = weighted_mean(X[y == 0], weights[y == 0])
    if center_mode == "zero":
        center = np.zeros(X.shape[1], dtype=np.float64)
    elif center_mode == "global":
        center = weighted_mean(X, weights)
    elif center_mode == "class-midpoint":
        center = 0.5 * (m_T + m_H)
    else:
        raise ValueError(f"unknown center_mode: {center_mode}")
    return m_T, m_H, center


def _validate(X, y, geometry, covariance_source, shrinkage, variance_floor,
              cov_rank, center_mode, whitening_power, weights):
    if X.ndim != 2 or y.shape != (len(X),) or weights.shape != (len(X),):
        raise ValueError("X must be [N,D], y and weights must be [N]")
    if geometry not in GEOMETRIES or center_mode not in CENTER_MODES:
        raise ValueError("invalid geometry or center mode")
    if covariance_source not in ("pooled", "global"):
        raise ValueError("covariance_source must be pooled or global")
    if not 0 <= shrinkage <= 1 or variance_floor <= 0:
        raise ValueError("invalid shrinkage or variance floor")
    if not 0 <= whitening_power <= 0.5:
        raise ValueError("whitening_power must satisfy 0 <= p <= 0.5")
    if geometry == "ellipsoid-lowrank" and not 1 <= cov_rank < X.shape[1]:
        raise ValueError("cov_rank must satisfy 1 <= rank < hidden_dim")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("activations must be finite with binary labels")
    if not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("both classes are required")
    if weights[y == 0].sum() <= 0 or weights[y == 1].sum() <= 0:
        raise ValueError("both classes must have positive total weight")
    effective_weight_denominator(weights)


def _variance_scale(geometry, dimension=None):
    if geometry.get("variance_scale") is not None:
        return geometry["variance_scale"]
    if "diag_var" in geometry:
        return np.asarray(geometry["diag_var"]).mean()
    d = dimension or geometry["basis"].shape[0]
    k = len(geometry["eigvals"])
    return (np.sum(geometry["eigvals"]) + (d - k) * geometry["residual_var"]) / d


def _torch_scale(geometry, v):
    scale = geometry.get("variance_scale")
    if scale is None:
        if geometry.get("diag_var") is not None:
            scale = geometry["diag_var"].mean()
        else:
            d, k = geometry["basis"].shape
            scale = (geometry["eigvals"].sum() + (d-k)*geometry["residual_var"]) / d
    return torch.as_tensor(scale, dtype=v.dtype, device=v.device)


def whiten_numpy(v, geometry):
    v = np.asarray(v)
    p = float(geometry.get("whitening_power", 0.5))
    tau = float(_variance_scale(geometry, v.shape[-1]))
    if "diag_var" in geometry:
        return v * np.power(np.asarray(geometry["diag_var"]) / tau, -p)
    U, lam, residual = geometry["basis"], geometry["eigvals"], geometry["residual_var"]
    a = np.power(lam / tau, -p); a_perp = float((residual / tau) ** (-p))
    return a_perp * v + (v @ U) @ ((a-a_perp)[:, None] * U.T)


def color_numpy(v, geometry):
    v = np.asarray(v)
    p = float(geometry.get("whitening_power", 0.5))
    tau = float(_variance_scale(geometry, v.shape[-1]))
    if "diag_var" in geometry:
        return v * np.power(np.asarray(geometry["diag_var"]) / tau, p)
    U, lam, residual = geometry["basis"], geometry["eigvals"], geometry["residual_var"]
    a = np.power(lam / tau, p); a_perp = float((residual / tau) ** p)
    return a_perp * v + (v @ U) @ ((a-a_perp)[:, None] * U.T)


def whiten_torch(v, geometry):
    p = float(geometry.get("whitening_power", 0.5)); tau = _torch_scale(geometry, v)
    if geometry.get("diag_var") is not None:
        return v * torch.pow(geometry["diag_var"] / tau, -p)
    U, lam, residual = geometry["basis"], geometry["eigvals"], geometry["residual_var"]
    a, ap = torch.pow(lam/tau, -p), torch.pow(residual/tau, -p)
    return ap*v + ((v@U)*(a-ap))@U.T


def color_torch(v, geometry):
    p = float(geometry.get("whitening_power", 0.5)); tau = _torch_scale(geometry, v)
    if geometry.get("diag_var") is not None:
        return v * torch.pow(geometry["diag_var"] / tau, p)
    U, lam, residual = geometry["basis"], geometry["eigvals"], geometry["residual_var"]
    a, ap = torch.pow(lam/tau, p), torch.pow(residual/tau, p)
    return ap*v + ((v@U)*(a-ap))@U.T


def fitting_question_hash(question_ids):
    return hashlib.sha256(np.sort(np.unique(question_ids)).astype(np.int64).tobytes()).hexdigest()[:16]


def fit_raw_spectral(X, y, weights=None, covariance_source="pooled",
                     center_mode="global", max_rank=128, random_state=0):
    """Fit an unshrunk maximum-rank spectrum suitable for sweep caching."""
    from sklearn.utils.extmath import randomized_svd
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    weights = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=np.float64)
    m_T, m_H, center = class_statistics(X, y, weights, center_mode)
    residuals = X-center if covariance_source == "global" else np.where(
        (y == 1)[:, None], X-m_T, X-m_H)
    denom = effective_weight_denominator(weights)
    weighted = residuals * np.sqrt(weights / denom)[:, None]
    effective_rank = min(max_rank, min(weighted.shape)-1)
    if effective_rank < 1:
        raise ValueError("not enough samples for low-rank fitting")
    _, singular, basis_t = randomized_svd(weighted, n_components=effective_rank,
                                           random_state=random_state)
    eigvals = singular**2
    trace = float(np.sum(weighted**2))
    return dict(basis=basis_t.T, raw_eigvals=eigvals, total_covariance_trace=trace,
                m_T=m_T, m_H=m_H, centers={
                    "zero": np.zeros(X.shape[1]), "global": weighted_mean(X, weights),
                    "class-midpoint": 0.5*(m_T+m_H)}, denominator=denom)


def derive_lowrank_geometry(raw, rank, shrinkage=0.1, variance_floor=1e-5,
                            center_mode="global", whitening_power=0.5):
    d = raw["basis"].shape[0]
    if not 1 <= rank <= raw["basis"].shape[1] or rank >= d:
        raise ValueError("requested rank unavailable in spectral cache")
    basis = raw["basis"][:, :rank]; raw_eig = raw["raw_eigvals"][:rank]
    trace = raw["total_covariance_trace"]
    raw_residual = max((trace-raw_eig.sum())/(d-rank), 0.0); tau_raw = trace/d
    floor = max(variance_floor*tau_raw, variance_floor)
    eig = np.maximum((1-shrinkage)*raw_eig+shrinkage*tau_raw, floor)
    residual = float(max((1-shrinkage)*raw_residual+shrinkage*tau_raw, floor))
    tau = float((eig.sum()+(d-rank)*residual)/d)
    geom = dict(center=raw["centers"][center_mode], m_T=raw["m_T"], m_H=raw["m_H"],
                basis=basis, eigvals=eig, residual_var=residual,
                variance_scale=tau, whitening_power=whitening_power,
                axis_sizes=np.sqrt(eig), cov_rank=rank)
    direction = whiten_numpy(raw["m_T"]-raw["m_H"], geom)
    n = np.linalg.norm(direction)
    if n < 1e-12 or not np.isfinite(n): raise ValueError("near-zero whitened contrast")
    geom.update(mu_T=direction/n, mu_H=-direction/n)
    return geom


def fit_ellipsoid_geometry(X, y, geometry="ellipsoid-diag", covariance_source="pooled",
                           shrinkage=0.1, variance_floor=1e-5, cov_rank=128,
                           radius_quantiles=(.5,.9,.95,.99), sample_weights=None,
                           center_mode="global", whitening_power=0.5):
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
    weights = np.ones(len(X)) if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)
    _validate(X, y, geometry, covariance_source, shrinkage, variance_floor,
              cov_rank, center_mode, whitening_power, weights)
    m_T, m_H, center = class_statistics(X, y, weights, center_mode)
    residuals = X-center if covariance_source == "global" else np.where(
        (y == 1)[:,None], X-m_T, X-m_H)
    denom = effective_weight_denominator(weights)
    if geometry == "ellipsoid-diag":
        raw = np.sum(weights[:,None]*residuals**2, axis=0)/denom; tau_raw=float(raw.mean())
        floor=max(variance_floor*tau_raw, variance_floor)
        diag=np.maximum((1-shrinkage)*raw+shrinkage*tau_raw, floor)
        result=dict(center=center,m_T=m_T,m_H=m_H,diag_var=diag,
                    variance_scale=float(diag.mean()),axis_sizes=np.sqrt(diag),
                    whitening_power=whitening_power)
    elif geometry == "ellipsoid-lowrank":
        raw=fit_raw_spectral(X,y,weights,covariance_source,center_mode,cov_rank)
        result=derive_lowrank_geometry(raw,min(cov_rank,raw["basis"].shape[1]),
                                       shrinkage,variance_floor,center_mode,whitening_power)
    else: raise ValueError("ellipsoid geometry required")
    direction=whiten_numpy(m_T-m_H,result); n=np.linalg.norm(direction)
    if n<1e-12 or not np.isfinite(n): raise ValueError("near-zero whitened contrast")
    result.update(mu_T=direction/n,mu_H=-direction/n)
    radii=np.linalg.norm(whiten_numpy(X-center,result),axis=1)
    levels=np.asarray(tuple(radius_quantiles),dtype=np.float64)
    if levels.ndim!=1 or not len(levels) or np.any((levels<0)|(levels>1)):
        raise ValueError("invalid radius quantiles")
    result.update(radius_quantile_levels=levels,radius_quantiles=np.quantile(radii,levels))
    return result
