"""Safe artifact loading and intervention construction."""

import numpy as np
import torch

from spherical_steering import get_spherical_intervention
from ellipsoidal_steering import get_ellipsoidal_intervention


def _scalar(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    return value.item() if np.asarray(value).ndim == 0 else value


def load_steering_artifact(path, device="cpu"):
    with np.load(path, allow_pickle=False) as data:
        geometry = str(_scalar(data, "geometry", "sphere"))
        if geometry not in ("sphere", "ellipsoid-diag", "ellipsoid-lowrank"):
            raise ValueError(f"unsupported artifact geometry: {geometry}")
        artifact = {"geometry": geometry,
                    "mu_T": torch.as_tensor(data["mu_T"], dtype=torch.float32, device=device),
                    "mu_H": torch.as_tensor(data["mu_H"], dtype=torch.float32, device=device)}
        if artifact["mu_T"].ndim != 1 or artifact["mu_H"].shape != artifact["mu_T"].shape:
            raise ValueError("artifact prototypes must be same-sized vectors")
        if not torch.isfinite(artifact["mu_T"]).all() or not torch.isfinite(artifact["mu_H"]).all():
            raise ValueError("artifact prototypes must be finite")
        for key in ("artifact_version", "covariance_source", "shrinkage",
                    "variance_floor", "cov_rank", "fold_idx"):
            artifact[key] = _scalar(data, key)
        if "test_q_indices" in data:
            artifact["test_q_indices"] = np.array(data["test_q_indices"])
        if geometry != "sphere":
            artifact["center"] = torch.as_tensor(data["center"], dtype=torch.float32, device=device)
            if geometry == "ellipsoid-diag":
                artifact["geometry_data"] = {"diag_var": torch.as_tensor(data["diag_var"], dtype=torch.float32, device=device)}
            else:
                artifact["geometry_data"] = {
                    "basis": torch.as_tensor(data["basis"], dtype=torch.float32, device=device),
                    "eigvals": torch.as_tensor(data["eigvals"], dtype=torch.float32, device=device),
                    "residual_var": torch.as_tensor(data["residual_var"], dtype=torch.float32, device=device)}
            if artifact["center"].shape != artifact["mu_T"].shape:
                raise ValueError("artifact center and prototypes have inconsistent dimensions")
            geom = artifact["geometry_data"]
            if geometry == "ellipsoid-diag":
                if (geom["diag_var"].shape != artifact["center"].shape or
                        not torch.isfinite(geom["diag_var"]).all() or
                        torch.any(geom["diag_var"] <= 0)):
                    raise ValueError("invalid diagonal covariance artifact")
            elif (geom["basis"].ndim != 2 or geom["basis"].shape[0] != artifact["center"].numel()
                  or geom["basis"].shape[1] != geom["eigvals"].numel()
                  or not torch.isfinite(geom["basis"]).all()
                  or not torch.isfinite(geom["eigvals"]).all()
                  or not torch.isfinite(geom["residual_var"])
                  or torch.any(geom["eigvals"] <= 0) or geom["residual_var"] <= 0):
                raise ValueError("invalid low-rank covariance artifact")
        return artifact


def resolve_steering_geometry(artifact, requested="auto"):
    actual = "sphere" if artifact["geometry"] == "sphere" else "ellipsoid"
    if requested not in ("auto", "sphere", "ellipsoid"):
        raise ValueError("steering geometry must be auto, sphere, or ellipsoid")
    if requested != "auto" and requested != actual:
        raise ValueError(f"requested {requested} steering but artifact uses {artifact['geometry']}")
    return actual


def build_intervention(artifact, kappa, alpha, beta, stats=None,
                       steering_geometry="auto"):
    actual = resolve_steering_geometry(artifact, steering_geometry)
    if actual == "sphere":
        return get_spherical_intervention(artifact["mu_T"], artifact["mu_H"],
                                          kappa, alpha, beta, stats=stats)
    return get_ellipsoidal_intervention(
        artifact["mu_T"], artifact["mu_H"], artifact["center"],
        artifact["geometry_data"], kappa, alpha, beta, stats=stats)


def evaluation_diagnostics(artifact, stats):
    total = stats.get("total", 0)
    ellipsoid = artifact["geometry"] != "sphere"
    return {
        "steering_geometry": artifact["geometry"],
        "covariance_source": artifact.get("covariance_source"),
        "cov_rank": artifact.get("cov_rank"),
        "shrinkage": artifact.get("shrinkage"),
        "variance_floor": artifact.get("variance_floor"),
        "trigger_rate": stats.get("steered", 0) / total if total else 0.0,
        "mean_mahalanobis_radius_before": stats.get("radius_before_sum", 0.0) / total if ellipsoid and total else None,
        "mean_mahalanobis_radius_after": stats.get("radius_after_sum", 0.0) / total if ellipsoid and total else None,
        "max_relative_radius_error": stats.get("max_relative_radius_error") if ellipsoid and total else None,
        "mean_angle_to_truth_before": stats.get("angle_before_sum", 0.0) / total if ellipsoid and total else None,
        "mean_angle_to_truth_after": stats.get("angle_after_sum", 0.0) / total if ellipsoid and total else None,
    }
