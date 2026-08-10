import numpy as np
import pytest
import torch

from ellipsoid_geometry import (fit_ellipsoid_geometry, whiten_numpy,
                                color_numpy, whiten_torch, color_torch)
from ellipsoidal_steering import (ellipsoidal_geometric_logic,
                                  ellipsoidal_baukit_hook_fn)
from spherical_steering import spherical_geometric_logic
from steering_artifacts import load_steering_artifact, build_intervention


def diag_case(d=8):
    return {"diag_var": torch.linspace(0.5, 2.0, d)}


def lowrank_case(d=8, k=3):
    q, _ = torch.linalg.qr(torch.randn(d, k))
    return {"basis": q, "eigvals": torch.linspace(0.4, 1.8, k),
            "residual_var": torch.tensor(0.9)}


@pytest.mark.parametrize("geometry", [diag_case(), lowrank_case()])
def test_mahalanobis_invariance_and_target_progress(geometry):
    torch.manual_seed(2)
    d = 8
    center = torch.randn(d)
    mu = torch.nn.functional.normalize(torch.randn(d), dim=0)
    h = center + color_torch(torch.randn(d), geometry)
    before = whiten_torch(h - center, geometry)
    out, triggered = ellipsoidal_geometric_logic(
        h, mu, -mu, center, geometry, 20.0, 0.8, -1.0)
    after = whiten_torch(out - center, geometry)
    assert triggered
    assert abs(before.norm().item() - after.norm().item()) < 1e-5
    assert torch.acos(torch.dot(after / after.norm(), mu)) <= torch.acos(torch.dot(before / before.norm(), mu)) + 1e-6


def test_identity_equivalence_to_spherical():
    torch.manual_seed(3)
    x = torch.randn(12)
    mu = torch.nn.functional.normalize(torch.randn(12), dim=0)
    spherical, a = spherical_geometric_logic(x, mu, -mu, 10.0, 0.6, -0.5)
    ellipsoid, b = ellipsoidal_geometric_logic(
        x, mu, -mu, torch.zeros(12), {"diag_var": torch.ones(12)},
        10.0, 0.6, -0.5)
    assert a == b
    torch.testing.assert_close(ellipsoid, spherical, atol=2e-6, rtol=2e-6)


def test_noop_gate_and_zero_strength():
    x = torch.tensor([1.0, 0.0, 0.0])
    mu = x.clone()
    args = (mu, -mu, torch.zeros(3), {"diag_var": torch.ones(3)}, 20.0)
    out, triggered = ellipsoidal_geometric_logic(x, *args, 0.8, 0.9)
    assert not triggered
    assert torch.equal(out, x)
    out, triggered = ellipsoidal_geometric_logic(-x, *args, 0.0, -1.0)
    assert not triggered
    assert torch.equal(out, -x)


@pytest.mark.parametrize("geometry", [diag_case(), lowrank_case()])
def test_whitening_roundtrip(geometry):
    v = torch.randn(5, 8)
    torch.testing.assert_close(color_torch(whiten_torch(v, geometry), geometry), v,
                               atol=2e-6, rtol=2e-6)


def test_lowrank_matches_explicit_covariance():
    geometry = lowrank_case(6, 2)
    U, lam, residual = geometry["basis"], geometry["eigvals"], geometry["residual_var"]
    covariance = U @ torch.diag(lam) @ U.T + residual * (torch.eye(6) - U @ U.T)
    eig, vec = torch.linalg.eigh(covariance)
    tau = torch.trace(covariance) / covariance.shape[0]
    W = vec @ torch.diag((eig / tau).rsqrt()) @ vec.T
    C = vec @ torch.diag((eig / tau).sqrt()) @ vec.T
    v = torch.randn(4, 6)
    torch.testing.assert_close(whiten_torch(v, geometry), v @ W, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(color_torch(v, geometry), v @ C, atol=2e-5, rtol=2e-5)


def test_numerical_edges_are_finite():
    mu = torch.tensor([1.0, 0.0, 0.0])
    geom = {"diag_var": torch.ones(3)}
    for x in (torch.zeros(3), mu, -mu, torch.tensor([-1.0, 1e-8, 0.0])):
        out, _ = ellipsoidal_geometric_logic(x, mu, -mu, torch.zeros(3), geom,
                                              20.0, 0.7, -1.0)
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_dtype_is_preserved(dtype):
    mu = torch.tensor([1.0, 0.0, 0.0])
    x = torch.tensor([-1.0, 0.1, 0.0], dtype=dtype)
    out, _ = ellipsoidal_geometric_logic(
        x, mu, -mu, torch.zeros(3), {"diag_var": torch.ones(3)},
        20.0, 0.5, -1.0)
    assert out.dtype == dtype and torch.isfinite(out).all()


def test_invalid_parameters_and_covariance_fail():
    mu = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError):
        ellipsoidal_geometric_logic(mu, mu, -mu, torch.zeros(2),
                                    {"diag_var": torch.tensor([1.0, 0.0])},
                                    20.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        ellipsoidal_geometric_logic(mu, mu, -mu, torch.zeros(2),
                                    {"diag_var": torch.ones(2)},
                                    float("nan"), 0.5, 0.0)


def test_hook_tensor_tuple_and_ranges():
    mu = torch.tensor([1.0, 0.0, 0.0])
    kwargs = dict(layer_name="x", mu_T=mu, mu_H=-mu, center=torch.zeros(3),
                  geometry={"diag_var": torch.ones(3)}, kappa=20.0,
                  alpha=0.5, beta=-1.0, stats={"total": 0, "steered": 0})
    h = torch.tensor([[[-1., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]]])
    out = ellipsoidal_baukit_hook_fn(h.clone(), start_idx=None, **kwargs)
    assert torch.equal(out[:, 0], h[:, 0]) and not torch.equal(out[:, -1], h[:, -1])
    tup = ellipsoidal_baukit_hook_fn((h.clone(), "cache"), start_idx=1, **kwargs)
    assert tup[1] == "cache" and torch.equal(tup[0][:, 0], h[:, 0])


def test_legacy_artifact(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, mu_T=np.array([1., 0.]), mu_H=np.array([-1., 0.]),
             fold_idx=np.array(0), test_q_indices=np.array([2, 3]))
    artifact = load_steering_artifact(path)
    assert artifact["geometry"] == "sphere"
    assert build_intervention(artifact, 20, .5, 0) is not None
    with pytest.raises(ValueError, match="requested ellipsoid"):
        build_intervention(artifact, 20, .5, 0, steering_geometry="ellipsoid")


@pytest.mark.parametrize("kind,rank", [("ellipsoid-diag", 3), ("ellipsoid-lowrank", 3)])
def test_fit_uses_only_passed_training_data_and_safe_artifact(tmp_path, kind, rank):
    rng = np.random.default_rng(4)
    train = rng.normal(size=(30, 7)); labels = np.tile([0, 1], 15)
    heldout_a = rng.normal(size=(10, 7)); heldout_b = heldout_a * 1000
    heldout_labels = np.tile([0, 1], 5)
    mask = np.arange(40) < 30
    all_a = np.concatenate([train, heldout_a])
    all_b = np.concatenate([train, heldout_b])
    all_labels = np.concatenate([labels, heldout_labels])
    fit_a = fit_ellipsoid_geometry(all_a[mask], all_labels[mask], kind, cov_rank=rank)
    fit_b = fit_ellipsoid_geometry(all_b[mask], all_labels[mask], kind, cov_rank=rank)
    for key in (("center", "diag_var", "mu_T") if kind.endswith("diag") else
                ("center", "basis", "eigvals", "residual_var", "mu_T")):
        np.testing.assert_array_equal(fit_a[key], fit_b[key])
    assert np.allclose(color_numpy(whiten_numpy(train, fit_a), fit_a), train)
