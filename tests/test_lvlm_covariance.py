import torch

from ellipsoid_steering.covariance import LowRankCovariance, StreamingCovarianceEstimator
from ellipsoid_steering.cache import save_geometry, load_geometry
from ellipsoid_steering.hallucination_modes import HallucinationMode


def make_covariance(d=12, rank=4, dtype=torch.float64):
    q, _ = torch.linalg.qr(torch.randn(d, rank, dtype=dtype))
    return LowRankCovariance(torch.randn(d, dtype=dtype), q,
                             torch.linspace(0.5, 3, rank, dtype=dtype),
                             torch.tensor(0.3, dtype=dtype), eps=1e-12)


def test_covariance_shapes_and_streaming_fit():
    torch.manual_seed(1)
    estimator = StreamingCovarianceEstimator(10, 3, max_pca_samples=100, seed=2)
    estimator.update(torch.randn(5, 7, 10), torch.ones(5, 7, dtype=torch.bool))
    covariance = estimator.finalize()
    assert covariance.mean.shape == (10,)
    assert covariance.basis.shape == (10, 3)
    assert covariance.sample_count == 35
    assert torch.all(covariance.eigenvalues > 0)
    assert covariance.residual_variance > 0


def test_covariance_serialization_roundtrip():
    covariance = make_covariance()
    restored = LowRankCovariance.from_state_dict(covariance.state_dict())
    torch.testing.assert_close(restored.mean, covariance.mean)
    torch.testing.assert_close(restored.basis, covariance.basis)


def test_cache_metadata_validation(tmp_path):
    covariance=make_covariance(dtype=torch.float32)
    mode=HallucinationMode(torch.zeros(12),covariance.basis[:,:1],torch.ones(1),4)
    path=tmp_path/'geometry.pt'; metadata={'model_name':'tiny','target_layers':[2],
        'hidden_size':12,'version':1}
    save_geometry(path,{2:covariance},{2:[mode]},metadata)
    covariances,modes,_=load_geometry(path,'tiny',[2],12)
    assert covariances[2].hidden_size==12 and len(modes[2])==1
    import pytest
    with pytest.raises(ValueError,match='model mismatch'):
        load_geometry(path,'wrong',[2],12)
