import torch

from ellipsoid_steering.covariance import LowRankCovariance
from ellipsoid_steering.hallucination_modes import HallucinationMode
from ellipsoid_steering.steering import steer_activation
from ellipsoid_steering.whitening import mahalanobis_norm


def fixture(dtype=torch.float64, variance=4.0):
    covariance = LowRankCovariance(
        torch.tensor([.2,-.1,1.], dtype=dtype),
        torch.tensor([[1.,0.],[0.,1.],[0.,0.]], dtype=dtype),
        torch.tensor([variance, 2.], dtype=dtype), torch.tensor(0.5, dtype=dtype), eps=1e-12)
    mode = HallucinationMode(torch.zeros(3, dtype=dtype),
        torch.tensor([[0.],[1.],[0.]], dtype=dtype), torch.tensor([3.], dtype=dtype), 20)
    return covariance, [mode]


def test_mahalanobis_preservation_and_energy_reduction():
    covariance, modes = fixture(); z = torch.tensor([[[2.,2.,2.]]], dtype=torch.float64)
    out, diagnostics = steer_activation(
        z, covariance, modes, torch.ones(1,1,1, dtype=z.dtype), beta=.35,
        steering_threshold=0, beta_max=.5, return_diagnostics=True, debug=True)
    torch.testing.assert_close(mahalanobis_norm(z,covariance), mahalanobis_norm(out,covariance),
                               rtol=1e-8, atol=1e-9)
    assert diagnostics.energy_after.item() < diagnostics.energy_before.item()


def test_beta_zero_and_false_mask_are_exact_identity():
    covariance, modes = fixture(); z = torch.randn(2,4,3, dtype=torch.float64)
    weights = torch.ones(2,4,1, dtype=z.dtype)
    assert torch.equal(steer_activation(z,covariance,modes,weights,beta=0,
                                       steering_threshold=0), z)
    assert torch.equal(steer_activation(z,covariance,modes,weights,beta=.3,
        steering_mask=torch.zeros(2,4,dtype=torch.bool),steering_threshold=0), z)


def test_isotropic_covariance_reduces_to_euclidean_result():
    covariance, modes = fixture(variance=2.0)
    covariance.residual_variance = torch.tensor(2., dtype=torch.float64)
    identity = LowRankCovariance(covariance.mean.clone(), covariance.basis.clone(),
        torch.ones(2,dtype=torch.float64), torch.tensor(1.,dtype=torch.float64), eps=1e-12)
    z=torch.randn(2,3,dtype=torch.float64)+covariance.mean; weights=torch.ones(2,1,dtype=z.dtype)
    a=steer_activation(z,covariance,modes,weights,beta=.2,steering_threshold=0)
    b=steer_activation(z,identity,modes,weights,beta=.2,steering_threshold=0)
    torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)


def test_near_zero_centered_activation_returns_exact_input():
    covariance, modes = fixture(); z = covariance.mean.reshape(1, 1, -1).clone()
    out = steer_activation(z, covariance, modes, torch.ones(1,1,1,dtype=z.dtype),
                           beta=.5, steering_threshold=0)
    assert torch.equal(out, z)
