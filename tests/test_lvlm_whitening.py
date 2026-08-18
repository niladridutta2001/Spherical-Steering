import torch

from ellipsoid_steering.whitening import (apply_cov_inv_sqrt, apply_cov_sqrt,
    whiten, unwhiten, whiten_delta)
from tests.test_lvlm_covariance import make_covariance


def test_whitening_inversion_float64():
    torch.manual_seed(3); covariance = make_covariance()
    x = torch.randn(4, 5, covariance.hidden_size, dtype=torch.float64)
    reconstructed = apply_cov_sqrt(apply_cov_inv_sqrt(x, covariance), covariance)
    error = (reconstructed-x).norm()/x.norm()
    assert error < 1e-10
    torch.testing.assert_close(unwhiten(whiten(x, covariance), covariance), x,
                               rtol=1e-10, atol=1e-10)


def test_displacement_whitening_does_not_subtract_mean():
    covariance = make_covariance(); delta = torch.randn(3, covariance.hidden_size, dtype=torch.float64)
    expected = apply_cov_inv_sqrt(delta, covariance)
    torch.testing.assert_close(whiten_delta(delta, covariance), expected)
    assert not torch.allclose(whiten_delta(delta, covariance), whiten(delta, covariance))
