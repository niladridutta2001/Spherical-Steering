import torch

from ellipsoid_steering.hallucination_modes import HallucinationMode, apply_mode_operator
from ellipsoid_steering.mode_router import routing_scores, route_modes


def modes():
    return [
        HallucinationMode(torch.tensor([1.,0.,0.]), torch.tensor([[1.],[0.],[0.]]),
                          torch.tensor([2.]), 10),
        HallucinationMode(torch.tensor([0.,1.,0.]), torch.tensor([[0.],[1.],[0.]]),
                          torch.tensor([3.]), 10)]


def test_orthogonal_signal_and_dominant_routing():
    signal = torch.tensor([[[0.,4.,0.]]])
    scores = routing_scores(signal, modes())
    assert scores[0,0,0] == 0
    weights = route_modes(signal, modes(), temperature=.1)
    assert weights.argmax(-1).item() == 1 and weights[0,0,1] > .99


def test_spectral_operator_is_matrix_free_soft_projection():
    q = torch.tensor([[2.,3.,4.]])
    output = apply_mode_operator(q, modes()[0], gamma=2.)
    torch.testing.assert_close(output, torch.tensor([[1.,0.,0.]]))
