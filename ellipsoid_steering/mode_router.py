import torch

from .hallucination_modes import HallucinationMode


def routing_scores(signal_white: torch.Tensor,
                   modes: list[HallucinationMode]) -> torch.Tensor:
    """Return squared subspace alignment with shape `[...,K]`."""
    if not modes:
        raise ValueError("at least one hallucination mode is required")
    scores = []
    for mode in modes:
        basis = mode.basis.to(signal_white)
        scores.append((signal_white @ basis).square().sum(dim=-1))
    return torch.stack(scores, dim=-1)


def route_modes(signal_white: torch.Tensor, modes: list[HallucinationMode],
                temperature: float = 0.1, position_aware: bool = True,
                hard: bool = False) -> torch.Tensor:
    """Route `[B,T,D]` signals to `[B,T,K]` or instance-level `[B,1,K]`."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    signal = signal_white
    if not position_aware and signal.ndim >= 3:
        signal = signal.mean(dim=-2, keepdim=True)
    scores = routing_scores(signal, modes)
    if hard:
        index = scores.argmax(dim=-1)
        return torch.nn.functional.one_hot(index, len(modes)).to(scores.dtype)
    return torch.softmax(scores/temperature, dim=-1)


def routing_entropy(weights: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(weights*weights.clamp_min(eps).log()).sum(dim=-1)
