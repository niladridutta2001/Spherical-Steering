import torch


def relative_radius_error(before: torch.Tensor, after: torch.Tensor,
                          eps: float = 1e-8) -> torch.Tensor:
    return (after-before).abs()/before.clamp_min(eps)


def energy_reduction(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    return before-after


def relative_intervention(input_: torch.Tensor, output: torch.Tensor,
                          eps: float = 1e-8) -> torch.Tensor:
    return torch.linalg.vector_norm(output-input_, dim=-1)/torch.linalg.vector_norm(
        input_, dim=-1).clamp_min(eps)


def categorical_kl(logits_before: torch.Tensor, logits_after: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(logits_before, dim=-1)
    log_q = torch.log_softmax(logits_after, dim=-1)
    return (log_p.exp()*(log_p-log_q)).sum(dim=-1)
