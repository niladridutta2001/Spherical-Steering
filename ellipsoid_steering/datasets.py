from dataclasses import dataclass
from typing import Any, Protocol

import torch


@dataclass
class PairedCaptionExample:
    example_id: str
    image: Any
    grounded_caption: str
    hallucinated_caption: str
    grounded_window: tuple[int, int] | None = None
    hallucinated_window: tuple[int, int] | None = None


class LVLMAdapter(Protocol):
    """Model-specific boundary; geometry code never assumes an LVLM family."""

    def prepare_inputs(self, image: Any, text: str) -> dict[str, torch.Tensor]: ...
    def token_masks(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


def selected_token_mask(policy: str, masks: dict[str, torch.Tensor],
                        selected_positions: torch.Tensor | None = None) -> torch.Tensor:
    if policy == "all":
        first = next(iter(masks.values()))
        return torch.ones_like(first, dtype=torch.bool)
    if policy == "selected_positions":
        if selected_positions is None:
            raise ValueError("selected_positions policy requires an explicit mask")
        return selected_positions.bool()
    if policy not in masks:
        raise ValueError(f"adapter did not provide a {policy!r} token mask")
    return masks[policy].bool()
