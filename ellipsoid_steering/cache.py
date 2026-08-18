from pathlib import Path
from typing import Any

import torch

from .covariance import LowRankCovariance
from .hallucination_modes import HallucinationMode


ARTIFACT_VERSION = 1


def save_geometry(path: str | Path, covariances: dict[int, LowRankCovariance],
                  modes: dict[int, list[HallucinationMode]] | None,
                  metadata: dict[str, Any]) -> None:
    payload = {
        "version": ARTIFACT_VERSION, "metadata": metadata,
        "covariances": {k: v.state_dict() for k, v in covariances.items()},
        "modes": ({k: [mode.state_dict() for mode in values]
                   for k, values in modes.items()} if modes is not None else None)}
    torch.save(payload, Path(path))


def load_geometry(path: str | Path, expected_model: str | None = None,
                  expected_layers: list[int] | None = None,
                  expected_hidden_size: int | None = None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("version") != ARTIFACT_VERSION:
        raise ValueError("unsupported geometry artifact version")
    metadata = payload["metadata"]
    if expected_model is not None and metadata.get("model_name") != expected_model:
        raise ValueError("artifact model mismatch")
    if expected_layers is not None and metadata.get("target_layers") != expected_layers:
        raise ValueError("artifact target-layer mismatch")
    if expected_hidden_size is not None and metadata.get("hidden_size") != expected_hidden_size:
        raise ValueError("artifact hidden-size mismatch")
    covariances = {int(k): LowRankCovariance.from_state_dict(v)
                   for k, v in payload["covariances"].items()}
    mode_states = payload.get("modes")
    modes = ({int(k): [HallucinationMode.from_state_dict(v) for v in values]
              for k, values in mode_states.items()} if mode_states is not None else None)
    return covariances, modes, metadata
