from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EllipsoidSteeringConfig:
    """Configuration for offline fitting and inference-time steering."""

    target_layers: list[int]
    covariance_rank: int = 128
    hallucination_rank: int = 16
    num_hallucination_modes: int = 8
    covariance_eps: float = 1e-5
    residual_variance_floor: float = 1e-5
    routing_temperature: float = 0.1
    beta_max: float = 0.5
    steering_gain: float = 1.0
    steering_threshold: float = 0.05
    preserve_mahalanobis_radius: bool = True
    token_policy: str = "text"
    position_aware: bool = True
    layer_aware: bool = True
    spectral_gamma: float = 1e-2
    spectral_weighting: str = "soft"
    hard_routing: bool = False
    uniform_beta: float | None = None
    use_masked_image_signal: bool = True
    routing_mode: str = "teacher_forced"
    dtype: str = "float32"
    debug_invariants: bool = False

    def __post_init__(self) -> None:
        if not self.target_layers or len(set(self.target_layers)) != len(self.target_layers):
            raise ValueError("target_layers must be a non-empty unique list")
        if min(self.target_layers) < 0:
            raise ValueError("target layer indices must be non-negative")
        if self.covariance_rank < 1 or self.hallucination_rank < 1:
            raise ValueError("covariance and hallucination ranks must be positive")
        if self.num_hallucination_modes < 1:
            raise ValueError("num_hallucination_modes must be positive")
        if self.covariance_eps <= 0 or self.residual_variance_floor <= 0:
            raise ValueError("covariance floors must be positive")
        if self.routing_temperature <= 0 or self.spectral_gamma <= 0:
            raise ValueError("routing temperature and spectral gamma must be positive")
        if not 0 <= self.beta_max <= 1 or self.steering_gain < 0:
            raise ValueError("invalid beta_max or steering_gain")
        if self.token_policy not in {"all", "text", "generated", "visual", "selected_positions"}:
            raise ValueError("unsupported token_policy")
        if self.routing_mode not in {"teacher_forced", "instance", "position"}:
            raise ValueError("unsupported routing_mode")
        if self.spectral_weighting not in {"soft", "hard", "variance_ratio"}:
            raise ValueError("unsupported spectral_weighting")
        if self.uniform_beta is not None and not 0 <= self.uniform_beta <= self.beta_max:
            raise ValueError("uniform_beta must lie in [0,beta_max]")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EllipsoidSteeringConfig":
        return cls(**values)
