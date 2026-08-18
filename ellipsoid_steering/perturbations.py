from abc import ABC, abstractmethod
from typing import Any


class VisualPerturbation(ABC):
    @abstractmethod
    def __call__(self, image: Any) -> Any:
        raise NotImplementedError


class GaussianBlurPerturbation(VisualPerturbation):
    def __init__(self, radius: float = 8.0) -> None:
        self.radius = radius

    def __call__(self, image):
        from PIL import ImageFilter
        return image.filter(ImageFilter.GaussianBlur(self.radius))


class PatchMaskPerturbation(VisualPerturbation):
    """Mask a fractional image rectangle; useful as a model-agnostic baseline."""

    def __init__(self, box_fraction=(0.25, 0.25, 0.75, 0.75), fill=0) -> None:
        self.box_fraction, self.fill = box_fraction, fill

    def __call__(self, image):
        result = image.copy()
        w, h = result.size
        x0, y0, x1, y1 = self.box_fraction
        box = (int(x0*w), int(y0*h), int(x1*w), int(y1*h))
        from PIL import ImageDraw
        ImageDraw.Draw(result).rectangle(box, fill=self.fill)
        return result
