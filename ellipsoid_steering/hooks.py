from dataclasses import dataclass
from typing import Callable

import torch

from .utils import extract_hidden, replace_hidden


@dataclass
class HookHandleGroup:
    handles: list[torch.utils.hooks.RemovableHandle]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()


def register_layer_hooks(layers, target_layers: list[int],
                         callback: Callable[[int, torch.Tensor], torch.Tensor | None],
                         edit: bool = False) -> HookHandleGroup:
    """Register callbacks only on selected residual-stream layers."""
    handles = []
    for index in target_layers:
        if not 0 <= index < len(layers):
            raise IndexError(f"target layer {index} outside model layer range")

        def hook(_module, _inputs, output, layer_index=index):
            hidden = extract_hidden(output)
            replacement = callback(layer_index, hidden)
            if edit and replacement is not None:
                if replacement.shape != hidden.shape:
                    raise ValueError("hook replacement shape mismatch")
                return replace_hidden(output, replacement)
            return None

        handles.append(layers[index].register_forward_hook(hook))
    return HookHandleGroup(handles)
