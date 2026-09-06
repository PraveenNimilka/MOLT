"""Explicit, experimental Qwen2 checkpoint placement; no optimizer changes."""
from __future__ import annotations

import torch

from molt_stream.core.errors import CapabilityError


def configure_checkpoint_stride(model: torch.nn.Module, stride: int) -> int:
    """Keep recomputation on every nth decoder, retaining other activations.

    Requires checkpointing to have been enabled by Transformers already. This
    is ordinary selective activation checkpointing, not a novel algorithm.
    """
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen2":
        raise CapabilityError("Selective checkpoint placement currently requires Qwen2")
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None or not len(layers) or any(
        not getattr(layer, "gradient_checkpointing", False)
        or not callable(getattr(layer, "_gradient_checkpointing_func", None))
        for layer in layers
    ):
        raise CapabilityError("Qwen2 layer checkpoint controls are unavailable or disabled")
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = index % stride == 0
    return sum(layer.gradient_checkpointing for layer in layers)
