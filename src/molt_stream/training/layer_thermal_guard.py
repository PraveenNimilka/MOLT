"""Exact-math thermal checkpoints inside long QLoRA updates."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from molt_stream.core.errors import CapabilityError


class TrainingThermalStop(RuntimeError):
    """An update was interrupted at a safe decoder-layer boundary."""


@contextmanager
def guarded_decoder_training(
    model: torch.nn.Module,
    *,
    boundary_after_layers: int,
    checkpoint: Callable[[], bool],
) -> Iterator[None]:
    """Check thermals midway through forward and backward without changing tensors.

    ``boundary_after_layers`` is one-based: 14 splits a 28-layer decoder in half.
    Hooks are temporary and are removed even if the checkpoint stops the update.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(getattr(base, "config", None), "model_type", None) != "qwen2":
        raise CapabilityError("Intra-step layer thermal guarding currently requires Qwen2")
    layers = base.model.layers
    if not 0 < boundary_after_layers < len(layers):
        raise CapabilityError(
            f"Thermal boundary must be between 1 and {len(layers) - 1} layers"
        )

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def check(module: torch.nn.Module, _unused: object) -> None:
        parameter = next(module.parameters(), None)
        if parameter is not None and parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)
        if not checkpoint():
            raise TrainingThermalStop("Thermal stop inside a decoder update")

    try:
        handles.append(layers[boundary_after_layers].register_forward_pre_hook(check))
        handles.append(
            layers[boundary_after_layers - 1].register_full_backward_pre_hook(check)
        )
        yield
    finally:
        for handle in handles:
            handle.remove()
