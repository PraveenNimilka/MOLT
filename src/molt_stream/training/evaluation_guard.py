"""Bound validation work between temperature checks without changing tensors."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from molt_stream.core.errors import CapabilityError


class EvaluationThermalStop(RuntimeError):
    """Validation was interrupted before producing a complete metric."""


@contextmanager
def guarded_decoder_evaluation(
    model: torch.nn.Module, gate: Callable[[], bool],
) -> Iterator[None]:
    """Install temporary Qwen2 layer checks; always remove hooks afterward.

    Hooks are used only in evaluation. GPU synchronization makes each boundary
    observable; it does not interrupt a kernel already in flight.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(getattr(base, "config", None), "model_type", None) != "qwen2":
        raise CapabilityError("Layer-guarded evaluation currently requires Qwen2")
    layers = base.model.layers
    handles = []

    def check(module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        parameter = next(module.parameters(), None)
        if parameter is not None and parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)
        if not gate():
            raise EvaluationThermalStop("Thermal stop during decoder evaluation")

    try:
        for layer in layers:
            handles.append(layer.register_forward_pre_hook(check))
        # Check again before the potentially expensive vocabulary projection.
        handles.append(base.model.norm.register_forward_pre_hook(check))
        yield
    finally:
        for handle in handles:
            handle.remove()
