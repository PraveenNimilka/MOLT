"""Bound validation work between temperature checks without changing tensors."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.kernels.execution_plan import resolve_decoder_architecture


class EvaluationThermalStop(RuntimeError):
    """Validation was interrupted before producing a complete metric."""


@contextmanager
def guarded_decoder_evaluation(
    model: torch.nn.Module, gate: Callable[[], bool],
) -> Iterator[None]:
    """Install temporary decoder checks; always remove hooks afterward.

    Hooks are used only in evaluation. GPU synchronization makes each boundary
    observable; it does not interrupt a kernel already in flight.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    architecture = resolve_decoder_architecture(base)
    decoder = getattr(base, "model", None)
    layers = getattr(decoder, "layers", None)
    norm = getattr(decoder, "norm", None)
    if layers is None or norm is None:
        raise CapabilityError(
            f"{architecture.family} decoder does not expose layer-guard boundaries"
        )
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
        handles.append(norm.register_forward_pre_hook(check))
        yield
    finally:
        for handle in handles:
            handle.remove()
