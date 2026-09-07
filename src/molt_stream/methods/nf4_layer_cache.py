from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Iterator

import torch
from torch.nn import functional as F

from molt_stream.core.errors import CapabilityError


def _cached_linear_forward(
    weight: torch.Tensor, bias: torch.Tensor | None
):
    def forward(_module: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, weight, bias)

    return forward


@contextmanager
def dense_nf4_layer_cache(layer: torch.nn.Module) -> Iterator[int]:
    """Temporarily decode only one layer's frozen NF4 base projections.

    The cache is scoped to a single decoder layer and is always removed in a
    ``finally`` block. LoRA modules and their FP32 master parameters are not
    replaced or merged.
    """

    try:
        import bitsandbytes.functional as bnb_functional
        from peft.tuners.lora.bnb import Linear4bit
    except ImportError as exc:  # pragma: no cover - optional QLoRA stack
        raise CapabilityError("dense NF4 layer cache requires PEFT and bitsandbytes") from exc

    patched: list[tuple[torch.nn.Module, object]] = []
    dense_weights: list[torch.Tensor] = []
    try:
        for module in layer.modules():
            if not isinstance(module, Linear4bit):
                continue
            base = module.base_layer
            weight = base.weight
            if weight.requires_grad or getattr(weight, "quant_state", None) is None:
                raise CapabilityError("layer cache requires frozen NF4 base weights")
            if base.bias is not None:
                raise CapabilityError(
                    "layer cache has not qualified biased NF4 projections"
                )
            dense = bnb_functional.dequantize_4bit(
                weight.data, weight.quant_state
            ).to(dtype=torch.bfloat16)
            dense_weights.append(dense)
            original = base.forward
            patched.append((base, original))
            base.forward = MethodType(
                _cached_linear_forward(dense, None), base
            )
        if not patched:
            raise CapabilityError("decoder layer contains no PEFT NF4 projections")
        yield len(patched)
    finally:
        for base, original in reversed(patched):
            base.forward = original
        dense_weights.clear()
