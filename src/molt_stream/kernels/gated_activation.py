from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F

from molt_stream.core.errors import CapabilityError
from molt_stream.kernels.compiler import configure_windows_compiler_cache
from molt_stream.kernels.execution_plan import resolve_decoder_architecture


configure_windows_compiler_cache()

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _silu_mul_forward(gate, up, output, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        gate_value = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
        up_value = tl.load(up + offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = tl.sigmoid(gate_value)
        tl.store(output + offsets, gate_value * sigmoid * up_value, mask=mask)

    @triton.jit
    def _silu_mul_backward(
        grad_output, gate, up, grad_gate, grad_up, n_elements, BLOCK: tl.constexpr
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        gradient = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_value = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
        up_value = tl.load(up + offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = tl.sigmoid(gate_value)
        silu = gate_value * sigmoid
        silu_gradient = sigmoid * (1.0 + gate_value * (1.0 - sigmoid))
        tl.store(grad_gate + offsets, gradient * up_value * silu_gradient, mask=mask)
        tl.store(grad_up + offsets, gradient * silu, mask=mask)


class _FusedSiLUMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
            raise ValueError("gate and up tensors must have identical shape, dtype, and device")
        if gate.device.type != "cuda" or not _TRITON_AVAILABLE:
            ctx.triton = False
            ctx.save_for_backward(gate, up)
            return F.silu(gate) * up
        contiguous_gate = gate.contiguous()
        contiguous_up = up.contiguous()
        output = torch.empty_like(contiguous_gate)
        block = 256
        _silu_mul_forward[(triton.cdiv(output.numel(), block),)](
            contiguous_gate, contiguous_up, output, output.numel(), block
        )
        ctx.triton = True
        ctx.original_shape = gate.shape
        ctx.save_for_backward(contiguous_gate, contiguous_up)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        if not ctx.triton:
            sigmoid = torch.sigmoid(gate)
            silu = gate * sigmoid
            return (
                grad_output * up * sigmoid * (1.0 + gate * (1.0 - sigmoid)),
                grad_output * silu,
            )
        gradient = grad_output.contiguous()
        grad_gate = torch.empty_like(gate)
        grad_up = torch.empty_like(up)
        block = 256
        _silu_mul_backward[(triton.cdiv(gate.numel(), block),)](
            gradient, gate, up, grad_gate, grad_up, gate.numel(), block
        )
        return grad_gate.reshape(ctx.original_shape), grad_up.reshape(ctx.original_shape)


def fused_silu_multiply(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return _FusedSiLUMultiply.apply(gate, up)


def _fused_mlp_forward(module: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    return module.down_proj(
        fused_silu_multiply(module.gate_proj(hidden), module.up_proj(hidden))
    )


def enable_fused_silu_mlp(model: torch.nn.Module) -> int:
    """Patch compatible SiLU-gated decoder MLPs without changing state keys."""

    architecture = resolve_decoder_architecture(model)
    config = getattr(model, "config", None)
    activation = str(
        getattr(config, "hidden_activation", getattr(config, "hidden_act", ""))
    ).lower()
    if activation not in {"silu", "swish"}:
        raise CapabilityError(
            f"fused SiLU MLP requires silu/swish, but {architecture.family} uses "
            f"{activation or '<unknown>'}"
        )
    patched = 0
    for module in model.modules():
        if hasattr(module, "_molt_original_mlp_forward"):
            continue
        if not all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")):
            continue
        module._molt_original_mlp_forward = module.forward
        module.forward = MethodType(_fused_mlp_forward, module)
        patched += 1
    if patched == 0:
        raise CapabilityError(f"No compatible {architecture.family} gated MLP modules found")
    return patched
