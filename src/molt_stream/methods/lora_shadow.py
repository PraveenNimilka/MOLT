"""BF16 compute shadows for FP32 LoRA masters.

The optimizer continues to own FP32 parameters and FP32 state.  Shadows are
invalidated by PyTorch parameter version counters after an optimizer update,
so activation recomputation and accumulated microbatches reuse one conversion
without ever training the shadow tensors themselves.
"""
from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F

from molt_stream.core.errors import CapabilityError


class _ShadowLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        a_master: torch.Tensor,
        b_master: torch.Tensor,
        a_shadow: torch.Tensor,
        b_shadow: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        if a_master.dtype != torch.float32 or b_master.dtype != torch.float32:
            raise CapabilityError("LoRA shadow compute requires FP32 adapter masters")
        low_rank = F.linear(x, a_shadow)
        ctx.scaling = float(scaling)
        ctx.save_for_backward(x, low_rank, a_shadow, b_shadow)
        return F.linear(low_rank, b_shadow).mul_(ctx.scaling)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor):
        x, low_rank, a_shadow, b_shadow = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        flat_low_rank = low_rank.reshape(-1, low_rank.shape[-1])
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            grad_through_b = flat_grad @ b_shadow
            grad_x = grad_through_b @ a_shadow
            grad_b = flat_grad.t() @ flat_low_rank
            grad_a = grad_through_b.t() @ flat_x
        scale = ctx.scaling
        return (
            grad_x.reshape_as(x).mul_(scale),
            grad_a.mul(scale).float(),
            grad_b.mul(scale).float(),
            None,
            None,
            None,
        )


def shadow_lora(
    x: torch.Tensor,
    a_master: torch.Tensor,
    b_master: torch.Tensor,
    a_shadow: torch.Tensor,
    b_shadow: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    return _ShadowLoRAFunction.apply(
        x, a_master, b_master, a_shadow, b_shadow, scaling
    )


def _shadow_forward(
    module: torch.nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    if args or kwargs or module.disable_adapters or module.merged:
        return module._molt_shadow_original_forward(x, *args, **kwargs)
    active = list(module.active_adapters)
    if len(active) != 1:
        return module._molt_shadow_original_forward(x, *args, **kwargs)
    adapter = active[0]
    if (
        adapter not in module.lora_A
        or adapter in module.lora_variant
        or module.use_dora.get(adapter, False)
        or not isinstance(module.lora_dropout[adapter], torch.nn.Identity)
        or module.base_layer.bias is not None
    ):
        return module._molt_shadow_original_forward(x, *args, **kwargs)
    a_master = module.lora_A[adapter].weight
    b_master = module.lora_B[adapter].weight
    version = (a_master._version, b_master._version)
    if module._molt_shadow_version != version:
        module._molt_shadow_a = a_master.detach().to(dtype=x.dtype)
        module._molt_shadow_b = b_master.detach().to(dtype=x.dtype)
        module._molt_shadow_version = version
    base = module.base_layer(x)
    return base + shadow_lora(
        x,
        a_master,
        b_master,
        module._molt_shadow_a,
        module._molt_shadow_b,
        float(module.scaling[adapter]),
    )


def enable_bf16_lora_shadows(model: torch.nn.Module) -> int:
    """Patch compatible PEFT 4-bit LoRA modules with versioned BF16 shadows."""
    try:
        from peft.tuners.lora.bnb import Linear4bit
    except ImportError as exc:  # pragma: no cover
        raise CapabilityError("PEFT bitsandbytes Linear4bit support is unavailable") from exc

    patched = 0
    for module in model.modules():
        if not isinstance(module, Linear4bit) or hasattr(module, "_molt_shadow_original_forward"):
            continue
        module._molt_shadow_original_forward = module.forward
        module._molt_shadow_version = None
        module._molt_shadow_a = None
        module._molt_shadow_b = None
        module.forward = MethodType(_shadow_forward, module)
        patched += 1
    if not patched:
        raise CapabilityError("No compatible PEFT NF4-LoRA modules were found")
    return patched


def invalidate_bf16_lora_shadows(model: torch.nn.Module) -> int:
    """Invalidate every installed shadow after an optimizer update.

    Fused optimizers may update parameter storage without changing the Python
    tensor version counter, so correctness cannot depend on ``_version`` alone.
    """
    invalidated = 0
    for module in model.modules():
        if hasattr(module, "_molt_shadow_original_forward"):
            module._molt_shadow_version = None
            invalidated += 1
    return invalidated
