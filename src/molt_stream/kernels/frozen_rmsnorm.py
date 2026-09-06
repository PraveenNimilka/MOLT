from __future__ import annotations

import torch
from torch import nn

from molt_stream.core.errors import CapabilityError


class _FrozenRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        if weight.requires_grad:
            raise ValueError("frozen RMSNorm requires a frozen weight")
        values = hidden.float()
        inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
        ctx.save_for_backward(hidden, weight)
        ctx.eps = eps
        return (values * inverse_rms * weight.float()).to(hidden.dtype)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight = ctx.saved_tensors
        values = hidden.float()
        inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + ctx.eps)
        normalized = values * inverse_rms
        weighted_gradient = grad_output.float() * weight.float()
        correction = (weighted_gradient * normalized).mean(dim=-1, keepdim=True)
        grad_hidden = inverse_rms * (weighted_gradient - normalized * correction)
        return grad_hidden.to(hidden.dtype), None, None


class FrozenRMSNormBF16(nn.Module):
    """FP32 RMS reduction with a BF16 residual-stream contract."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        weight = getattr(original, "weight", None)
        eps = getattr(original, "variance_epsilon", None)
        if not isinstance(weight, nn.Parameter) or weight.requires_grad or eps is None:
            raise CapabilityError("RMSNorm replacement requires a frozen weight and epsilon")
        self.weight = weight
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _FrozenRMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon
        )


def enable_qwen2_frozen_rmsnorm_bf16(model: nn.Module) -> int:
    """Replace frozen Qwen2 RMSNorms without changing state-dict key paths."""
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen2":
        raise CapabilityError("mixed-precision frozen RMSNorm currently requires Qwen2")
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if type(child).__name__ == "Qwen2RMSNorm":
                replacements.append((parent, name, child))
    if not replacements:
        raise CapabilityError("Qwen2 RMSNorm modules were not found")
    for parent, name, child in replacements:
        setattr(parent, name, FrozenRMSNormBF16(child))
    return len(replacements)
