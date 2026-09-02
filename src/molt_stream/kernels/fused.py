from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from molt_stream.core.errors import CapabilityError


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(value.dtype)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden_width: int):
        super().__init__()
        self.gate_up = nn.Linear(width, hidden_width * 2, bias=False)
        self.down = nn.Linear(hidden_width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(value).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


@dataclass(frozen=True)
class KernelCapabilities:
    torch_sdpa: bool
    triton: bool
    liger: bool
    selected_backend: str


class KernelSuite:
    """Explicit dispatch: Liger is used only when installed and requested."""

    def __init__(self, *, require_liger: bool = False):
        triton = importlib.util.find_spec("triton") is not None
        liger = importlib.util.find_spec("liger_kernel") is not None
        if require_liger and not liger:
            raise CapabilityError(
                "Liger was required but is not installed; MOLT will not label PyTorch fallbacks as fused"
            )
        self.capabilities = KernelCapabilities(True, triton, liger, "liger" if liger else "torch")

    def rms_norm(self, width: int, eps: float = 1e-6) -> nn.Module:
        if self.capabilities.liger:
            from liger_kernel.transformers import LigerRMSNorm

            return LigerRMSNorm(width, eps=eps)
        return RMSNorm(width, eps)

    def swiglu(self, width: int, hidden_width: int) -> nn.Module:
        if self.capabilities.liger:
            from liger_kernel.transformers import LigerSwiGLUMLP

            return LigerSwiGLUMLP(width, hidden_width)
        return SwiGLU(width, hidden_width)

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)

    @staticmethod
    def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
