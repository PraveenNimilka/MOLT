from __future__ import annotations

import math

import torch
from torch import nn

from molt_stream.core.specs import ModelSpec
from molt_stream.kernels.fused import KernelSuite


class CausalBlock(nn.Module):
    def __init__(self, spec: ModelSpec, kernels: KernelSuite):
        super().__init__()
        self.heads = spec.heads
        self.head_width = spec.width // spec.heads
        self.norm1 = kernels.rms_norm(spec.width)
        self.qkv = nn.Linear(spec.width, 3 * spec.width, bias=False)
        self.out = nn.Linear(spec.width, spec.width, bias=False)
        self.norm2 = kernels.rms_norm(spec.width)
        self.mlp = kernels.swiglu(spec.width, spec.hidden_width)
        self.kernels = kernels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = value.shape
        normalized = self.norm1(value)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        reshape = lambda item: item.view(batch, sequence, self.heads, self.head_width).transpose(1, 2)
        attended = self.kernels.attention(reshape(q), reshape(k), reshape(v))
        value = value + self.out(attended.transpose(1, 2).reshape(batch, sequence, width))
        return value + self.mlp(self.norm2(value))


class SmallCausalLM(nn.Module):
    def __init__(self, spec: ModelSpec, *, require_liger: bool = False):
        super().__init__()
        self.spec = spec
        self.kernels = KernelSuite(require_liger=require_liger)
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.width)
        self.position_embedding = nn.Embedding(spec.context_length, spec.width)
        self.blocks = nn.ModuleList(CausalBlock(spec, self.kernels) for _ in range(spec.layers))
        self.norm = self.kernels.rms_norm(spec.width)
        self.lm_head = nn.Linear(spec.width, spec.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        value = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            value = block(value)
        return self.norm(value)

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.logits_from_hidden(self.hidden_states(tokens))

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.kernels.cross_entropy(self(tokens), targets)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
