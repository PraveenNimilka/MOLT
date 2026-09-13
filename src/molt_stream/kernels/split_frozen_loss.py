"""Experimental two-launch vocabulary loss with explicit buffer lifetime boundaries."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from molt_stream.kernels.compiler import configure_windows_compiler_cache

configure_windows_compiler_cache()


_QUALIFIED_WEIGHT_SHAPE = (151_936, 1_536)
_QUALIFIED_TOKEN_COUNT = 256
_QUALIFIED_CHUNK_SIZE = 96


def split_frozen_head_supported(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
) -> bool:
    """Restrict automatic dispatch to the fully qualified Qwen1.5B geometry."""

    return bool(
        hidden.device.type == "cuda"
        and weight.device == hidden.device
        and targets.device == hidden.device
        and hidden.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == _QUALIFIED_WEIGHT_SHAPE
        and targets.numel() == _QUALIFIED_TOKEN_COUNT
        and chunk_size == _QUALIFIED_CHUNK_SIZE
        and hidden.shape[-1] == _QUALIFIED_WEIGHT_SHAPE[1]
    )


@triton.jit
def _statistics(logits, targets, losses, maximum, denominator, N: tl.constexpr,
                TOTAL: float, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = -float("inf")
    d = 0.0
    for off in range(0, N, BLOCK):
        mask = off + cols < N
        value = tl.load(logits + row * N + off + cols, mask, other=-float("inf")).to(tl.float32)
        new_m = tl.maximum(m, tl.max(value, 0))
        old = tl.where(d > 0, d * tl.exp(m - new_m), 0.0)
        d = old + tl.sum(tl.where(mask, tl.exp(value - new_m), 0.0), 0)
        m = new_m
    target = tl.load(targets + row)
    valid = (target >= 0) & (target < N)
    target_logit = tl.load(logits + row * N + tl.where(valid, target, 0)).to(tl.float32)
    loss = tl.where(valid, (m + tl.log(d) - target_logit) / TOTAL, 0.0)
    tl.store(losses + row, loss)
    tl.store(maximum + row, m)
    tl.store(denominator + row, d)


@triton.jit
def _overwrite_gradient(logits, targets, maximum, denominator, N: tl.constexpr,
                        TOTAL: float, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    m, d = tl.load(maximum + row), tl.load(denominator + row)
    target = tl.load(targets + row)
    value = tl.load(logits + row * N + cols, mask, other=-float("inf")).to(tl.float32)
    grad = (tl.exp(value - m) / d - tl.where(cols == target, 1.0, 0.0)) / TOTAL
    tl.store(logits + row * N + cols, grad, mask)


def split_frozen_head_forward(hidden, weight, targets, chunk_size):
    """Reclaim logits only after a separate kernel has consumed all row data.

    The statistics launch writes O(tokens) FP32 values. The subsequent launch
    owns disjoint elements and reads each logit before replacing that element.
    Same-stream ordering supplies the global read-before-write boundary. The
    algebra and rounding boundaries match the separated-buffer loss; exact
    numerical equivalence remains a tested requirement, not an assumption.
    """
    flat = hidden.flatten(0, -2).contiguous()
    labels = targets.reshape(-1).contiguous()
    count = labels.numel()
    n = weight.shape[0]
    gradient = torch.empty_like(flat)
    losses = torch.empty(count, device=hidden.device, dtype=torch.float32)
    maximum = torch.empty(min(chunk_size, count), device=hidden.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    block = 4096 if n >= 4096 else max(1024, triton.next_power_of_2(n))
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        part = flat[start:end].to(weight.dtype)
        logits = F.linear(part, weight)
        _statistics[(end-start,)](logits, labels[start:end], losses[start:end], maximum,
                                 denominator, n, float(count), block, num_warps=8 if block >= 2048 else 4)
        _overwrite_gradient[(end-start, triton.cdiv(n, 1024))](logits, labels[start:end],
                                 maximum, denominator, n, float(count), 1024, num_warps=4)
        gradient[start:end].copy_(logits @ weight)
        # Drop the previous vocabulary tile before Python evaluates the next
        # F.linear RHS. CUDA graph capture can then reuse the exact allocation
        # instead of briefly retaining two identically sized logits buffers.
        del logits
    return losses.sum(), gradient.reshape_as(hidden)
