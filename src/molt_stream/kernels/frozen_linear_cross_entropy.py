from __future__ import annotations

import os
from typing import Any

import torch
import torch.library
from torch.nn import functional as F

from molt_stream.kernels.compiler import configure_windows_compiler_cache

# Ensure short compiler cache paths on Windows
configure_windows_compiler_cache()

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_ce_softmax_grad_kernel_online(
        logits_ptr,
        targets_ptr,
        grad_out_ptr,
        loss_out_ptr,
        stride_lm,
        stride_ln,
        stride_gm,
        stride_gn,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        total_tokens: float,
    ) -> None:
        """Fused online log-sum-exp, loss reduction, and softmax gradient.

        Computes exact unfiltered cross-entropy and hidden gradient contributions
        in two streaming passes over vocabulary blocks:
          Pass 1: Numerically stable online row max and sum-of-exponentials.
          Pass 2: Forms softmax probabilities, subtracts 1.0 at target index,
                  and overwrites the private low-precision chunk buffer with the
                  scaled gradient (P - 1_y) / total_tokens. This avoids full
                  FP32 logits but still materializes one bounded token chunk.
        """
        row_id = tl.program_id(0)
        target = tl.load(targets_ptr + row_id)
        cols = tl.arange(0, BLOCK_N)

        valid_target = (target >= 0) & (target < N)
        target_idx = tl.where(valid_target, target, 0)

        # 1. Online max and sum exp in a single pass over vocabulary tiles
        m_prev = -float("inf")
        d_prev = 0.0
        for off in range(0, N, BLOCK_N):
            mask = (off + cols) < N
            vals = tl.load(
                logits_ptr + row_id * stride_lm + (off + cols) * stride_ln,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            m_curr = tl.max(vals, axis=0)
            m_new = tl.maximum(m_prev, m_curr)
            scaled_prev = tl.where(d_prev > 0.0, d_prev * tl.exp(m_prev - m_new), 0.0)
            curr_sum = tl.sum(tl.where(mask, tl.exp(vals - m_new), 0.0), axis=0)
            d_prev = scaled_prev + curr_sum
            m_prev = m_new

        lse = m_prev + tl.log(d_prev)

        # Exact target logit and scalar loss accumulation
        target_logit = tl.load(logits_ptr + row_id * stride_lm + target_idx * stride_ln).to(
            tl.float32
        )
        loss = tl.where(valid_target, (lse - target_logit) / total_tokens, 0.0)
        tl.atomic_add(loss_out_ptr, loss)

        # 2. Compute softmax - 1(target) and store directly in output tensor dtype
        for off in range(0, N, BLOCK_N):
            mask = (off + cols) < N
            col_indices = off + cols
            vals = tl.load(
                logits_ptr + row_id * stride_lm + col_indices * stride_ln,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            p = tl.exp(vals - m_prev) / d_prev
            is_target = (col_indices == target) & valid_target
            grad = (p - tl.where(is_target, 1.0, 0.0)) / total_tokens
            tl.store(
                grad_out_ptr + row_id * stride_gm + col_indices * stride_gn,
                grad.to(grad_out_ptr.dtype.element_ty),
                mask=mask,
            )


def _analytical_frozen_head_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact analytical CPU/fallback path without materializing full logits."""
    flat = hidden.flatten(0, -2)
    labels = targets.reshape(-1)
    token_count = labels.numel()
    gradient = torch.empty_like(flat)
    total = torch.zeros(
        (),
        device=hidden.device,
        dtype=torch.float64 if hidden.dtype == torch.float64 else torch.float32,
    )
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        part = flat[start:end]
        compute_part = part if part.dtype == weight.dtype else part.to(weight.dtype)
        logits = F.linear(compute_part, weight)
        reduction_logits = (
            logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
        )
        loss = (
            F.cross_entropy(reduction_logits, labels[start:end], reduction="sum")
            / token_count
        )
        probabilities = reduction_logits.softmax(dim=-1)
        probabilities[
            torch.arange(end - start, device=labels.device), labels[start:end]
        ] -= 1.0
        probabilities /= token_count
        local_gradient = probabilities.to(weight.dtype) @ weight
        total.add_(loss.detach())
        gradient[start:end].copy_(local_gradient.to(gradient.dtype))
    return total, gradient.reshape_as(hidden)


def _triton_frozen_head_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated chunked vocabulary loss and hidden gradient."""
    flat_hidden = hidden.flatten(0, -2).contiguous()
    flat_targets = targets.reshape(-1).contiguous()
    token_count = flat_targets.numel()
    hidden_gradient = torch.empty_like(flat_hidden)
    loss_total = torch.zeros((), device=hidden.device, dtype=torch.float32)

    N = weight.shape[0]
    if N >= 4096:
        block_n = 4096
    else:
        block_n = 1024
        while block_n < N:
            block_n *= 2

    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        c_len = end - start
        part = flat_hidden[start:end]
        compute_part = part if part.dtype == weight.dtype else part.to(weight.dtype)
        sub_targets = flat_targets[start:end]

        # Forward GEMM (FP16/BF16 Tensor Cores via cuBLAS)
        # Preserve the frozen head's native layout. Calling contiguous() here
        # copied Qwen's 151,936 x 1,536 head on every microbatch.
        logits = F.linear(compute_part, weight)
        # The projection is a private temporary. Overwrite it with dL/dlogits
        # during the Triton pass instead of allocating a second [chunk, vocab]
        # probability buffer.
        _fused_ce_softmax_grad_kernel_online[(c_len,)](
            logits,
            sub_targets,
            logits,
            loss_total,
            logits.stride(0),
            logits.stride(1),
            logits.stride(0),
            logits.stride(1),
            N,
            block_n,
            float(token_count),
            num_warps=8 if block_n >= 2048 else 4,
        )

        # Backward GEMM (FP16/BF16 Tensor Cores via cuBLAS)
        local_gradient = logits @ weight
        hidden_gradient[start:end].copy_(local_gradient)

    return loss_total, hidden_gradient.reshape_as(hidden)


def _validate_inputs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
) -> None:
    if hidden.ndim < 2 or weight.ndim != 2:
        raise ValueError("hidden must be [..., width] and weight must be [vocab, width]")
    if hidden.shape[:-1] != targets.shape:
        raise ValueError("targets must match every non-feature hidden dimension")
    if hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden width and classifier weight width must match")
    if targets.dtype != torch.long:
        raise ValueError("targets must use torch.long indices")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if targets.numel() == 0:
        raise ValueError("targets must not be empty")
    if weight.requires_grad:
        raise ValueError(
            "frozen_linear_cross_entropy requires a frozen classifier (weight.requires_grad=False)"
        )


def triton_frozen_loss_supported(hidden: torch.Tensor, weight: torch.Tensor) -> bool:
    """Report whether this exact invocation can use the Triton implementation."""
    return bool(
        _TRITON_AVAILABLE
        and hidden.device.type == "cuda"
        and hidden.dtype in {torch.bfloat16, torch.float16, torch.float32}
        and weight.dtype in {torch.bfloat16, torch.float16, torch.float32}
        and (
            weight.dtype == hidden.dtype
            or (hidden.dtype == torch.float32 and weight.dtype in {torch.bfloat16, torch.float16})
        )
    )


@torch.library.custom_op("molt::frozen_linear_cross_entropy", mutates_args=())
def frozen_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Exact cross-entropy and analytical hidden gradient for frozen linear heads.

    Tiled over token chunks to avoid allocating complete [tokens, vocabulary]
    FP32 logits. A fused Triton reduction overwrites each private low-precision
    chunk with its exact softmax gradient before the hidden-gradient GEMM.
    """
    _validate_inputs(hidden, weight, targets, chunk_size)

    use_triton = triton_frozen_loss_supported(hidden, weight)
    if use_triton:
        loss, grad = _triton_frozen_head_forward(hidden, weight, targets, chunk_size)
    else:
        loss, grad = _analytical_frozen_head_forward(hidden, weight, targets, chunk_size)

    loss._saved_grad = grad  # type: ignore[attr-defined]
    return loss


@frozen_linear_cross_entropy.register_fake
def _(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    _validate_inputs(hidden, weight, targets, chunk_size)
    out_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
    return torch.empty((), device=hidden.device, dtype=out_dtype)


def _setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    hidden, weight, targets, chunk_size = inputs
    if hasattr(output, "_saved_grad"):
        ctx.save_for_backward(output._saved_grad)
    else:
        ctx.save_for_backward(hidden, weight, targets)
    ctx.chunk_size = chunk_size


def _backward(
    ctx: Any, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, None, None]:
    tensors = ctx.saved_tensors
    if len(tensors) == 1:
        return tensors[0] * grad_output, None, None, None
    hidden, weight, targets = tensors
    _, grad = _analytical_frozen_head_forward(hidden, weight, targets, ctx.chunk_size)
    return grad * grad_output, None, None, None


frozen_linear_cross_entropy.register_autograd(_backward, setup_context=_setup_context)
