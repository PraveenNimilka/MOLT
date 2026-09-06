from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.nn import functional as F


def _autocast_context(
    device_type: str, enabled: bool, dtype: torch.dtype
) -> Any:
    if device_type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(device_type=device_type, enabled=enabled, dtype=dtype)


class _ExactPartitionedLinearCrossEntropy(torch.autograd.Function):
    """Exact linear+CE with token-chunk recomputation in backward.

    The primitive never retains the complete [tokens, vocabulary] logits
    tensor. It is intentionally implemented with public PyTorch operations so
    the correctness path works on native Windows without a custom compiler.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        flat_hidden = hidden.flatten(0, -2)
        flat_targets = targets.reshape(-1)
        token_count = flat_targets.numel()
        accumulation_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        total = torch.zeros((), device=hidden.device, dtype=accumulation_dtype)
        device_type = hidden.device.type
        autocast_enabled = torch.is_autocast_enabled(device_type)
        autocast_dtype = (
            torch.get_autocast_dtype(device_type) if device_type in {"cuda", "cpu"} else hidden.dtype
        )
        with _autocast_context(device_type, autocast_enabled, autocast_dtype):
            for start in range(0, token_count, chunk_size):
                end = min(start + chunk_size, token_count)
                logits = F.linear(flat_hidden[start:end], weight)
                total = total + F.cross_entropy(
                    logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits,
                    flat_targets[start:end],
                    reduction="sum",
                )
        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk_size = chunk_size
        ctx.autocast_enabled = autocast_enabled
        ctx.autocast_dtype = autocast_dtype
        return total / token_count

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        hidden, weight, targets = ctx.saved_tensors
        flat_hidden = hidden.flatten(0, -2)
        flat_targets = targets.reshape(-1)
        token_count = flat_targets.numel()
        hidden_gradient = torch.zeros_like(flat_hidden) if ctx.needs_input_grad[0] else None
        weight_gradient = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        device_type = hidden.device.type
        for start in range(0, token_count, ctx.chunk_size):
            end = min(start + ctx.chunk_size, token_count)
            with torch.enable_grad():
                hidden_chunk = flat_hidden[start:end].detach().requires_grad_(ctx.needs_input_grad[0])
                local_weight = weight.detach().requires_grad_(ctx.needs_input_grad[1])
                with _autocast_context(
                    device_type, ctx.autocast_enabled, ctx.autocast_dtype
                ):
                    logits = F.linear(hidden_chunk, local_weight)
                    loss = F.cross_entropy(
                        logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits,
                        flat_targets[start:end],
                        reduction="sum",
                    )
                inputs = []
                if ctx.needs_input_grad[0]:
                    inputs.append(hidden_chunk)
                if ctx.needs_input_grad[1]:
                    inputs.append(local_weight)
                gradients = torch.autograd.grad(
                    loss,
                    inputs,
                    grad_outputs=grad_output / token_count,
                    create_graph=False,
                )
            gradient_index = 0
            if hidden_gradient is not None:
                hidden_gradient[start:end].copy_(gradients[gradient_index])
                gradient_index += 1
            if weight_gradient is not None:
                weight_gradient.add_(gradients[gradient_index])
        return (
            hidden_gradient.reshape_as(hidden) if hidden_gradient is not None else None,
            weight_gradient,
            None,
            None,
        )


class _FrozenHeadCrossEntropy(torch.autograd.Function):
    """Compute each head projection once, retaining only its hidden gradient.

    First-order scalar-loss training only. The classifier is frozen. Chunked
    gradient precomputation is established prior art, not a novelty claim.
    """

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, weight: torch.Tensor,
                targets: torch.Tensor, chunk_size: int) -> torch.Tensor:
        flat = hidden.flatten(0, -2)
        labels = targets.reshape(-1)
        enabled = torch.is_autocast_enabled(hidden.device.type)
        dtype = torch.get_autocast_dtype(hidden.device.type)
        gradient = torch.empty_like(flat)
        total = torch.zeros((), device=hidden.device,
                            dtype=torch.float64 if hidden.dtype == torch.float64 else torch.float32)
        for start in range(0, labels.numel(), chunk_size):
            end = min(start + chunk_size, labels.numel())
            with _autocast_context(hidden.device.type, enabled, dtype):
                part = flat[start:end].detach()
                logits = F.linear(part, weight)
            # Cross-entropy is reduced in FP32, exactly as the public QLoRA
            # path.  Form dL/dlogits directly so the frozen classifier does
            # not pay for a nested autograd graph per chunk.
            reduction_logits = (
                logits.float()
                if logits.dtype in {torch.float16, torch.bfloat16}
                else logits
            )
            loss = F.cross_entropy(
                reduction_logits, labels[start:end], reduction="sum"
            ) / labels.numel()
            probabilities = reduction_logits.softmax(dim=-1)
            probabilities[
                torch.arange(end - start, device=labels.device), labels[start:end]
            ] -= 1.0
            probabilities /= labels.numel()
            with _autocast_context(hidden.device.type, enabled, dtype):
                local_gradient = probabilities.to(weight.dtype) @ weight
            total.add_(loss.detach())
            gradient[start:end].copy_(local_gradient.to(gradient.dtype))
        ctx.save_for_backward(gradient.reshape_as(hidden))
        return total

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        gradient, = ctx.saved_tensors
        return gradient * grad_output, None, None, None


def exact_partitioned_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int,
    *,
    precompute_frozen_gradient: bool = False,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute exact mean CE while bounding materialized logits to one chunk."""
    if backend not in {"auto", "triton", "analytical"}:
        raise ValueError(f"Unknown backend: {backend}")
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
    if precompute_frozen_gradient:
        if weight.requires_grad:
            raise ValueError("Gradient precomputation requires a frozen classifier")
        if backend in {"auto", "triton"}:
            from molt_stream.kernels.frozen_linear_cross_entropy import (
                frozen_linear_cross_entropy,
                triton_frozen_loss_supported,
            )
            if backend == "triton" and not triton_frozen_loss_supported(hidden, weight):
                raise RuntimeError("Triton frozen-loss backend is unavailable for these tensors")
            return frozen_linear_cross_entropy(hidden, weight, targets, chunk_size)
        elif backend == "analytical":
            if torch.is_grad_enabled() and hidden.requires_grad:
                return _FrozenHeadCrossEntropy.apply(hidden, weight, targets, chunk_size)
    return _ExactPartitionedLinearCrossEntropy.apply(hidden, weight, targets, chunk_size)
