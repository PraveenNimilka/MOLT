"""Randomized exact frozen-vocabulary-head component benchmark.

This is a kernel screen, not an end-to-end training result.  It compares
identical fixed tensors and reports CUDA-event time for forward plus hidden
gradient.  Cut Cross-Entropy is optional and must be supplied explicitly.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from molt_stream.kernels.partitioned_loss import exact_partitioned_linear_cross_entropy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cce-site", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.tokens, args.width, args.vocab, args.chunk_size, args.samples) <= 0:
        raise ValueError("dimensions, chunk size, and sample count must be positive")

    torch.manual_seed(1337)
    device = torch.device("cuda")
    hidden = torch.randn(args.tokens, args.width, device=device, dtype=torch.bfloat16)
    weight = torch.randn(args.vocab, args.width, device=device, dtype=torch.bfloat16)
    targets = torch.randint(args.vocab, (args.tokens,), device=device)

    class LegacyFrozenHead(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h: torch.Tensor) -> torch.Tensor:  # type: ignore[no-untyped-def]
            gradient = torch.empty_like(h)
            total = torch.zeros((), device=device, dtype=torch.float32)
            for start in range(0, args.tokens, args.chunk_size):
                end = min(start + args.chunk_size, args.tokens)
                with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    part = h[start:end].detach().requires_grad_(True)
                    logits = F.linear(part, weight)
                    loss = F.cross_entropy(
                        logits.float(), targets[start:end], reduction="sum"
                    ) / args.tokens
                    local_gradient, = torch.autograd.grad(loss, part)
                total.add_(loss.detach())
                gradient[start:end].copy_(local_gradient)
            ctx.save_for_backward(gradient)
            return total

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):  # type: ignore[no-untyped-def]
            gradient, = ctx.saved_tensors
            return gradient * grad_output

    def autograd_partitioned(h: torch.Tensor) -> torch.Tensor:
        return LegacyFrozenHead.apply(h)

    backends: dict[str, object] = {
        "molt-autograd-partitioned": autograd_partitioned,
        "molt-analytical-partitioned": lambda h: exact_partitioned_linear_cross_entropy(
            h, weight, targets, args.chunk_size, precompute_frozen_gradient=True, backend="analytical"
        ),
        "molt-fused-triton": lambda h: exact_partitioned_linear_cross_entropy(
            h, weight, targets, args.chunk_size, precompute_frozen_gradient=True, backend="triton"
        ),
    }
    if args.cce_site is not None:
        sys.path.insert(0, str(args.cce_site.resolve()))
        from cut_cross_entropy import linear_cross_entropy

        backends["cut-cross-entropy"] = lambda h: linear_cross_entropy(
            h, weight, targets, shift=False, reduction="mean", filter_eps=None
        )

    def measure(function: object) -> tuple[float, float, torch.Tensor, int]:
        local_hidden = hidden.detach().requires_grad_(True)
        torch.cuda.reset_peak_memory_stats()
        starting_bytes = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = function(local_hidden)  # type: ignore[operator]
        loss.backward()
        end.record()
        end.synchronize()
        assert local_hidden.grad is not None
        scratch_bytes = max(0, torch.cuda.max_memory_allocated() - starting_bytes)
        return (
            float(start.elapsed_time(end)),
            float(loss.detach()),
            local_hidden.grad.detach(),
            scratch_bytes,
        )

    for function in backends.values():
        for _ in range(args.warmup):
            measure(function)
    timings: dict[str, list[float]] = {name: [] for name in backends}
    losses: dict[str, list[float]] = {name: [] for name in backends}
    scratch: dict[str, list[int]] = {name: [] for name in backends}
    gradients: dict[str, torch.Tensor] = {}
    order = [name for _ in range(args.samples) for name in backends]
    random.Random(1337).shuffle(order)
    for name in order:
        elapsed, loss, gradient, scratch_bytes = measure(backends[name])
        timings[name].append(elapsed)
        losses[name].append(loss)
        scratch[name].append(scratch_bytes)
        gradients[name] = gradient

    reference_name = next(reversed(backends))
    reference_gradient = gradients[reference_name].float()
    result = {
        "scope": "synthetic exact frozen-head component; not end-to-end training",
        "shape": {"tokens": args.tokens, "width": args.width, "vocab": args.vocab},
        "dtype": "bfloat16 with FP32 loss reduction",
        "chunk_size": args.chunk_size,
        "samples_per_backend": args.samples,
        "backends": {
            name: {
                "median_ms": statistics.median(values),
                "minimum_ms": min(values),
                "maximum_ms": max(values),
                "mean_loss": statistics.fmean(losses[name]),
                "median_peak_incremental_allocated_bytes": statistics.median(scratch[name]),
                "max_abs_gradient_difference_vs_reference": float(
                    (gradients[name].float() - reference_gradient).abs().max()
                ),
                "mean_abs_gradient_difference_vs_reference": float(
                    (gradients[name].float() - reference_gradient).abs().mean()
                ),
            }
            for name, values in timings.items()
        },
    }
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
