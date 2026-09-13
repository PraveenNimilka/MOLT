"""Randomized component benchmark for MOLT's scheduled NF4-LoRA backward."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from molt_stream.methods.nf4_lora import scheduled_nf4_lora


def _measure(call, upstream: torch.Tensor, tensors: tuple[torch.Tensor, ...]) -> tuple[float, int]:
    for tensor in tensors:
        tensor.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    output = call()
    output.backward(upstream)
    ended.record()
    ended.synchronize()
    return float(started.elapsed_time(ended)), int(torch.cuda.max_memory_allocated())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=1536)
    parser.add_argument("--output-width", type=int, default=1536)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8447)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.tokens, args.input_width, args.output_width, args.rank, args.samples) <= 0:
        parser.error("all numeric arguments must be positive")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)

    import bitsandbytes as bnb

    torch.manual_seed(args.seed)
    dense = torch.randn(
        args.output_width, args.input_width, device="cuda", dtype=torch.bfloat16
    )
    packed, quant_state = bnb.functional.quantize_4bit(
        dense, blocksize=64, compress_statistics=True, quant_type="nf4"
    )
    del dense
    x = torch.randn(
        args.tokens, args.input_width, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    a = torch.randn(args.rank, args.input_width, device="cuda", dtype=torch.float32,
                    requires_grad=True)
    b = torch.randn(args.output_width, args.rank, device="cuda", dtype=torch.float32,
                    requires_grad=True)
    upstream = torch.randn(
        args.tokens, args.output_width, device="cuda", dtype=torch.bfloat16
    )
    scale = 2.0

    def reference() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base = bnb.matmul_4bit(x, packed, quant_state=quant_state)
            return base + F.linear(F.linear(x, a), b) * scale

    def candidate() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return scheduled_nf4_lora(x, packed, a, b, quant_state, scale)

    # Qualify the actual benchmark geometry before reporting any performance.
    # These tolerances match the existing component tests; they do not qualify
    # a complete optimizer trajectory or imply bitwise gradient equality.
    expected = reference()
    expected.backward(upstream)
    expected_grads = [tensor.grad.detach().clone() for tensor in (x, a, b)]
    for tensor in (x, a, b):
        tensor.grad = None
    actual = candidate()
    actual.backward(upstream)
    checks = {"output_equal": torch.equal(actual, expected)}
    errors = {}
    for name, tensor, expected_grad in zip(("input", "adapter_a", "adapter_b"), (x, a, b), expected_grads):
        checks[name] = torch.allclose(tensor.grad, expected_grad, rtol=4e-3, atol=4e-3)
        errors[name] = float((tensor.grad.float() - expected_grad.float()).abs().max())
    correctness = {"passed": all(checks.values()), "checks": checks,
                   "maximum_absolute_errors": errors, "rtol": 4e-3, "atol": 4e-3}
    if not correctness["passed"]:
        from molt_stream.methods.nf4_backward import (
            packed_nf4_backward_input,
            packed_nf4_lora_backward_input,
        )

        with torch.no_grad():
            decoded = bnb.functional.dequantize_4bit(packed, quant_state)
            base_reference = upstream @ decoded
            base_candidate = packed_nf4_backward_input(upstream, packed, quant_state)
            through_b = upstream @ b.to(torch.bfloat16)
            lora_reference = (through_b @ a.to(torch.bfloat16)) * scale
            lora_candidate = packed_nf4_lora_backward_input(
                torch.zeros_like(upstream), packed, quant_state, through_b, a, scale
            )
            correctness["separate_packed_kernel_input_terms"] = {
                name: {"matches": torch.allclose(left, right, rtol=4e-3, atol=4e-3),
                       "maximum_absolute_error": float((left.float() - right.float()).abs().max())}
                for name, left, right in (
                    ("frozen_base", base_candidate, base_reference),
                    ("lora", lora_candidate, lora_reference),
                )
            }
            oracle = upstream.double() @ decoded.double()
            correctness["separate_packed_kernel_fp64_reference_errors"] = {
                name: {"maximum_absolute_error": float((value.double() - oracle).abs().max()),
                       "relative_l2_error": float(torch.linalg.vector_norm(value.double() - oracle)
                                                  / torch.linalg.vector_norm(oracle))}
                for name, value in (("baseline", base_reference), ("candidate", base_candidate))
            }
        result = {"status": "rejected_correctness", "seed": args.seed,
                  "shape": [args.tokens, args.input_width, args.output_width, args.rank],
                  "correctness": correctness}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 1
    del actual, expected, expected_grads, expected_grad

    for function in (reference, candidate, reference, candidate):
        _measure(function, upstream, (x, a, b))

    order = ["reference", "candidate"] * args.samples
    random.Random(args.seed).shuffle(order)
    times: dict[str, list[float]] = {"reference": [], "candidate": []}
    peaks: dict[str, list[int]] = {"reference": [], "candidate": []}
    for name in order:
        elapsed, peak = _measure(
            reference if name == "reference" else candidate, upstream, (x, a, b)
        )
        times[name].append(elapsed)
        peaks[name].append(peak)

    result = {
        "candidate_backend": "scheduled backward with dense dequantize-plus-GEMM fallback",
        "seed": args.seed,
        "correctness": correctness,
        "sample_order": order,
        "raw_times_ms": times,
        "scope": "synthetic Qwen-shaped NF4-LoRA forward and backward component",
        "shape": {
            "tokens": args.tokens,
            "input_width": args.input_width,
            "output_width": args.output_width,
            "rank": args.rank,
        },
        "samples_per_backend": args.samples,
        "backends": {
            name: {
                "median_ms": statistics.median(times[name]),
                "minimum_ms": min(times[name]),
                "maximum_ms": max(times[name]),
                "median_peak_allocated_bytes": int(statistics.median(peaks[name])),
            }
            for name in times
        },
        "created_unix_seconds": time.time(),
        "limitations": [
            "Component benchmark; not end-to-end training",
            "One projection shape; Qwen decoder contains several geometries",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
