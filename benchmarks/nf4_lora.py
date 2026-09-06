"""Randomized component benchmark for MOLT's scheduled NF4-LoRA backward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import time

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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.tokens, args.input_width, args.output_width, args.rank, args.samples) <= 0:
        parser.error("all numeric arguments must be positive")
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)

    import bitsandbytes as bnb

    torch.manual_seed(8447)
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

    for function in (reference, candidate, reference, candidate):
        _measure(function, upstream, (x, a, b))

    order = ["reference", "candidate"] * args.samples
    random.Random(8447).shuffle(order)
    times: dict[str, list[float]] = {"reference": [], "candidate": []}
    peaks: dict[str, list[int]] = {"reference": [], "candidate": []}
    for name in order:
        elapsed, peak = _measure(
            reference if name == "reference" else candidate, upstream, (x, a, b)
        )
        times[name].append(elapsed)
        peaks[name].append(peak)

    result = {
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
