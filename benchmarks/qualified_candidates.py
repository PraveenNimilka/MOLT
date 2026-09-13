"""Numerical and randomized timing screen for bounded-memory experiments."""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("loss",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    def save():
        args.output.write_text(json.dumps({"rows": rows, "promoted": False,
             "scope": "synthetic component screen, not full training"}, indent=2), encoding="utf-8")
    def screen(label, functions):
        expected = functions["reference"]()
        if not isinstance(expected, tuple):
            expected = (expected,)
        for name, function in functions.items():
            actual = function()
            actual = actual if isinstance(actual, tuple) else (actual,)
            equal = all(torch.equal(a, b) for a, b in zip(actual, expected))
            errors = [float((a.float()-b.float()).abs().max()) for a, b in zip(actual, expected)]
            passed = all(torch.isfinite(a).all() and torch.equal(a, b)
                         for a, b in zip(actual, expected))
            row = {"case": label, "name": name, "passes": bool(passed), "bitwise": equal, "errors": errors}
            rows.append(row)
            del actual
            if not passed:
                print(f"REJECT {label} {name}: {errors}", flush=True)
        del expected
        gc.collect()
        active = [row for row in rows if row["case"] == label and row["passes"]]
        for row in active:
            for _ in range(3):
                functions[row["name"]]()
            row["times_ms"] = []
            row["peak_increment_bytes"] = []
        order = [row for row in active for _ in range(12)]
        random.Random(982).shuffle(order)
        for row in order:
            torch.cuda.synchronize()
            start_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = functions[row["name"]]()
            end.record()
            end.synchronize()
            row["times_ms"].append(start.elapsed_time(end))
            row["peak_increment_bytes"].append(torch.cuda.max_memory_allocated()-start_bytes)
            del result
        for row in active:
            row["median_ms"] = statistics.median(row["times_ms"])
            row["median_increment_bytes"] = statistics.median(row["peak_increment_bytes"])
            print(f"{label} {row['name']}: exact={row['bitwise']} {row['median_ms']:.3f} ms; "
                  f"{row['median_increment_bytes']/2**20:.2f} MiB", flush=True)
        save()
    with torch.no_grad():
        for seed in (1337, 2027, 4099):
            torch.manual_seed(seed)
            if args.kind == "loss":
                from molt_stream.kernels.frozen_linear_cross_entropy import (
                    _triton_frozen_head_forward,
                )
                from molt_stream.kernels.split_frozen_loss import (
                    split_frozen_head_forward,
                )
                for tokens, width, vocab in ((256, 1536, 151936), (17, 64, 317)):
                    hidden = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
                    weight = torch.randn(vocab, width, device="cuda", dtype=torch.bfloat16) * 0.05
                    targets = torch.randint(vocab, (tokens,), device="cuda")
                    screen(f"seed{seed}-m{tokens}-k{width}-n{vocab}", {
                        "reference": lambda hidden=hidden, weight=weight, targets=targets: _triton_frozen_head_forward(hidden, weight, targets, 96),
                        "split": lambda hidden=hidden, weight=weight, targets=targets: split_frozen_head_forward(hidden, weight, targets, 96)})
                    del hidden, weight, targets
            gc.collect()
            torch.cuda.empty_cache()
    save()


if __name__ == "__main__":
    main()
