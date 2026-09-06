"""Isolated vocabulary projection precision probe; never a training benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    old = torch.get_float32_matmul_precision()
    torch.manual_seed(43)
    hidden = torch.randn(128, 1536, device="cuda") * 0.1
    weight = torch.randn(151936, 1536, device="cuda") * 0.1
    targets = torch.randint(151936, (128,), device="cuda")
    results: dict[str, object] = {}
    try:
        for precision in ("highest", "high"):
            torch.set_float32_matmul_precision(precision)
            times = []
            losses = []
            for _ in range(5):
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss = torch.nn.functional.cross_entropy(hidden @ weight.T, targets)
                losses.append(float(loss))
                times.append(time.perf_counter() - started)
            results[precision] = {"seconds": times, "losses": losses,
                "median_warm_seconds": statistics.median(times[1:])}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"diagnostic_only": True,
            "shape": [128, 1536, 151936], "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(), "results": results}, indent=2), encoding="utf-8")
    finally:
        torch.set_float32_matmul_precision(old)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
