from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from molt_stream.measurement.comparison import write_comparison
from molt_stream.methods.counterfactual_rate import counterfactual_optimizer_step


def _fixture(elements_per_group: int) -> tuple[
    torch.nn.Parameter, torch.nn.Parameter, torch.optim.AdamW
]:
    device = torch.device("cuda")
    first = torch.nn.Parameter(torch.randn(elements_per_group, device=device) * 0.01)
    second = torch.nn.Parameter(torch.randn(elements_per_group, device=device) * 0.01)
    optimizer = torch.optim.AdamW(
        [{"params": [first], "lr": 1e-4}, {"params": [second], "lr": 1e-4}],
        weight_decay=0.01,
    )
    first.grad = torch.randn_like(first)
    second.grad = torch.randn_like(second)
    optimizer.step()
    first.grad = torch.randn_like(first)
    second.grad = torch.randn_like(second)
    return first, second, optimizer


def _score(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first.square().mean() + second.square().mean()).detach())


def benchmark(*, parameter_count: int, ratios: list[float], trials: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("counterfactual-rate CUDA benchmark requires CUDA")
    if parameter_count <= 0 or parameter_count % 2:
        raise ValueError("parameter_count must be positive and even")
    if trials <= 0 or not ratios or any(ratio <= 0 for ratio in ratios):
        raise ValueError("trials and candidate ratios must be positive")
    elements_per_group = parameter_count // 2
    first, second, baseline_optimizer = _fixture(elements_per_group)
    probe_first, probe_second, probe_optimizer = _fixture(elements_per_group)
    candidates = tuple((1e-4, 1e-4 * ratio) for ratio in ratios)
    torch.cuda.synchronize()

    # Explicit warm-up is excluded from measured trials.
    baseline_optimizer.step()
    _score(first, second)
    counterfactual_optimizer_step(
        probe_optimizer, candidates, lambda: _score(probe_first, probe_second)
    )
    torch.cuda.synchronize()
    baseline_seconds: list[float] = []
    counterfactual_seconds: list[float] = []
    selected_indices: list[int] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(trials):
        started = time.perf_counter()
        baseline_optimizer.step()
        _score(first, second)
        torch.cuda.synchronize()
        baseline_seconds.append(time.perf_counter() - started)

        started = time.perf_counter()
        result = counterfactual_optimizer_step(
            probe_optimizer, candidates, lambda: _score(probe_first, probe_second)
        )
        torch.cuda.synchronize()
        counterfactual_seconds.append(time.perf_counter() - started)
        selected_indices.append(result.selected_index)
    median_baseline = statistics.median(baseline_seconds)
    median_counterfactual = statistics.median(counterfactual_seconds)
    return {
        "schema_version": 1,
        "benchmark": "counterfactual-rate-optimizer-proxy",
        "parameter_count": parameter_count,
        "candidate_ratios": ratios,
        "candidate_count": len(ratios),
        "trials": trials,
        "baseline_seconds": baseline_seconds,
        "counterfactual_seconds": counterfactual_seconds,
        "median_baseline_seconds": median_baseline,
        "median_counterfactual_seconds": median_counterfactual,
        "optimizer_proxy_overhead_ratio": median_counterfactual / median_baseline,
        "selected_indices": selected_indices,
        "peak_allocated_bytes_for_both_fixtures": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "scope_limitation": (
            "Optimizer/state-copy proxy only; excludes model forward, backward, "
            "validation scoring, convergence, energy, and thermal behavior."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=int, default=4_400_000)
    parser.add_argument("--ratios", default="4,8,14")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--output", default="artifacts/benchmarks/counterfactual-rate-proxy.json"
    )
    args = parser.parse_args()
    ratios = [float(value) for value in args.ratios.split(",")]
    value = benchmark(parameter_count=args.parameters, ratios=ratios, trials=args.trials)
    value["artifact_path"] = str(write_comparison(Path(args.output), value))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
