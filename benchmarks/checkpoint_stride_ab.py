"""Preregistered paired all-layer checkpoint-stride experiment.

Runs each arm in a fresh process, alternates arm order across seeds, and measures
interpolated time and board energy to one predeclared held-out NLL target.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from molt_stream.measurement.target import interpolate_nll_crossing


SEEDS = (1337, 2027, 4099)
ORDERS = (("stride1", "stride16"), ("stride16", "stride1"), ("stride1", "stride16"))


def _temperature_c() -> float | None:
    process = subprocess.run([
        "nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode:
        return None
    try:
        return float(process.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def _wait_for_start_temperature(maximum_c: float, timeout_seconds: float) -> float:
    deadline = time.monotonic() + timeout_seconds
    while True:
        value = _temperature_c()
        if value is None:
            raise RuntimeError("A fresh nvidia-smi temperature sample is required")
        if value <= maximum_c:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU did not cool to {maximum_c} C; latest sample was {value} C")
        time.sleep(2.0)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run(spec: dict[str, Any], spec_path: Path) -> tuple[Path, dict[str, Any]]:
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    process = subprocess.run([
        sys.executable, "-m", "molt_stream.cli", "--json", "train",
        "--config", str(spec_path), "-y",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode:
        raise RuntimeError(f"Training failed for {spec_path}: {(process.stderr or process.stdout)[-4000:]}")
    response = json.loads(process.stdout)
    run = Path(response["run"])
    metrics = json.loads((run / "metrics.summary.json").read_text(encoding="utf-8"))
    return run, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-nll", type=float, default=1.58)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--start-temperature-c", type=float, default=50.0)
    parser.add_argument("--cooldown-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if not math.isfinite(args.target_nll) or args.steps < 2:
        parser.error("target NLL must be finite and steps must be at least two")
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    pairs: list[dict[str, Any]] = []
    for seed, order in zip(SEEDS, ORDERS, strict=True):
        trials: dict[str, dict[str, Any]] = {}
        for arm in order:
            start_temperature = _wait_for_start_temperature(
                args.start_temperature_c, args.cooldown_timeout_seconds,
            )
            spec = copy.deepcopy(template)
            spec.update(seed=seed, max_steps=args.steps, evaluation_interval=2,
                        qlora_validation_batches=4, qlora_train_last_layers=None,
                        qlora_checkpoint_stride=1 if arm == "stride1" else 16,
                        artifacts_dir=str(root / "runs"))
            path, metrics = _run(spec, root / f"seed{seed}-{arm}.json")
            crossing = interpolate_nll_crossing(metrics["evaluations"], args.target_nll)
            trials[arm] = {
                "run": str(path), "state": metrics.get("state"),
                "start_temperature_c": start_temperature,
                "peak_temperature_c": metrics.get("telemetry", {}).get("peak_gpu_temperature_c"),
                "trainable_parameters": metrics.get("active_trainable_parameter_count"),
                "allocated_bytes": metrics.get("cuda_peak_allocated_bytes"),
                "compute_tokens_per_second": metrics.get("update_compute_tokens_per_second"),
                "final_nll": metrics["evaluations"][-1]["nll"], "crossing": crossing,
            }
        baseline, candidate = trials["stride1"], trials["stride16"]
        reasons: list[str] = []
        if baseline["state"] != "completed" or candidate["state"] != "completed":
            reasons.append("incomplete run")
        if baseline["crossing"] is None or candidate["crossing"] is None:
            reasons.append("target NLL not reached")
        if candidate["final_nll"] > baseline["final_nll"] * 1.01:
            reasons.append("quality regression exceeds 1%")
        if candidate["peak_temperature_c"] > 72:
            reasons.append("candidate exceeded 72 C")
        if baseline["trainable_parameters"] != candidate["trainable_parameters"]:
            reasons.append("trainable parameter mismatch")
        metrics: dict[str, float] = {}
        if not reasons:
            metrics = {
                "time_improvement_percent": 100 * (1 - candidate["crossing"]["seconds"] / baseline["crossing"]["seconds"]),
                "energy_improvement_percent": 100 * (1 - candidate["crossing"]["joules"] / baseline["crossing"]["joules"]),
                "final_nll_change_percent": 100 * (candidate["final_nll"] / baseline["final_nll"] - 1),
            }
            if metrics["time_improvement_percent"] <= 0:
                reasons.append("no time-to-target improvement")
            if metrics["energy_improvement_percent"] <= 0:
                reasons.append("no energy-to-target improvement")
        pairs.append({"seed": seed, "order": list(order), "trials": trials,
                      "metrics": metrics, "passed": not reasons, "reasons": reasons})
        _atomic_json(root / "results.partial.json", pairs)
    result = {
        "schema_version": 1, "target_nll": args.target_nll,
        "start_temperature_max_c": args.start_temperature_c, "pairs": pairs,
        "passed": all(pair["passed"] for pair in pairs),
        "promotion_allowed": all(pair["passed"] for pair in pairs),
        "limitations": ["Three paired short runs are not a 30-minute endurance benchmark",
                        "CPU package temperature is unavailable through current telemetry"],
    }
    _atomic_json(root / "results.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
