"""Preregistered all-layer MOLT versus Unsloth AB/BA acceptance harness."""
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
ORDERS = (("unsloth", "molt"), ("molt", "unsloth"), ("unsloth", "molt"))


def _temperature_c() -> float:
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "nvidia-smi failed")
    return float(process.stdout.strip().splitlines()[0])


def _stable_cold_start(maximum_c: float, dwell_seconds: int, timeout_seconds: float) -> float:
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    latest = math.inf
    while stable < dwell_seconds:
        latest = _temperature_c()
        stable = stable + 1 if latest <= maximum_c else 0
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"GPU did not sustain <= {maximum_c} C for {dwell_seconds}s; latest={latest} C"
            )
        time.sleep(1.0)
    return latest


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_process(command: list[str], log_root: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (log_root.with_suffix(".stdout.txt")).write_text(process.stdout, encoding="utf-8")
    (log_root.with_suffix(".stderr.txt")).write_text(process.stderr, encoding="utf-8")
    return process


def _run_molt(template: dict[str, Any], root: Path, seed: int) -> dict[str, Any]:
    spec = copy.deepcopy(template)
    spec.update(seed=seed, artifacts_dir=str(root / "molt-runs"))
    spec_path = root / "molt-config.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    process = _run_process(
        [sys.executable, "-m", "molt_stream.cli", "--json", "train", "--config", str(spec_path), "-y"],
        root / "molt-process",
    )
    if process.returncode:
        raise RuntimeError(f"MOLT process failed: {process.stderr[-4000:]}")
    run = Path(json.loads(process.stdout)["run"])
    metrics = json.loads((run / "metrics.summary.json").read_text(encoding="utf-8"))
    metrics["artifact"] = str(run)
    metrics["engine"] = "molt"
    return metrics


def _run_unsloth(args: argparse.Namespace, root: Path, seed: int) -> dict[str, Any]:
    destination = root / "unsloth-run"
    command = [
        sys.executable,
        str(Path(__file__).with_name("unsloth_qwen_matched.py")),
        "--unsloth-site", args.unsloth_site,
        "--model", args.model,
        "--train-data", args.train_data,
        "--validation-data", args.validation_data,
        "--output", str(destination),
        "--steps", str(args.steps),
        "--evaluation-interval", str(args.evaluation_interval),
        "--validation-batches", str(args.validation_batches),
        "--seed", str(seed),
        "--thermal-target-c", str(args.unsloth_thermal_target_c),
        "--thermal-guard-c", str(args.unsloth_thermal_guard_c),
        "--thermal-abort-c", str(args.thermal_abort_c),
        "--thermal-power-target-watts", str(args.unsloth_power_target_watts),
        "--thermal-initial-pause-seconds", str(args.unsloth_initial_pause_seconds),
        "--thermal-max-pause-seconds", str(args.unsloth_max_pause_seconds),
    ]
    process = _run_process(command, root / "unsloth-process")
    metrics_path = destination / "metrics.summary.json"
    if not metrics_path.exists():
        raise RuntimeError(
            f"Unsloth produced no metrics (exit {process.returncode}): {process.stderr[-4000:]}"
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["artifact"] = str(destination)
    metrics["engine"] = "unsloth"
    return metrics


def _trial(metrics: dict[str, Any], target_nll: float, start_c: float) -> dict[str, Any]:
    evaluations = metrics.get("evaluations", [])
    telemetry = metrics.get("telemetry", {})
    return {
        "artifact": metrics["artifact"],
        "state": metrics.get("state"),
        "start_temperature_c": start_c,
        "peak_temperature_c": telemetry.get("peak_gpu_temperature_c"),
        "trainable_parameters": metrics.get("active_trainable_parameter_count", metrics.get("trainable_parameters")),
        "steps": metrics.get("step", metrics.get("steps")),
        "compute_tokens_per_second": metrics.get("committed_update_compute_tokens_per_second", metrics.get("compute_tokens_per_second")),
        "end_to_end_tokens_per_second": metrics.get("tokens_per_second", metrics.get("end_to_end_tokens_per_second")),
        "allocated_bytes": metrics.get("cuda_peak_allocated_bytes"),
        "nvml_used_bytes": telemetry.get("peak_gpu_used_bytes"),
        "initial_nll": evaluations[0]["nll"] if evaluations else None,
        "final_nll": evaluations[-1]["nll"] if evaluations else None,
        "crossing": interpolate_nll_crossing(evaluations, target_nll) if evaluations else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molt-template", required=True)
    parser.add_argument("--unsloth-site", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--evaluation-interval", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--target-nll", type=float, default=1.54)
    parser.add_argument("--start-temperature-c", type=float, default=48.0)
    parser.add_argument("--cold-dwell-seconds", type=int, default=5)
    parser.add_argument("--cooldown-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--thermal-abort-c", type=float, default=72.0)
    parser.add_argument("--unsloth-thermal-target-c", type=float, default=58.0)
    parser.add_argument("--unsloth-thermal-guard-c", type=float, default=63.0)
    parser.add_argument("--unsloth-power-target-watts", type=float, default=30.0)
    parser.add_argument("--unsloth-initial-pause-seconds", type=float, default=0.6)
    parser.add_argument("--unsloth-max-pause-seconds", type=float, default=1.5)
    args = parser.parse_args()
    if min(args.steps, args.evaluation_interval, args.validation_batches, args.cold_dwell_seconds) < 1:
        parser.error("counts must be positive")
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    template = json.loads(Path(args.molt_template).read_text(encoding="utf-8"))
    template.update(max_steps=args.steps, evaluation_interval=args.evaluation_interval,
                    qlora_validation_batches=args.validation_batches)
    pairs: list[dict[str, Any]] = []
    for seed, order in zip(SEEDS, ORDERS, strict=True):
        trials: dict[str, dict[str, Any]] = {}
        for engine in order:
            start_c = _stable_cold_start(
                args.start_temperature_c,
                args.cold_dwell_seconds,
                args.cooldown_timeout_seconds,
            )
            trial_root = root / f"seed{seed}-{engine}"
            trial_root.mkdir()
            metrics = (
                _run_molt(template, trial_root, seed)
                if engine == "molt"
                else _run_unsloth(args, trial_root, seed)
            )
            trials[engine] = _trial(metrics, args.target_nll, start_c)
        molt, unsloth = trials["molt"], trials["unsloth"]
        reasons: list[str] = []
        if molt["state"] != "completed" or unsloth["state"] != "completed":
            reasons.append("both engines must complete")
        if molt["crossing"] is None or unsloth["crossing"] is None:
            reasons.append("both engines must reach target NLL")
        if molt["trainable_parameters"] != unsloth["trainable_parameters"]:
            reasons.append("trainable parameter mismatch")
        if (
            molt["initial_nll"] is None
            or unsloth["initial_nll"] is None
            or abs(molt["initial_nll"] / unsloth["initial_nll"] - 1.0) > 0.01
        ):
            reasons.append("initial validation objective mismatch exceeds 1%")
        if max(molt["peak_temperature_c"] or math.inf, unsloth["peak_temperature_c"] or math.inf) > args.thermal_abort_c:
            reasons.append("thermal boundary exceeded")
        comparison: dict[str, float] = {}
        if not reasons:
            comparison = {
                "molt_time_improvement_percent": 100.0 * (1.0 - molt["crossing"]["seconds"] / unsloth["crossing"]["seconds"]),
                "molt_energy_improvement_percent": 100.0 * (1.0 - molt["crossing"]["joules"] / unsloth["crossing"]["joules"]),
            }
            if min(comparison.values()) <= 0:
                reasons.append("MOLT must improve both time and energy to target")
        pairs.append({"seed": seed, "order": list(order), "trials": trials,
                      "comparison": comparison, "passed": not reasons, "reasons": reasons})
        _atomic_json(root / "results.partial.json", pairs)
        if reasons:
            break
    result = {
        "schema_version": 1,
        "experiment": "qwen2.5-1.5b-all-layer-molt-vs-unsloth",
        "target_nll": args.target_nll,
        "registered_seeds": list(SEEDS),
        "registered_orders": [list(order) for order in ORDERS],
        "pairs": pairs,
        "passed": len(pairs) == len(SEEDS) and all(pair["passed"] for pair in pairs),
        "limitations": [
            "A short screen does not satisfy the separate 30-minute endurance gate",
            "The harness stops after the first failed required pair",
        ],
    }
    _atomic_json(root / "results.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
