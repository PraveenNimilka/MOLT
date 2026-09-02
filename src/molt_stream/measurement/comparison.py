from __future__ import annotations

import json
import os
import random
import statistics
from pathlib import Path
from typing import Any


_CONTROLLED_PATHS = (
    "mode",
    "data",
    "model",
    "batch_size",
    "gradient_accumulation",
    "seed",
    "execution_backend",
    "activation_checkpointing",
    "evaluation_interval",
    "thermal_target_c",
    "thermal_abort_c",
    "thermal_control_mode",
    "stream.device",
    "stream.compute_dtype",
    "stream.quant_block_size",
    "stream.lora_rank",
    "stream.lora_alpha",
    "stream.lora_target_modules",
    "stream.double_buffer",
    "stream.bundle_size",
    "stream.cuda_graphs",
    "stream.pin_host_memory",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _nested(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _final_nll(metrics: dict[str, Any]) -> float:
    evaluations = metrics.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("run metrics contain no evaluations")
    return float(evaluations[-1]["nll"])


def _energy_joules(metrics: dict[str, Any]) -> float:
    telemetry = metrics.get("telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("gpu_board_energy_joules") is None:
        raise ValueError("run metrics contain no measured GPU board energy")
    return float(telemetry["gpu_board_energy_joules"])


def compare_runs(
    baseline_run: str | Path,
    candidate_run: str | Path,
    *,
    quality_tolerance_percent: float = 1.0,
    minimum_improvement_percent: float = 1.0,
    thermal_peak_tolerance_c: float = 1.0,
) -> dict[str, Any]:
    """Compare two complete runs while making workload mismatches explicit."""
    if min(quality_tolerance_percent, minimum_improvement_percent, thermal_peak_tolerance_c) < 0:
        raise ValueError("comparison tolerances must be non-negative")
    baseline_root = Path(baseline_run)
    candidate_root = Path(candidate_run)
    baseline_spec = _read_json(baseline_root / "spec.resolved.json")
    candidate_spec = _read_json(candidate_root / "spec.resolved.json")
    baseline = _read_json(baseline_root / "metrics.summary.json")
    candidate = _read_json(candidate_root / "metrics.summary.json")
    mismatches = {
        path: {"baseline": _nested(baseline_spec, path), "candidate": _nested(candidate_spec, path)}
        for path in _CONTROLLED_PATHS
        if _nested(baseline_spec, path) != _nested(candidate_spec, path)
    }
    baseline_nll = _final_nll(baseline)
    candidate_nll = _final_nll(candidate)
    baseline_seconds = float(baseline["seconds"])
    candidate_seconds = float(candidate["seconds"])
    baseline_energy = _energy_joules(baseline)
    candidate_energy = _energy_joules(candidate)
    baseline_memory = int(baseline["cuda_peak_allocated_bytes"])
    candidate_memory = int(candidate["cuda_peak_allocated_bytes"])
    quality_delta = 100.0 * (candidate_nll / baseline_nll - 1.0)
    time_improvement = 100.0 * (1.0 - candidate_seconds / baseline_seconds)
    energy_improvement = 100.0 * (1.0 - candidate_energy / baseline_energy)
    memory_delta = 100.0 * (candidate_memory / baseline_memory - 1.0)
    baseline_telemetry = baseline.get("telemetry", {})
    candidate_telemetry = candidate.get("telemetry", {})
    power_state_matches = all(
        baseline_telemetry.get(key) == candidate_telemetry.get(key)
        for key in (
            "minimum_enforced_power_limit_watts",
            "maximum_enforced_power_limit_watts",
        )
    )
    baseline_peak_temperature = baseline_telemetry.get("peak_gpu_temperature_c")
    candidate_peak_temperature = candidate_telemetry.get("peak_gpu_temperature_c")
    thermal_peak_not_regressed = (
        baseline_peak_temperature is not None
        and candidate_peak_temperature is not None
        and float(candidate_peak_temperature)
        <= float(baseline_peak_temperature) + thermal_peak_tolerance_c
    )
    throttle_not_regressed = (
        not bool(candidate_telemetry.get("thermal_throttle_observed"))
        or bool(baseline_telemetry.get("thermal_throttle_observed"))
    )
    gates = {
        "controlled_workload_matches": not mismatches,
        "both_completed": baseline.get("state") == candidate.get("state") == "completed",
        "quality_within_tolerance": quality_delta <= quality_tolerance_percent,
        "time_improved": time_improvement >= minimum_improvement_percent,
        "energy_improved": energy_improvement >= minimum_improvement_percent,
        "memory_not_regressed": memory_delta <= quality_tolerance_percent,
        "candidate_did_not_thermal_abort": not bool(candidate.get("thermal_abort")),
        "thermal_peak_not_regressed": thermal_peak_not_regressed,
        "thermal_throttle_not_regressed": throttle_not_regressed,
        "enforced_power_state_matches": power_state_matches,
    }
    return {
        "schema_version": 1,
        "baseline_run": str(baseline_root),
        "candidate_run": str(candidate_root),
        "seed": baseline_spec.get("seed"),
        "controlled_mismatches": mismatches,
        "treatment": {
            "baseline_max_steps": baseline_spec.get("max_steps"),
            "candidate_max_steps": candidate_spec.get("max_steps"),
            "baseline_learning_rate": baseline_spec.get("learning_rate"),
            "candidate_learning_rate": candidate_spec.get("learning_rate"),
            "baseline_lora_plus_lr_ratio": _nested(baseline_spec, "stream.lora_plus_lr_ratio"),
            "candidate_lora_plus_lr_ratio": _nested(candidate_spec, "stream.lora_plus_lr_ratio"),
        },
        "metrics": {
            "baseline_validation_nll": baseline_nll,
            "candidate_validation_nll": candidate_nll,
            "validation_nll_change_percent": quality_delta,
            "end_to_end_time_improvement_percent": time_improvement,
            "board_energy_improvement_percent": energy_improvement,
            "peak_allocated_memory_change_percent": memory_delta,
            "baseline_peak_temperature_c": baseline_peak_temperature,
            "candidate_peak_temperature_c": candidate_peak_temperature,
        },
        "thresholds": {
            "quality_tolerance_percent": quality_tolerance_percent,
            "minimum_improvement_percent": minimum_improvement_percent,
            "thermal_peak_tolerance_c": thermal_peak_tolerance_c,
        },
        "gates": gates,
        "survived": all(gates.values()),
    }


def _bootstrap_median_interval(
    values: list[float],
    *,
    samples: int,
    generator: random.Random,
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    medians = sorted(
        statistics.median(generator.choices(values, k=len(values)))
        for _ in range(samples)
    )
    lower = medians[int(0.025 * (samples - 1))]
    upper = medians[int(0.975 * (samples - 1))]
    return [float(lower), float(upper)]


def aggregate_comparisons(
    comparisons: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260902,
) -> dict[str, Any]:
    """Aggregate paired run comparisons without hiding seed-level failures."""
    if len(comparisons) < 2:
        raise ValueError("paired aggregation requires at least two comparisons")
    seeds = [comparison.get("seed") for comparison in comparisons]
    unique_seeds = len(set(seeds)) == len(seeds) and None not in seeds
    treatments = [comparison.get("treatment") for comparison in comparisons]
    treatment_consistent = all(value == treatments[0] for value in treatments[1:])
    keys = {
        "end_to_end_time_improvement_percent": "time_improvement_percent",
        "board_energy_improvement_percent": "board_energy_improvement_percent",
        "validation_nll_change_percent": "validation_nll_change_percent",
        "peak_allocated_memory_change_percent": "memory_change_percent",
    }
    generator = random.Random(bootstrap_seed)
    summary: dict[str, dict[str, float | list[float]]] = {}
    for metric_key, output_key in keys.items():
        values = [float(comparison["metrics"][metric_key]) for comparison in comparisons]
        summary[output_key] = {
            "median": float(statistics.median(values)),
            "minimum": min(values),
            "maximum": max(values),
            "bootstrap_95_percent_interval": _bootstrap_median_interval(
                values, samples=bootstrap_samples, generator=generator
            ),
        }
    gates = {
        "at_least_two_pairs": len(comparisons) >= 2,
        "unique_seeds": unique_seeds,
        "treatment_consistent": treatment_consistent,
        "every_pair_survived": all(bool(value.get("survived")) for value in comparisons),
        "time_interval_excludes_regression": (
            summary["time_improvement_percent"]["bootstrap_95_percent_interval"][0] > 0
        ),
        "energy_interval_excludes_regression": (
            summary["board_energy_improvement_percent"]["bootstrap_95_percent_interval"][0] > 0
        ),
    }
    return {
        "schema_version": 1,
        "comparison_count": len(comparisons),
        "seeds": seeds,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "treatment": treatments[0] if treatment_consistent else None,
        "summary": summary,
        "pairs": comparisons,
        "gates": gates,
        "survived": all(gates.values()),
    }


def write_comparison(path: str | Path, value: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination
