from __future__ import annotations

import json
from pathlib import Path

from molt_stream.measurement.comparison import (
    aggregate_comparisons,
    compare_runs,
    write_comparison,
)


def _run(
    root: Path, *, steps: int, seconds: float, joules: float, nll: float, seed: int = 8111
) -> Path:
    root.mkdir()
    spec = {
        "mode": "qlora",
        "data": {"path": "train.bin", "context_length": 256},
        "model": {"layers": 24, "width": 896},
        "stream": {
            "device": "cuda",
            "compute_dtype": "bfloat16",
            "quant_block_size": 64,
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "lora_target_modules": "all-linear",
            "double_buffer": False,
            "bundle_size": 4,
            "cuda_graphs": False,
            "pin_host_memory": True,
        },
        "batch_size": 2,
        "gradient_accumulation": 2,
        "seed": seed,
        "execution_backend": "eager",
        "activation_checkpointing": False,
        "evaluation_interval": 5,
        "thermal_target_c": 70.0,
        "thermal_abort_c": 78.0,
        "thermal_control_mode": "reactive",
        "max_steps": steps,
        "learning_rate": 2e-4,
    }
    metrics = {
        "state": "completed",
        "seconds": seconds,
        "evaluations": [{"nll": nll}],
        "cuda_peak_allocated_bytes": 3_000_000_000,
        "thermal_abort": False,
        "telemetry": {
            "gpu_board_energy_joules": joules,
            "minimum_enforced_power_limit_watts": 100.0,
            "maximum_enforced_power_limit_watts": 100.0,
            "peak_gpu_temperature_c": 70.0,
        },
    }
    (root / "spec.resolved.json").write_text(json.dumps(spec), encoding="utf-8")
    (root / "metrics.summary.json").write_text(json.dumps(metrics), encoding="utf-8")
    return root


def test_compare_runs_accepts_faster_equal_quality_early_stop(tmp_path: Path):
    baseline = _run(tmp_path / "baseline", steps=10, seconds=7.0, joules=380.0, nll=2.10)
    candidate = _run(tmp_path / "candidate", steps=5, seconds=4.5, joules=195.0, nll=2.105)

    result = compare_runs(baseline, candidate)

    assert result["survived"] is True
    assert result["controlled_mismatches"] == {}
    assert result["metrics"]["end_to_end_time_improvement_percent"] > 30


def test_compare_runs_rejects_workload_mismatch_and_writes_atomically(tmp_path: Path):
    baseline = _run(tmp_path / "baseline", steps=10, seconds=7.0, joules=380.0, nll=2.10)
    candidate = _run(tmp_path / "candidate", steps=5, seconds=4.5, joules=195.0, nll=2.105)
    spec_path = candidate / "spec.resolved.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["batch_size"] = 4
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = compare_runs(baseline, candidate)
    artifact = write_comparison(tmp_path / "comparison.json", result)

    assert result["survived"] is False
    assert "batch_size" in result["controlled_mismatches"]
    assert json.loads(artifact.read_text(encoding="utf-8"))["survived"] is False


def test_compare_runs_rejects_new_thermal_regression(tmp_path: Path):
    baseline = _run(tmp_path / "baseline", steps=10, seconds=7.0, joules=380.0, nll=2.10)
    candidate = _run(tmp_path / "candidate", steps=5, seconds=4.5, joules=195.0, nll=2.105)
    metrics_path = candidate / "metrics.summary.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["telemetry"]["peak_gpu_temperature_c"] = 73.0
    metrics["telemetry"]["thermal_throttle_observed"] = True
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    result = compare_runs(baseline, candidate)

    assert result["survived"] is False
    assert result["gates"]["thermal_peak_not_regressed"] is False
    assert result["gates"]["thermal_throttle_not_regressed"] is False


def test_aggregate_comparisons_reports_seed_level_and_interval_gates(tmp_path: Path):
    comparisons = []
    for index, seed in enumerate((8111, 12143, 16381)):
        baseline = _run(
            tmp_path / f"baseline-{seed}", steps=10, seconds=7.0 + index / 10,
            joules=380.0 + index, nll=2.10, seed=seed,
        )
        candidate = _run(
            tmp_path / f"candidate-{seed}", steps=5, seconds=4.5 + index / 10,
            joules=195.0 + index, nll=2.105, seed=seed,
        )
        comparisons.append(compare_runs(baseline, candidate))

    result = aggregate_comparisons(comparisons, bootstrap_samples=1_000)

    assert result["survived"] is True
    assert result["seeds"] == [8111, 12143, 16381]
    assert result["summary"]["time_improvement_percent"]["median"] > 30


def test_aggregate_comparisons_rejects_duplicate_seeds(tmp_path: Path):
    baseline = _run(tmp_path / "baseline", steps=10, seconds=7.0, joules=380.0, nll=2.10)
    candidate = _run(tmp_path / "candidate", steps=5, seconds=4.5, joules=195.0, nll=2.105)
    comparison = compare_runs(baseline, candidate)

    result = aggregate_comparisons([comparison, comparison], bootstrap_samples=100)

    assert result["survived"] is False
    assert result["gates"]["unique_seeds"] is False
