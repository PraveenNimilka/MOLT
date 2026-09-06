"""Run one matched, thermally bounded MOLT/Unsloth iso-clock experiment.

This command changes the selected GPU's graphics-clock range only after explicit
confirmation. Each arm restores automatic clocks in a ``finally`` block. It is
an experiment runner, not a source of precomputed benchmark claims.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from qwen_unsloth_ab import _run_molt, _run_unsloth, _stable_cold_start

from molt_stream.measurement.gpu_profile import (
    GPUClockProfile,
    is_windows_administrator,
    temporary_graphics_clock,
    verify_measured_clock_profile,
)

GIB = 1024 ** 3


def _default_data(name: str) -> str | None:
    root = os.environ.get("MOLT_DATA_ROOT")
    return None if not root else str(Path(root) / "qwen2.5-1.5b" / name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/molt-qwen2.5-1.5b-endurance.json")
    parser.add_argument("--model", default=os.environ.get("MOLT_QWEN_1P5B_MODEL"))
    parser.add_argument("--train-data", default=_default_data("train.bin"))
    parser.add_argument("--validation-data", default=_default_data("validation.bin"))
    parser.add_argument("--unsloth-site", default=os.environ.get("MOLT_UNSLOTH_SITE"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--order", choices=("molt-first", "unsloth-first"), default="molt-first")
    parser.add_argument("--clock-min-mhz", type=int, default=1500)
    parser.add_argument("--clock-max-mhz", type=int, default=1650)
    parser.add_argument("--start-temperature-c", type=float, default=48.0)
    parser.add_argument("--cold-dwell-seconds", type=int, default=5)
    parser.add_argument("--cooldown-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--thermal-abort-c", type=float, default=72.0)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=5.5)
    parser.add_argument("--quality-tolerance-percent", type=float, default=1.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true", help="authorize temporary clock changes")
    return parser


def _path(value: str | None, label: str, errors: list[str]) -> Path | None:
    if not value:
        errors.append(f"{label} is missing")
        return None
    result = Path(value).expanduser().resolve()
    if not result.exists():
        errors.append(f"{label} does not exist: {result}")
    return result


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """Return measured capabilities and blocking errors without mutating hardware."""
    errors: list[str] = []
    config = _path(args.config, "MOLT config", errors)
    model = _path(args.model, "model", errors)
    train = _path(args.train_data, "training data", errors)
    validation = _path(args.validation_data, "validation data", errors)
    unsloth_site = _path(args.unsloth_site, "isolated Unsloth site", errors)
    if args.steps < 1:
        errors.append("steps must be positive")
    if args.clock_min_mhz <= 0 or args.clock_max_mhz < args.clock_min_mhz:
        errors.append("clock range must satisfy 0 < minimum <= maximum")
    if args.minimum_free_vram_gib <= 0:
        errors.append("minimum free VRAM must be positive")
    if args.quality_tolerance_percent < 0:
        errors.append("quality tolerance must be non-negative")
    cuda = torch.cuda.is_available()
    if not cuda:
        errors.append("CUDA is unavailable")
    free_bytes = total_bytes = 0
    gpu_name = None
    if cuda:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gpu_name = torch.cuda.get_device_name(0)
        if free_bytes < args.minimum_free_vram_gib * GIB:
            errors.append(
                f"free VRAM is {free_bytes / GIB:.2f} GiB; "
                f"at least {args.minimum_free_vram_gib:.2f} GiB is required"
            )
    elevated = is_windows_administrator()
    if not elevated and not args.preflight_only:
        errors.append("run from an Administrator PowerShell to apply and restore GPU clocks")
    if config is not None and config.is_file():
        try:
            spec = json.loads(config.read_text(encoding="utf-8"))
            stream = spec.get("stream", {})
            data = spec.get("data", {})
            if spec.get("mode") != "qlora":
                errors.append("MOLT config must use qlora mode")
            if data.get("context_length") != 512 or spec.get("batch_size") != 1 or spec.get("gradient_accumulation") != 4:
                errors.append("matched protocol requires context 512, batch 1, accumulation 4")
            if stream.get("lora_rank") != 8 or stream.get("lora_target_modules") != "all-linear":
                errors.append("matched protocol requires rank-8 all-linear LoRA")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, AttributeError) as exc:
            errors.append(f"MOLT config is invalid: {type(exc).__name__}: {exc}")
    return {
        "cuda": cuda,
        "gpu": gpu_name,
        "free_vram_gib": free_bytes / GIB if cuda else None,
        "total_vram_gib": total_bytes / GIB if cuda else None,
        "administrator": elevated,
        "config": str(config) if config else None,
        "model": str(model) if model else None,
        "train_data": str(train) if train else None,
        "validation_data": str(validation) if validation else None,
        "unsloth_site": str(unsloth_site) if unsloth_site else None,
        "requested_clock_mhz": [args.clock_min_mhz, args.clock_max_mhz],
        "telemetry_interval_ms": 100,
    }, errors


def _value(metrics: dict[str, Any], *names: str) -> float | int | None:
    for name in names:
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return value
    return None


def _normalize(metrics: dict[str, Any], profile: GPUClockProfile) -> dict[str, Any]:
    telemetry = metrics.get("telemetry") or {}
    energy = _value(telemetry, "gpu_board_energy_joules")
    tokens = _value(metrics, "tokens", "session_tokens") or 0
    evaluations = metrics.get("evaluations", [])
    result = {
        "engine": metrics["engine"],
        "state": metrics.get("state"),
        "steps": _value(metrics, "step", "steps"),
        "tokens": tokens,
        "end_to_end_tokens_per_second": _value(
            metrics, "tokens_per_second", "end_to_end_tokens_per_second"
        ),
        "compute_tokens_per_second": _value(
            metrics, "committed_update_compute_tokens_per_second", "compute_tokens_per_second"
        ),
        "seconds": _value(metrics, "seconds"),
        "board_energy_joules": energy,
        "joules_per_token": energy / tokens if energy is not None and tokens else None,
        "thermal_pause_seconds": _value(metrics, "thermal_pause_seconds"),
        "peak_temperature_c": _value(telemetry, "peak_gpu_temperature_c"),
        "peak_allocated_gib": (
            float(_value(metrics, "cuda_peak_allocated_bytes")) / GIB
            if _value(metrics, "cuda_peak_allocated_bytes") is not None else None
        ),
        "trainable_parameters": _value(
            metrics, "active_trainable_parameter_count", "trainable_parameters"
        ),
        "initial_nll": evaluations[0].get("nll") if evaluations else None,
        "final_nll": evaluations[-1].get("nll") if evaluations else None,
        "evaluations": evaluations,
        "clock_verification": verify_measured_clock_profile(telemetry, profile),
        "artifact": metrics.get("artifact"),
    }
    return result


def _fmt(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):,.{digits}f}"


def _table(results: dict[str, dict[str, Any]]) -> str:
    columns = ("Metric", "MOLT", "Unsloth")
    rows = [
        ("State", results["molt"]["state"], results["unsloth"]["state"]),
        ("End-to-end tok/s", _fmt(results["molt"]["end_to_end_tokens_per_second"]), _fmt(results["unsloth"]["end_to_end_tokens_per_second"])),
        ("Compute tok/s", _fmt(results["molt"]["compute_tokens_per_second"]), _fmt(results["unsloth"]["compute_tokens_per_second"])),
        ("Wall seconds", _fmt(results["molt"]["seconds"]), _fmt(results["unsloth"]["seconds"])),
        ("Board joules", _fmt(results["molt"]["board_energy_joules"]), _fmt(results["unsloth"]["board_energy_joules"])),
        ("J/token", _fmt(results["molt"]["joules_per_token"], 5), _fmt(results["unsloth"]["joules_per_token"], 5)),
        ("Thermal pause s", _fmt(results["molt"]["thermal_pause_seconds"]), _fmt(results["unsloth"]["thermal_pause_seconds"])),
        ("Peak temperature C", _fmt(results["molt"]["peak_temperature_c"], 1), _fmt(results["unsloth"]["peak_temperature_c"], 1)),
        ("Peak allocated GiB", _fmt(results["molt"]["peak_allocated_gib"], 3), _fmt(results["unsloth"]["peak_allocated_gib"], 3)),
        ("Final validation NLL", _fmt(results["molt"]["final_nll"], 5), _fmt(results["unsloth"]["final_nll"], 5)),
    ]
    widths = [max(len(str(row[i])) for row in [columns, *rows]) for i in range(3)]
    render = lambda row: " | ".join(str(row[i]).ljust(widths[i]) for i in range(3))
    return "\n".join((render(columns), "-+-".join("-" * width for width in widths), *(render(row) for row in rows)))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    measured, errors = preflight(args)
    print("MOLT iso-clock preflight")
    print(json.dumps(measured, indent=2))
    if errors:
        print("\nBlocked:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    if args.preflight_only:
        print("\nPreflight passed; no hardware setting or training state was changed.")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("Refusing clock mutation without interactive confirmation or --yes.", file=sys.stderr)
            return 2
        answer = input(
            "This will temporarily set GPU 0 graphics clocks to "
            f"{args.clock_min_mhz}-{args.clock_max_mhz} MHz. Type RUN to continue: "
        )
        if answer.strip() != "RUN":
            print("Cancelled; no hardware setting was changed.")
            return 0
    output = Path(args.output or (
        "molt-workspace/benchmarks/isoclock-" + time.strftime("%Y%m%d-%H%M%S")
    )).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    template = json.loads(Path(args.config).read_text(encoding="utf-8"))
    template.update(
        base_model=str(Path(args.model).resolve()),
        seed=args.seed,
        max_steps=args.steps,
        evaluation_interval=args.steps,
        qlora_validation_batches=4,
    )
    template["data"].update(
        path=str(Path(args.train_data).resolve()),
        validation_path=str(Path(args.validation_data).resolve()),
    )
    profile = GPUClockProfile("isoclock", args.clock_min_mhz, args.clock_max_mhz)
    order = ("molt", "unsloth") if args.order == "molt-first" else ("unsloth", "molt")
    results: dict[str, dict[str, Any]] = {}
    for engine in order:
        start_c = _stable_cold_start(
            args.start_temperature_c, args.cold_dwell_seconds, args.cooldown_timeout_seconds
        )
        arm = output / engine
        arm.mkdir()
        namespace = copy.copy(args)
        namespace.evaluation_interval = args.steps
        namespace.validation_batches = 4
        namespace.unsloth_thermal_target_c = 62.0
        namespace.unsloth_thermal_guard_c = 66.0
        namespace.unsloth_power_target_watts = 40.0
        namespace.unsloth_initial_pause_seconds = 0.30
        namespace.unsloth_max_pause_seconds = 1.0
        namespace.telemetry_interval_seconds = 0.1
        with temporary_graphics_clock(profile):
            raw = (
                _run_molt(template, arm, args.seed)
                if engine == "molt" else _run_unsloth(namespace, arm, args.seed)
            )
        normalized = _normalize(raw, profile)
        normalized["start_temperature_c"] = start_c
        results[engine] = normalized
        (output / "results.partial.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    reasons: list[str] = []
    for engine, result in results.items():
        if result["state"] != "completed":
            reasons.append(f"{engine} did not complete")
        if result["peak_temperature_c"] is None or result["peak_temperature_c"] > args.thermal_abort_c:
            reasons.append(f"{engine} exceeded or did not report the thermal boundary")
        if not result["clock_verification"]["verified"]:
            reasons.append(f"{engine} telemetry did not verify the requested clock range")
    if results["molt"]["trainable_parameters"] != results["unsloth"]["trainable_parameters"]:
        reasons.append("trainable parameter counts differ")
    for field in ("initial_nll", "final_nll"):
        molt_value = results["molt"][field]
        unsloth_value = results["unsloth"][field]
        if not isinstance(molt_value, (int, float)) or not isinstance(unsloth_value, (int, float)):
            reasons.append(f"both engines must report {field}")
        elif unsloth_value == 0 or abs(molt_value / unsloth_value - 1.0) * 100 > args.quality_tolerance_percent:
            reasons.append(f"{field} differs by more than {args.quality_tolerance_percent:g}%")
    record = {
        "schema_version": 1,
        "protocol": "qwen2.5-1.5b-isoclock-single-seed-screen",
        "seed": args.seed,
        "order": list(order),
        "clock_mhz": [args.clock_min_mhz, args.clock_max_mhz],
        "telemetry_interval_seconds": 0.1,
        "results": results,
        "valid": not reasons,
        "invalid_reasons": reasons,
        "limitations": [
            "One seed and one run order are insufficient for a general competitor claim.",
            "A 16-update screen is not a 30-minute endurance comparison.",
            "Board energy excludes host CPU, display, storage, and cooling-system energy.",
        ],
    }
    (output / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("\n" + _table(results))
    print(f"\nArtifacts: {output}")
    if reasons:
        print("Invalid comparison: " + "; ".join(reasons), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
