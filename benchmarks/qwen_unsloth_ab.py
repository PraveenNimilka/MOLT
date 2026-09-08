"""Preregistered all-layer MOLT versus Unsloth AB/BA acceptance harness."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from molt_stream.measurement.gpu_profile import (
    GPUClockProfile,
    temporary_graphics_clock,
)
from molt_stream.measurement.target import interpolate_nll_crossing
from molt_stream.kernels.execution_plan import resolve_decoder_architecture

SEEDS = (1337, 2027, 4099)
ORDERS = (("unsloth", "molt"), ("molt", "unsloth"))
TRIALS = tuple((seed, order) for seed in SEEDS for order in ORDERS)


def _thermal_policy(args: argparse.Namespace) -> dict[str, float]:
    """Return the single thermal policy shared by both benchmark arms."""

    return {
        "target_c": float(getattr(args, "thermal_target_c", 58.0)),
        "guard_c": float(getattr(args, "thermal_guard_c", 63.0)),
        "abort_c": float(args.thermal_abort_c),
        "lookahead_seconds": float(
            getattr(args, "thermal_lookahead_seconds", 2.5)
        ),
        "stability_band_c": float(
            getattr(args, "thermal_stability_band_c", 1.5)
        ),
        "power_target_watts": float(
            getattr(args, "thermal_power_target_watts", 30.0)
        ),
        "initial_pause_seconds": float(
            getattr(args, "thermal_initial_pause_seconds", 0.6)
        ),
        "maximum_pause_seconds": float(
            getattr(args, "thermal_max_pause_seconds", 1.5)
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_template_paths(template: dict[str, Any]) -> dict[str, Any]:
    """Resolve the same environment-backed paths consumed by TrainingSpec."""

    resolved = copy.deepcopy(template)
    path_fields = (
        (resolved, "base_model"),
        (resolved["data"], "path"),
        (resolved["data"], "validation_path"),
    )
    for owner, field in path_fields:
        value = str(owner[field])
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ValueError(
                f"MOLT template {field} contains an unresolved environment variable"
            )
        owner[field] = expanded
    return resolved


def _model_weight_manifest(model_root: Path) -> list[dict[str, Any]]:
    files = sorted(model_root.glob("*.safetensors"))
    if not files:
        files = sorted(model_root.glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"No supported model weight files under {model_root}")
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]


def _protocol_contract(args: argparse.Namespace, template: dict[str, Any]) -> dict[str, Any]:
    model_config_path = Path(args.model).resolve() / "config.json"
    model_root = model_config_path.parent
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    architecture = resolve_decoder_architecture(
        SimpleNamespace(model_type=model_config.get("model_type"))
    )
    data = template["data"]
    stream = template["stream"]
    expected_paths = {
        "base_model": (template.get("base_model"), args.model),
        "train_data": (data.get("path"), args.train_data),
        "validation_data": (data.get("validation_path"), args.validation_data),
    }
    mismatches = [
        name
        for name, (configured, requested) in expected_paths.items()
        if configured is None
        or Path(str(configured)).resolve() != Path(str(requested)).resolve()
    ]
    if mismatches:
        raise ValueError(
            "MOLT template and benchmark arguments disagree for: "
            + ", ".join(mismatches)
        )
    if (
        data.get("storage_dtype") != "int32"
        or data.get("packing") != "contiguous"
        or data.get("sequential") is not True
    ):
        raise ValueError(
            "matched benchmark requires sequential contiguous int32 token data"
        )
    if stream.get("lora_target_modules", "all-linear") != "all-linear":
        raise ValueError("matched benchmark requires all-linear LoRA adapters")
    return {
        "schema_version": 3,
        "architecture_family": architecture.family,
        "model_type": model_config.get("model_type"),
        "model_config_sha256": _sha256(model_config_path),
        "model_weights": _model_weight_manifest(model_root),
        "train_data_sha256": _sha256(Path(args.train_data).resolve()),
        "validation_data_sha256": _sha256(Path(args.validation_data).resolve()),
        "context_length": int(data["context_length"]),
        "batch_size": int(template["batch_size"]),
        "gradient_accumulation": int(template["gradient_accumulation"]),
        "lora_rank": int(stream["lora_rank"]),
        "lora_alpha": float(stream["lora_alpha"]),
        "lora_target_modules": stream.get("lora_target_modules", "all-linear"),
        "learning_rate": float(template["learning_rate"]),
        "optimizer": "torch.optim.AdamW(fused=True), FP32 trainable parameters",
        "loss": "exact unfiltered shifted causal cross-entropy",
        "memory_accounting": (
            "torch peak begins immediately before model construction and includes "
            "model load, adapter construction, graph capture, training, evaluation, "
            "and checkpoint preparation"
        ),
        "steps": int(args.steps),
        "evaluation_interval": int(args.evaluation_interval),
        "validation_batches": int(args.validation_batches),
        "thermal_policy": _thermal_policy(args),
        "startup_power_gate": {
            "expected_enforced_power_limit_watts": getattr(
                args, "expected_enforced_power_limit_watts", None
            ),
            "tolerance_watts": float(
                getattr(args, "startup_power_tolerance_watts", 0.5)
            ),
        },
        "clock_control": getattr(args, "clock_control", "managed"),
        "graphics_clock_mhz": (
            [int(args.graphics_clock_min_mhz), int(args.graphics_clock_max_mhz)]
            if getattr(args, "clock_control", "managed") == "managed"
            else None
        ),
        "seeds": list(SEEDS),
        "orders_per_seed": [list(order) for order in ORDERS],
        "molt_execution": {
            "attention_backend": template.get("qlora_attention_backend", "sdpa"),
            "sdpa_kernel": template.get("qlora_sdpa_kernel", "auto"),
            "checkpoint_stride": int(template.get("qlora_checkpoint_stride", 1)),
            "loss_backend": template.get("qlora_loss_backend", "analytical"),
            "native_nf4_roles": template.get("qlora_native_nf4_roles", ""),
            "frozen_rmsnorm_bf16": bool(
                template.get("qlora_frozen_rmsnorm_bf16", False)
            ),
            "static_cuda_graph": bool(stream.get("cuda_graphs", False)),
            "joint_cuda_graph": bool(
                template.get("qlora_joint_cuda_graph", False)
            ),
        },
    }


def _temperature_c() -> float:
    return _hardware_state()[0]


def _hardware_state() -> tuple[float, float]:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return (
            float(pynvml.nvmlDeviceGetTemperature(handle, 0)),
            float(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)) / 1000.0,
        )
    finally:
        pynvml.nvmlShutdown()


def _stable_cold_start(
    maximum_c: float,
    dwell_seconds: int,
    timeout_seconds: float,
    *,
    expected_power_limit_watts: float | None = None,
    power_tolerance_watts: float = 0.5,
) -> tuple[float, float]:
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    latest_c = latest_power = math.inf
    reference_power = expected_power_limit_watts
    while stable < dwell_seconds:
        latest_c, latest_power = _hardware_state()
        if reference_power is None:
            reference_power = latest_power
        power_stable = (
            math.isfinite(latest_power)
            and abs(latest_power - reference_power) <= power_tolerance_watts
        )
        stable = stable + 1 if latest_c <= maximum_c and power_stable else 0
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"GPU did not sustain <= {maximum_c} C and "
                f"{reference_power} +/- {power_tolerance_watts} W for "
                f"{dwell_seconds}s; latest={latest_c} C/{latest_power} W"
            )
        time.sleep(1.0)
    return latest_c, latest_power


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
        check=False,
    )
    (log_root.with_suffix(".stdout.txt")).write_text(process.stdout, encoding="utf-8")
    (log_root.with_suffix(".stderr.txt")).write_text(process.stderr, encoding="utf-8")
    return process


def _run_molt(
    template: dict[str, Any],
    root: Path,
    seed: int,
    thermal_policy: dict[str, float] | None = None,
) -> dict[str, Any]:
    spec = copy.deepcopy(template)
    policy = thermal_policy
    spec.update(
        seed=seed,
        artifacts_dir=str(root / "molt-runs"),
        # The harness applies the same external stable-cold-start gate before
        # each engine. Disable any product-level MOLT dwell so one arm is not
        # conditioned or charged twice.
        thermal_startup_max_c=None,
        thermal_startup_dwell_seconds=0.0,
    )
    if policy is not None:
        # Benchmark policy is a controlled variable, not an engine-specific
        # tuning surface.  Use MOLT's equivalent predictive controller with
        # exactly the values passed to the competitor runner.
        spec.update(
            thermal_control_mode="predictive-cruise",
            thermal_target_c=policy["target_c"],
            thermal_microbatch_guard_c=policy["guard_c"],
            thermal_abort_c=policy["abort_c"],
            thermal_lookahead_seconds=policy["lookahead_seconds"],
            thermal_stability_band_c=policy["stability_band_c"],
            thermal_power_target_watts=policy["power_target_watts"],
            thermal_initial_pause_seconds=policy["initial_pause_seconds"],
            thermal_max_pause_seconds=policy["maximum_pause_seconds"],
            thermal_recovery_mode="stop",
        )
    spec_path = root / "molt-config.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    process = _run_process(
        [sys.executable, "-m", "molt_stream.cli", "--json", "train", "--config", str(spec_path), "-y"],
        root / "molt-process",
    )
    try:
        run = Path(json.loads(process.stdout)["run"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"MOLT produced no run reference (exit {process.returncode}): "
            f"{process.stderr[-4000:]}"
        ) from exc
    metrics = json.loads((run / "metrics.summary.json").read_text(encoding="utf-8"))
    metrics["artifact"] = str(run)
    metrics["engine"] = "molt"
    metrics["process_returncode"] = process.returncode
    return metrics


def _run_unsloth(
    args: argparse.Namespace,
    root: Path,
    seed: int,
    template: dict[str, Any],
    thermal_policy: dict[str, float],
) -> dict[str, Any]:
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
        "--context-length", str(template["data"]["context_length"]),
        "--batch-size", str(template["batch_size"]),
        "--gradient-accumulation", str(template["gradient_accumulation"]),
        "--lora-rank", str(template["stream"]["lora_rank"]),
        "--lora-alpha", str(template["stream"]["lora_alpha"]),
        "--learning-rate", str(template["learning_rate"]),
        "--thermal-target-c", str(thermal_policy["target_c"]),
        "--thermal-guard-c", str(thermal_policy["guard_c"]),
        "--thermal-abort-c", str(thermal_policy["abort_c"]),
        "--thermal-lookahead-seconds", str(thermal_policy["lookahead_seconds"]),
        "--thermal-stability-band-c", str(thermal_policy["stability_band_c"]),
        "--thermal-power-target-watts", str(thermal_policy["power_target_watts"]),
        "--thermal-initial-pause-seconds", str(thermal_policy["initial_pause_seconds"]),
        "--thermal-max-pause-seconds", str(thermal_policy["maximum_pause_seconds"]),
        "--telemetry-interval-seconds", str(args.telemetry_interval_seconds),
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


def _trial(
    metrics: dict[str, Any], target_nll: float, start_c: float, start_power_watts: float
) -> dict[str, Any]:
    evaluations = metrics.get("evaluations", [])
    telemetry = metrics.get("telemetry", {})
    return {
        "artifact": metrics["artifact"],
        "state": metrics.get("state"),
        "start_temperature_c": start_c,
        "start_enforced_power_limit_watts": start_power_watts,
        "peak_temperature_c": telemetry.get("peak_gpu_temperature_c"),
        "trainable_parameters": metrics.get("active_trainable_parameter_count", metrics.get("trainable_parameters")),
        "steps": metrics.get("step", metrics.get("steps")),
        "compute_tokens_per_second": metrics.get("committed_update_compute_tokens_per_second", metrics.get("compute_tokens_per_second")),
        "end_to_end_tokens_per_second": metrics.get("tokens_per_second", metrics.get("end_to_end_tokens_per_second")),
        "allocated_bytes": metrics.get("cuda_peak_allocated_bytes"),
        "reserved_bytes": metrics.get("cuda_peak_reserved_bytes"),
        "nvml_used_bytes": telemetry.get("peak_gpu_used_bytes"),
        "process_nvml_used_bytes": telemetry.get("peak_process_gpu_used_bytes"),
        "minimum_enforced_power_limit_watts": telemetry.get(
            "minimum_enforced_power_limit_watts"
        ),
        "maximum_enforced_power_limit_watts": telemetry.get(
            "maximum_enforced_power_limit_watts"
        ),
        "execution_plan_fingerprint": (
            metrics.get("execution_plan", {}).get("fingerprint")
            if isinstance(metrics.get("execution_plan"), dict)
            else None
        ),
        "initial_nll": evaluations[0]["nll"] if evaluations else None,
        "final_nll": evaluations[-1]["nll"] if evaluations else None,
        "crossing": interpolate_nll_crossing(evaluations, target_nll) if evaluations else None,
    }


def _power_envelopes_match(
    left: dict[str, Any], right: dict[str, Any], tolerance_watts: float = 1.0
) -> bool:
    fields = (
        "minimum_enforced_power_limit_watts",
        "maximum_enforced_power_limit_watts",
    )
    if any(left.get(field) is None or right.get(field) is None for field in fields):
        return False
    return all(
        abs(float(left[field]) - float(right[field])) <= tolerance_watts
        for field in fields
    )


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
    parser.add_argument("--thermal-target-c", type=float, default=58.0)
    parser.add_argument("--thermal-guard-c", type=float, default=63.0)
    parser.add_argument("--thermal-lookahead-seconds", type=float, default=2.5)
    parser.add_argument("--thermal-stability-band-c", type=float, default=1.5)
    parser.add_argument("--thermal-power-target-watts", type=float, default=30.0)
    parser.add_argument("--thermal-initial-pause-seconds", type=float, default=0.6)
    parser.add_argument("--thermal-max-pause-seconds", type=float, default=1.5)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=0.1)
    parser.add_argument("--expected-enforced-power-limit-watts", type=float)
    parser.add_argument("--startup-power-tolerance-watts", type=float, default=0.5)
    parser.add_argument("--graphics-clock-min-mhz", type=int, default=1500)
    parser.add_argument("--graphics-clock-max-mhz", type=int, default=1650)
    parser.add_argument(
        "--clock-control",
        choices=("managed", "uncontrolled-diagnostic"),
        default="managed",
        help=(
            "managed enforces/restores the registered clock; uncontrolled-diagnostic "
            "never changes clocks and can never pass the official gate"
        ),
    )
    args = parser.parse_args()
    if min(args.steps, args.evaluation_interval, args.validation_batches, args.cold_dwell_seconds) < 1:
        parser.error("counts must be positive")
    if args.startup_power_tolerance_watts < 0:
        parser.error("startup power tolerance must be non-negative")
    if (
        args.expected_enforced_power_limit_watts is not None
        and args.expected_enforced_power_limit_watts <= 0
    ):
        parser.error("expected enforced power limit must be positive")
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    thermal_policy = _thermal_policy(args)
    clock_profile = GPUClockProfile(
        "matched-comparison", args.graphics_clock_min_mhz, args.graphics_clock_max_mhz
    )
    template = _expand_template_paths(
        json.loads(Path(args.molt_template).read_text(encoding="utf-8"))
    )
    template.update(max_steps=args.steps, evaluation_interval=args.evaluation_interval,
                    qlora_validation_batches=args.validation_batches)
    contract = _protocol_contract(args, template)
    contract_bytes = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    _atomic_json(root / "protocol.contract.json", {
        **contract, "contract_sha256": contract_sha256
    })
    pairs: list[dict[str, Any]] = []
    for pair_index, (seed, order) in enumerate(TRIALS, start=1):
        trials: dict[str, dict[str, Any]] = {}
        for engine in order:
            start_c, start_power = _stable_cold_start(
                args.start_temperature_c,
                args.cold_dwell_seconds,
                args.cooldown_timeout_seconds,
                expected_power_limit_watts=args.expected_enforced_power_limit_watts,
                power_tolerance_watts=args.startup_power_tolerance_watts,
            )
            trial_root = root / f"pair{pair_index:02d}-seed{seed}-{order[0]}-first-{engine}"
            trial_root.mkdir()
            clock_context = (
                temporary_graphics_clock(clock_profile)
                if args.clock_control == "managed"
                else nullcontext()
            )
            with clock_context:
                metrics = (
                    _run_molt(template, trial_root, seed, thermal_policy)
                    if engine == "molt"
                    else _run_unsloth(
                        args, trial_root, seed, template, thermal_policy
                    )
                )
            trials[engine] = _trial(metrics, args.target_nll, start_c, start_power)
            trials[engine]["contract_sha256"] = contract_sha256
        molt, unsloth = trials["molt"], trials["unsloth"]
        reasons: list[str] = []
        if args.clock_control != "managed":
            reasons.append("diagnostic run did not enforce identical graphics clocks")
        if molt["state"] != "completed" or unsloth["state"] != "completed":
            reasons.append("both engines must complete")
        if molt["crossing"] is None or unsloth["crossing"] is None:
            reasons.append("both engines must reach target NLL")
        if molt["trainable_parameters"] != unsloth["trainable_parameters"]:
            reasons.append("trainable parameter mismatch")
        if not _power_envelopes_match(molt, unsloth):
            reasons.append("enforced GPU power envelopes differ or are unavailable")
        if (
            molt["initial_nll"] is None
            or unsloth["initial_nll"] is None
            or abs(molt["initial_nll"] / unsloth["initial_nll"] - 1.0) > 0.01
        ):
            reasons.append("initial validation objective mismatch exceeds 1%")
        if max(molt["peak_temperature_c"] or math.inf, unsloth["peak_temperature_c"] or math.inf) > args.thermal_abort_c:
            reasons.append("thermal boundary exceeded")
        comparison: dict[str, float] = {}
        if molt["crossing"] is not None and unsloth["crossing"] is not None:
            comparison = {
                "molt_time_improvement_percent": 100.0 * (1.0 - molt["crossing"]["seconds"] / unsloth["crossing"]["seconds"]),
                "molt_energy_improvement_percent": 100.0 * (1.0 - molt["crossing"]["joules"] / unsloth["crossing"]["joules"]),
            }
            if min(comparison.values()) <= 0:
                reasons.append("MOLT must improve both time and energy to target")
        pairs.append({"seed": seed, "order": list(order),
                      "contract_sha256": contract_sha256, "trials": trials,
                      "comparison": comparison, "passed": not reasons, "reasons": reasons})
        _atomic_json(root / "results.partial.json", pairs)
    result = {
        "schema_version": 1,
        "experiment": f"{contract['architecture_family']}-all-layer-molt-vs-unsloth",
        "contract_sha256": contract_sha256,
        "target_nll": args.target_nll,
        "registered_seeds": list(SEEDS),
        "registered_orders_per_seed": [list(order) for order in ORDERS],
        "pairs": pairs,
        "passed": len(pairs) == len(TRIALS) and all(pair["passed"] for pair in pairs),
        "limitations": [
            "A short screen does not satisfy the separate 30-minute endurance gate",
            "A passing result is specific to the hashed model, data, and software environment",
        ],
    }
    _atomic_json(root / "results.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
