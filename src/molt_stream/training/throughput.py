from __future__ import annotations

import json
import os
import shutil
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from molt_stream.core.specs import GateSpec, TrainingMode, TrainingSpec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.measurement.telemetry import NVMLTelemetry, integrate_board_energy, manage_power_limit
from molt_stream.measurement.thermal import (
    build_thermal_controller,
    duty_cycle_pause_seconds,
    latest_telemetry_point,
    latest_temperature_c,
)
from molt_stream.kernels.compiler import configure_windows_compiler_cache, prepare_execution_model
from molt_stream.training.model import SmallCausalLM


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def training_throughput_benchmark(
    spec: TrainingSpec,
    *,
    warmup_steps: int = 10,
    measured_steps: int = 50,
    thermal_limit_c: float | None = None,
) -> dict[str, Any]:
    """Measure complete forward/backward/AdamW updates after explicit warm-up.

    Initialization, warm-up and allocator setup are reported separately from the
    sustained interval. This command is a benchmark, not a quality training run.
    """
    spec.validate()
    if spec.mode != TrainingMode.PRETRAIN:
        raise ValueError("throughput benchmark currently supports pretraining only")
    if spec.stream.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("production throughput benchmark requires CUDA")
    if min(warmup_steps, measured_steps) <= 0:
        raise ValueError("warmup_steps and measured_steps must be positive")
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    device = torch.device("cuda")
    abort_c = spec.thermal_abort_c if thermal_limit_c is None else thermal_limit_c
    total_started = time.perf_counter()
    telemetry = NVMLTelemetry(0.05)
    telemetry.start()
    time.sleep(0.06)
    initial_point = latest_telemetry_point(telemetry.points)
    initial_temperature_c = (
        float(initial_point.gpu_temperature_c)
        if initial_point is not None and initial_point.gpu_temperature_c is not None
        else None
    )
    power_limit_status = (
        manage_power_limit(spec.power_limit_watts, apply=False)
        if spec.power_limit_watts is not None else None
    )
    model = SmallCausalLM(spec.model).to(device)
    training_model = prepare_execution_model(model, spec.execution_backend)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, fused=True)
    batcher = MMapTokenBatcher(spec.data, seed=spec.seed, device=device)

    def update() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(spec.gradient_accumulation):
            x, y = batcher.batch(spec.batch_size)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = training_model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
                ) / spec.gradient_accumulation
            loss.backward()
            loss_sum += float(loss.detach())
        optimizer.step()
        return loss_sum

    pause_count = 0
    total_pause_seconds = 0.0
    cruise = build_thermal_controller(spec)
    latest_decision = None

    def regulate_temperature(step_seconds: float) -> bool:
        nonlocal pause_count, total_pause_seconds, latest_decision
        temperature = latest_temperature_c(telemetry.points)
        if cruise is not None:
            latest_decision = cruise.update(
                latest_telemetry_point(telemetry.points), step_seconds=max(step_seconds, 1e-6)
            )
            if latest_decision.abort:
                return False
            pause = latest_decision.pause_seconds
        else:
            latest_decision = None
            if temperature is not None and temperature >= abort_c:
                return False
            pause = duty_cycle_pause_seconds(
                temperature,
                target_c=spec.thermal_target_c,
                abort_c=abort_c,
                nominal_seconds=spec.thermal_pause_seconds,
            )
        if pause:
            time.sleep(pause)
            pause_count += 1
            total_pause_seconds += pause
        post_pause_temperature = latest_temperature_c(telemetry.points)
        return post_pause_temperature is None or post_pause_temperature < abort_c

    thermal_abort = False
    completed_warmup = 0
    for _ in range(warmup_steps):
        step_started = time.perf_counter()
        update()
        torch.cuda.synchronize()
        completed_warmup += 1
        if not regulate_temperature(time.perf_counter() - step_started):
            thermal_abort = True
            break
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    measured_started = time.perf_counter()
    losses: list[float] = []
    step_samples: list[dict[str, Any]] = []
    completed = 0
    for _ in range(0 if thermal_abort else measured_steps):
        step_started = time.perf_counter()
        losses.append(update())
        torch.cuda.synchronize()
        completed += 1
        compute_seconds = time.perf_counter() - step_started
        if not regulate_temperature(compute_seconds):
            thermal_abort = True
            break
        inclusive_seconds = time.perf_counter() - step_started
        point = latest_telemetry_point(telemetry.points)
        tokens_this_step = spec.batch_size * spec.gradient_accumulation * spec.model.context_length
        step_samples.append({
            "step": completed,
            "tokens_per_second": tokens_this_step / inclusive_seconds,
            "compute_seconds": compute_seconds,
            "inclusive_seconds": inclusive_seconds,
            "pause_seconds": latest_decision.pause_seconds if latest_decision is not None else max(0.0, inclusive_seconds - compute_seconds),
            "temperature_c": float(point.gpu_temperature_c) if point and point.gpu_temperature_c is not None else None,
            "power_watts": float(point.gpu_power_watts) if point and point.gpu_power_watts is not None else None,
            "gpu_utilization_percent": float(point.gpu_utilization_percent) if point and point.gpu_utilization_percent is not None else None,
            "thermal_phase": latest_decision.phase if latest_decision is not None else "reactive",
            "projected_temperature_c": latest_decision.projected_temperature_c if latest_decision is not None else None,
        })
    measured_seconds = time.perf_counter() - measured_started
    measured = telemetry.stop()
    interval_points = [point for point in telemetry.points if point.monotonic_seconds >= measured_started]
    interval_energy = integrate_board_energy(interval_points)
    interval_temperatures = [
        float(point.gpu_temperature_c)
        for point in interval_points
        if point.gpu_temperature_c is not None
    ]
    interval_powers = [
        float(point.gpu_power_watts)
        for point in interval_points
        if point.gpu_power_watts is not None
    ]
    interval_utilization = [
        float(point.gpu_utilization_percent)
        for point in interval_points
        if point.gpu_utilization_percent is not None
    ]
    interval_reason_mask = 0
    for point in interval_points:
        if point.gpu_clock_event_reasons is not None:
            interval_reason_mask |= int(point.gpu_clock_event_reasons)
    interval_peak_temperature = max(interval_temperatures) if interval_temperatures else None
    tokens = completed * spec.batch_size * spec.gradient_accumulation * spec.model.context_length
    section = max(1, len(step_samples) // 3)
    early_speed = statistics.fmean(item["tokens_per_second"] for item in step_samples[:section]) if step_samples else None
    late_speed = statistics.fmean(item["tokens_per_second"] for item in step_samples[-section:]) if step_samples else None
    sampled_temperatures = [float(item["temperature_c"]) for item in step_samples if item["temperature_c"] is not None]
    steady_temperature_range = max(sampled_temperatures) - min(sampled_temperatures) if sampled_temperatures else None
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "molt-production-training-throughput",
        "workload": spec.to_dict(),
        "parameter_count": model.parameter_count,
        "initial_gpu_temperature_c": initial_temperature_c,
        "warmup_steps": warmup_steps,
        "completed_warmup_steps": completed_warmup,
        "completed_measured_steps": completed,
        "tokens": tokens,
        "measured_seconds": measured_seconds,
        "total_seconds_including_setup_and_warmup": time.perf_counter() - total_started,
        "sustained_tokens_per_second": tokens / measured_seconds if measured_seconds else None,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "thermal_abort": thermal_abort,
        "thermal_target_c": spec.thermal_target_c,
        "thermal_abort_c": abort_c,
        "thermal_pause_count": pause_count,
        "thermal_pause_seconds": total_pause_seconds,
        "thermal_control_mode": spec.thermal_control_mode,
        "step_samples": step_samples,
        "early_tokens_per_second": early_speed,
        "late_tokens_per_second": late_speed,
        "late_to_early_throughput_ratio": late_speed / early_speed if early_speed and late_speed else None,
        "measured_temperature_range_c": steady_temperature_range,
        "power_limit": power_limit_status,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "telemetry": {key: value for key, value in measured.items() if key != "points"},
        "measured_interval_gpu_board_energy_joules": interval_energy,
        "measured_interval_peak_temperature_c": interval_peak_temperature,
        "measured_interval_mean_power_watts": (
            statistics.fmean(interval_powers) if interval_powers else None
        ),
        "measured_interval_mean_gpu_utilization_percent": (
            statistics.fmean(interval_utilization) if interval_utilization else None
        ),
        "measured_interval_gpu_clock_event_reason_mask": interval_reason_mask,
        "measured_interval_thermal_throttle_observed": bool(
            interval_reason_mask & (0x20 | 0x40)
        ),
    }
    result["measured_interval_joules_per_token"] = (
        float(interval_energy) / tokens if interval_energy is not None and tokens else None
    )
    result["end_to_end_joules_per_measured_token"] = (
        float(measured["gpu_board_energy_joules"]) / tokens
        if measured["gpu_board_energy_joules"] is not None and tokens else None
    )
    result["gates"] = {
        "throughput_35000": bool(result["sustained_tokens_per_second"] and result["sustained_tokens_per_second"] >= GateSpec().minimum_small_model_tokens_per_second),
        "throughput_150000": bool(result["sustained_tokens_per_second"] and result["sustained_tokens_per_second"] >= 150_000.0),
        "throughput_80000": bool(result["sustained_tokens_per_second"] and result["sustained_tokens_per_second"] >= 80_000.0),
        "profile_allocated_vram_3_5gb": result["peak_allocated_vram_bytes"] <= GateSpec().maximum_vram_bytes,
        "thermal_62c": bool(measured["peak_gpu_temperature_c"] is not None and measured["peak_gpu_temperature_c"] <= 62.0 and not thermal_abort),
        "steady_temperature_range_3c": bool(len(step_samples) >= 9 and steady_temperature_range is not None and steady_temperature_range <= 3.0),
        "late_throughput_ratio_097": bool(len(step_samples) >= 9 and early_speed and late_speed and late_speed / early_speed >= 0.97),
        "no_thermal_clock_throttle": not bool(measured["thermal_throttle_observed"]),
        "no_measured_interval_thermal_clock_throttle": not bool(
            interval_reason_mask & (0x20 | 0x40)
        ),
        "below_configured_cruise_ceiling": bool(
            spec.thermal_cruise_max_c is not None
            and interval_peak_temperature is not None
            and interval_peak_temperature <= spec.thermal_cruise_max_c
        ),
        "zero_abort": not thermal_abort and completed == measured_steps,
    }
    result["gate_scope"] = (
        "The VRAM result applies only to this reported small-model workload; it does not "
        "validate the separate 1B pretraining or 8B QLoRA memory gate."
    )
    output = Path(spec.artifacts_dir).parent / "benchmarks" / f"production-throughput-{time.strftime('%Y%m%d-%H%M%S')}.json"
    _atomic_json(output, result)
    result["artifact_path"] = str(output)
    return result


def fusion_memory_benchmark(spec: TrainingSpec) -> dict[str, Any]:
    """Compare eager and max-autotune activation peaks, or record the blocker.

    Failure is an artifact, not an implicit eager fallback. The measured peak is
    incremental allocated CUDA memory from immediately before forward through
    backward; model and existing gradients are excluded from the baseline.
    """
    spec.validate()
    if spec.stream.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fusion benchmark requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(spec.seed)
    x, y = MMapTokenBatcher(spec.data, seed=spec.seed, device=device).batch(spec.batch_size)

    def measure(model: torch.nn.Module) -> dict[str, float | int]:
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
        loss.backward()
        torch.cuda.synchronize()
        return {
            "loss": float(loss.detach()),
            "seconds": time.perf_counter() - started,
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "incremental_activation_and_backward_bytes": max(0, torch.cuda.max_memory_allocated() - baseline),
        }

    eager = SmallCausalLM(spec.model).to(device)
    eager_result = measure(eager)
    del eager
    torch.cuda.empty_cache()
    # Triton-Windows cache keys are long enough to exceed Win32 MAX_PATH when
    # nested below the run store. Keep this deterministic cache at a short path.
    configure_windows_compiler_cache()
    compile_results: dict[str, dict[str, Any]] = {}
    for mode in ("max-autotune", "max-autotune-no-cudagraphs"):
        try:
            torch.manual_seed(spec.seed)
            candidate = SmallCausalLM(spec.model).to(device)
            compiled = torch.compile(candidate, mode=mode)
            compilation_pass = measure(compiled)
            steady_state = measure(compiled)
            compile_results[mode] = {
                "available": True,
                "first_compile_and_update_seconds": compilation_pass["seconds"],
                **steady_state,
            }
            del compiled, candidate
            torch.cuda.empty_cache()
        except Exception as exc:
            compile_results[mode] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    compiled_result = compile_results["max-autotune"]
    selected_result = next(
        (compile_results[mode] for mode in ("max-autotune", "max-autotune-no-cudagraphs") if compile_results[mode].get("available")),
        compiled_result,
    )
    eager_bytes = int(eager_result["incremental_activation_and_backward_bytes"])
    compiled_bytes = selected_result.get("incremental_activation_and_backward_bytes")
    reduction = (
        (1.0 - int(compiled_bytes) / eager_bytes) * 100.0
        if compiled_bytes is not None and eager_bytes
        else None
    )
    from torch.utils.cpp_extension import CUDA_HOME

    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "molt-windows-fusion-memory",
        "workload": spec.to_dict(),
        "eager": eager_result,
        "torch_compile": compile_results,
        "activation_memory_reduction_percent": reduction,
        "gate_40_percent": bool(reduction is not None and reduction >= 40.0),
        "native_extension_toolchain": {
            "msvc_cl": shutil.which("cl"),
            "nvcc": shutil.which("nvcc"),
            "ninja": shutil.which("ninja"),
            "cuda_home": CUDA_HOME,
            "ready": bool(shutil.which("cl") and shutil.which("nvcc") and shutil.which("ninja") and CUDA_HOME),
        },
        "fallback_policy": "No silent fallback: unavailable fusion fails its gate.",
    }
    output = Path(spec.artifacts_dir).parent / "benchmarks" / f"fusion-memory-{time.strftime('%Y%m%d-%H%M%S')}.json"
    _atomic_json(output, report)
    report["artifact_path"] = str(output)
    return report
