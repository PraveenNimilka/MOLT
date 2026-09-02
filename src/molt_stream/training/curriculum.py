from __future__ import annotations

import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.core.specs import TrainingSpec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.kernels.compiler import prepare_execution_model
from molt_stream.measurement.telemetry import NVMLTelemetry, integrate_board_energy
from molt_stream.measurement.thermal import build_thermal_controller, latest_telemetry_point
from molt_stream.training.model import SmallCausalLM


@dataclass(frozen=True)
class CurriculumPhase:
    """One cumulative fraction of training and its active attention context."""

    end_fraction: float
    context_length: int


@dataclass(frozen=True)
class PhaseGeometry:
    start_step: int
    end_step: int
    context_length: int
    batch_size: int


DEFAULT_PHASES = (
    CurriculumPhase(0.30, 128),
    CurriculumPhase(0.60, 256),
    CurriculumPhase(1.00, 512),
)


def curriculum_geometry(
    *,
    max_steps: int,
    full_context: int,
    full_batch_size: int,
    phases: Iterable[CurriculumPhase],
) -> tuple[PhaseGeometry, ...]:
    """Resolve phases while preserving exact tokens per optimizer update."""
    if min(max_steps, full_context, full_batch_size) <= 0:
        raise ValueError("steps, context, and batch size must be positive")
    phase_values = tuple(phases)
    if not phase_values or phase_values[-1].end_fraction != 1.0:
        raise ValueError("the final curriculum phase must end at 1.0")
    effective_tokens = full_context * full_batch_size
    start = 0
    result: list[PhaseGeometry] = []
    previous_fraction = 0.0
    for index, phase in enumerate(phase_values):
        if not previous_fraction < phase.end_fraction <= 1.0:
            raise ValueError("phase end fractions must be strictly increasing")
        if phase.context_length <= 0 or phase.context_length > full_context:
            raise ValueError("phase context must be in (0, full_context]")
        if effective_tokens % phase.context_length:
            raise ValueError("every phase context must divide the fixed tokens per update")
        end = max(start + 1, round(max_steps * phase.end_fraction))
        if index == len(phase_values) - 1:
            end = max_steps
        end = min(end, max_steps)
        result.append(
            PhaseGeometry(start, end, phase.context_length, effective_tokens // phase.context_length)
        )
        start = end
        previous_fraction = phase.end_fraction
    if result[-1].end_step != max_steps:
        raise ValueError("curriculum does not cover all training steps")
    return tuple(item for item in result if item.end_step > item.start_step)


def interpolate_crossing(
    evaluations: list[dict[str, float]], target_nll: float
) -> dict[str, float] | None:
    """Linearly interpolate the first measured held-out NLL crossing."""
    for index, current in enumerate(evaluations):
        if current["nll"] <= target_nll:
            if index == 0:
                return {
                    "seconds": current["elapsed_seconds"],
                    "energy_joules": current["energy_joules"],
                }
            previous = evaluations[index - 1]
            high, low = previous["nll"], current["nll"]
            fraction = 1.0 if high == low else (high - target_nll) / (high - low)
            fraction = min(1.0, max(0.0, fraction))
            return {
                "seconds": previous["elapsed_seconds"]
                + fraction * (current["elapsed_seconds"] - previous["elapsed_seconds"]),
                "energy_joules": previous["energy_joules"]
                + fraction * (current["energy_joules"] - previous["energy_joules"]),
            }
    return None


def paired_improvement(
    *, baseline: dict[str, Any], candidate: dict[str, Any], quality_tolerance: float = 0.01
) -> dict[str, Any]:
    baseline_time = baseline.get("time_to_target_seconds")
    candidate_time = candidate.get("time_to_target_seconds")
    baseline_energy = baseline.get("energy_to_target_joules")
    candidate_energy = candidate.get("energy_to_target_joules")
    quality_ok = candidate["final_nll"] <= baseline["final_nll"] * (1 + quality_tolerance)
    time_gain = (
        100.0 * (1.0 - float(candidate_time) / float(baseline_time))
        if baseline_time and candidate_time is not None else None
    )
    energy_gain = (
        100.0 * (1.0 - float(candidate_energy) / float(baseline_energy))
        if baseline_energy and candidate_energy is not None else None
    )
    useful = bool(
        quality_ok and time_gain is not None and energy_gain is not None
        and time_gain >= 1.0 and energy_gain >= 1.0
    )
    return {
        "quality_within_tolerance": quality_ok,
        "time_improvement_percent": time_gain,
        "energy_improvement_percent": energy_gain,
        "useful_gate": useful,
        "strong_gate": bool(useful and time_gain >= 25.0 and energy_gain >= 25.0),
        "breakthrough_gate": bool(useful and time_gain >= 50.0 and energy_gain >= 50.0),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gpu_temperature_c() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return float(pynvml.nvmlDeviceGetTemperature(handle, 0))
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _wait_for_temperature_bin(
    device: torch.device,
    *,
    maximum_c: float = 60.0,
    dwell_seconds: float = 5.0,
    timeout_seconds: float = 120.0,
) -> dict[str, float | bool | None]:
    """Normalize paired-arm temperature and require a short stable dwell."""
    started = time.perf_counter()
    initial = _gpu_temperature_c() if device.type == "cuda" else None
    current = initial
    within_since: float | None = None
    while current is not None:
        now = time.perf_counter()
        if current <= maximum_c:
            within_since = within_since or now
            if now - within_since >= dwell_seconds:
                break
        else:
            within_since = None
        if time.perf_counter() - started >= timeout_seconds:
            break
        time.sleep(1.0)
        current = _gpu_temperature_c()
    return {
        "initial_temperature_c": initial,
        "starting_temperature_c": current,
        "wait_seconds": time.perf_counter() - started,
        "required_dwell_seconds": dwell_seconds,
        "temperature_bin_satisfied": (
            current is None
            or (
                current <= maximum_c
                and within_since is not None
                and time.perf_counter() - within_since >= dwell_seconds
            )
        ),
    }


@torch.no_grad()
def _validation_nll(
    model: SmallCausalLM, batcher: MMapTokenBatcher, *, batches: int = 4
) -> float:
    model.eval()
    state = batcher.state_dict()
    losses = []
    for _ in range(batches):
        x, y = batcher.batch(1)
        losses.append(float(model.loss(x, y)))
    batcher.load_state_dict(state)
    model.train()
    return statistics.fmean(losses)


def _run_arm(
    spec: TrainingSpec,
    *,
    seed: int,
    arm: str,
    geometry: tuple[PhaseGeometry, ...],
    eval_interval: int,
    cooldown: dict[str, float | bool | None],
) -> dict[str, Any]:
    device = torch.device(spec.stream.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CapabilityError("CUDA curriculum benchmark requested but unavailable")
    _seed(seed)
    telemetry = NVMLTelemetry(0.05, enable_gpu=device.type == "cuda")
    started = time.perf_counter()
    telemetry.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = SmallCausalLM(spec.model).to(device)
    execution_model = prepare_execution_model(model, spec.execution_backend)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, fused=device.type == "cuda"
    )
    train_batcher = MMapTokenBatcher(spec.data, seed=seed, device=device)
    validation_data = type(spec.data)(
        **{
            **asdict(spec.data),
            "path": spec.data.validation_path or spec.data.path,
            "validation_path": None,
        }
    )
    validation_batcher = MMapTokenBatcher(validation_data, seed=seed + 1, device=device)
    controller = build_thermal_controller(spec) if device.type == "cuda" else None
    evaluations: list[dict[str, float]] = []
    phase_index = 0
    tokens = 0
    thermal_abort = False
    total_pause_seconds = 0.0

    def evaluate(step: int) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        evaluations.append(
            {
                "step": float(step),
                "tokens": float(tokens),
                "nll": _validation_nll(model, validation_batcher),
                "elapsed_seconds": time.perf_counter() - started,
                "energy_joules": integrate_board_energy(telemetry.points) or 0.0,
            }
        )

    evaluate(0)
    step = 0
    while step < spec.max_steps:
        while step >= geometry[phase_index].end_step:
            phase_index += 1
        phase = geometry[phase_index]
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(spec.gradient_accumulation):
            x, y = train_batcher.batch(
                phase.batch_size, context_length=phase.context_length
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = execution_model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
                ) / spec.gradient_accumulation
            loss.backward()
            tokens += y.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step += 1
        if controller is not None:
            decision = controller.update(
                latest_telemetry_point(telemetry.points),
                step_seconds=max(time.perf_counter() - step_started, 1e-6),
            )
            if decision.abort:
                thermal_abort = True
                break
            if decision.pause_seconds:
                time.sleep(decision.pause_seconds)
                total_pause_seconds += decision.pause_seconds
        if step % eval_interval == 0 or step == spec.max_steps:
            evaluate(step)
    measured = telemetry.stop()
    total_seconds = time.perf_counter() - started
    final_nll = evaluations[-1]["nll"]
    return {
        "arm": arm,
        "seed": seed,
        "state": "thermal_abort" if thermal_abort else "completed",
        "geometry": [asdict(item) for item in geometry],
        "pre_arm_cooldown": cooldown,
        "evaluations": evaluations,
        "final_nll": final_nll,
        "total_tokens": tokens,
        "total_seconds": total_seconds,
        "total_energy_joules": measured["gpu_board_energy_joules"],
        "tokens_per_second": tokens / total_seconds if total_seconds else None,
        "peak_allocated_vram_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "thermal_pause_seconds": total_pause_seconds,
        "telemetry": {key: value for key, value in measured.items() if key != "points"},
    }


def _bootstrap_interval(values: list[float], *, seed: int = 20260902) -> list[float] | None:
    if len(values) < 3:
        return None
    generator = random.Random(seed)
    samples = [
        statistics.fmean(generator.choice(values) for _ in values) for _ in range(10_000)
    ]
    samples.sort()
    return [samples[249], samples[9749]]


def run_curriculum_experiment(
    spec: TrainingSpec,
    *,
    seeds: tuple[int, ...] = (1337, 2027, 4099),
    phases: tuple[CurriculumPhase, ...] = DEFAULT_PHASES,
    eval_interval: int | None = None,
) -> dict[str, Any]:
    """Run paired fixed-context versus sequence-warmup experiments.

    This is EXP-1016. It deliberately changes only attention context over time;
    parameter count, token/update budget, token order, optimizer, validation
    objective, and full-context finishing phase remain fixed.
    """
    spec.validate()
    if not seeds:
        raise ValueError("at least one seed is required")
    interval = eval_interval or max(1, spec.max_steps // 6)
    candidate_geometry = curriculum_geometry(
        max_steps=spec.max_steps,
        full_context=spec.model.context_length,
        full_batch_size=spec.batch_size,
        phases=phases,
    )
    baseline_geometry = (
        PhaseGeometry(0, spec.max_steps, spec.model.context_length, spec.batch_size),
    )
    pairs = []
    for pair_index, seed in enumerate(seeds):
        order = ("baseline", "candidate") if pair_index % 2 == 0 else ("candidate", "baseline")
        arms: dict[str, dict[str, Any]] = {}
        for arm in order:
            geometry = baseline_geometry if arm == "baseline" else candidate_geometry
            cooldown = _wait_for_temperature_bin(torch.device(spec.stream.device))
            arms[arm] = _run_arm(
                spec,
                seed=seed,
                arm=arm,
                geometry=geometry,
                eval_interval=interval,
                cooldown=cooldown,
            )
        baseline, candidate = arms["baseline"], arms["candidate"]
        target = baseline["final_nll"] * 1.01
        for item in (baseline, candidate):
            crossing = interpolate_crossing(item["evaluations"], target)
            item["target_nll"] = target
            item["time_to_target_seconds"] = crossing["seconds"] if crossing else None
            item["energy_to_target_joules"] = crossing["energy_joules"] if crossing else None
        decision = paired_improvement(baseline=baseline, candidate=candidate)
        decision["reliable"] = baseline["state"] == candidate["state"] == "completed"
        decision["thermal_safe"] = all(
            not item["telemetry"].get("thermal_throttle_observed", False)
            and (item["telemetry"].get("peak_gpu_temperature_c") or 0) <= 70.0
            for item in (baseline, candidate)
        )
        decision["matched_start_temperature"] = all(
            bool(item["pre_arm_cooldown"]["temperature_bin_satisfied"])
            for item in (baseline, candidate)
        )
        pairs.append({"seed": seed, "execution_order": list(order), "baseline": baseline, "candidate": candidate, "decision": decision})
    time_gains = [float(pair["decision"]["time_improvement_percent"]) for pair in pairs if pair["decision"]["time_improvement_percent"] is not None]
    energy_gains = [float(pair["decision"]["energy_improvement_percent"]) for pair in pairs if pair["decision"]["energy_improvement_percent"] is not None]
    all_pass = len(pairs) >= 3 and all(
        pair["decision"]["useful_gate"]
        and pair["decision"]["reliable"]
        and pair["decision"]["thermal_safe"]
        and pair["decision"]["matched_start_temperature"]
        for pair in pairs
    )
    time_ci = _bootstrap_interval(time_gains)
    energy_ci = _bootstrap_interval(energy_gains)
    promoted = bool(
        all_pass and time_ci and energy_ci and time_ci[0] >= 1.0 and energy_ci[0] >= 1.0
    )
    report = {
        "schema_version": 1,
        "experiment_id": "EXP-1016",
        "method": "sequence-length-warmup",
        "novelty_claim": "none; established prior art evaluated in MOLT",
        "immutable_controls": {
            "parameter_count": SmallCausalLM(spec.model).parameter_count,
            "tokens_per_optimizer_update": spec.batch_size * spec.model.context_length * spec.gradient_accumulation,
            "optimizer": "AdamW with FP32 parameter and optimizer state",
            "data_order": "same sequential contiguous token stream per paired seed",
            "validation_context": spec.model.context_length,
            "quality_tolerance": 0.01,
        },
        "spec": spec.to_dict(),
        "candidate_phases": [asdict(item) for item in candidate_geometry],
        "pairs": pairs,
        "aggregate": {
            "pair_count": len(pairs),
            "median_time_improvement_percent": statistics.median(time_gains) if time_gains else None,
            "median_energy_improvement_percent": statistics.median(energy_gains) if energy_gains else None,
            "time_improvement_bootstrap_95_percent": time_ci,
            "energy_improvement_bootstrap_95_percent": energy_ci,
            "promoted": promoted,
            "decision": "promote" if promoted else "reject-or-revise",
        },
    }
    output = Path(spec.artifacts_dir).parent / "experiments" / f"exp-1016-{time.strftime('%Y%m%d-%H%M%S')}.json"
    _atomic_json(output, report)
    report["artifact_path"] = str(output)
    return report
