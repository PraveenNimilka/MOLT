from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.core.contracts import ProgressEvent
from molt_stream.core.specs import TrainingMode, TrainingSpec, load_spec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.experiments.store import AtomicCheckpointStore
from molt_stream.measurement.telemetry import NVMLTelemetry, manage_power_limit
from molt_stream.measurement.thermal import (
    build_thermal_controller,
    duty_cycle_pause_seconds,
    latest_telemetry_point,
    latest_temperature_c,
    wait_for_stable_thermal_headroom,
)
from molt_stream.kernels.compiler import prepare_execution_model
from molt_stream.training.galore import GaLoreAdamW
from molt_stream.training.model import SmallCausalLM


def _write_json(path: Path, value: Any) -> None:
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


@torch.no_grad()
def _evaluate(model: SmallCausalLM, batcher: MMapTokenBatcher, batches: int = 4) -> float:
    model.eval()
    state = batcher.state_dict()
    losses = []
    for _ in range(batches):
        x, y = batcher.batch(1)
        losses.append(float(model.loss(x, y)))
    batcher.load_state_dict(state)
    model.train()
    return sum(losses) / len(losses)


def train(
    spec: TrainingSpec,
    *,
    resume: str | Path | None = None,
    use_galore: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> Path:
    spec.validate()
    if spec.mode == TrainingMode.QLORA:
        from molt_stream.training.qlora import train_qlora

        return train_qlora(spec, resume=resume, progress=progress)
    device = torch.device(spec.stream.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CapabilityError("CUDA training requested but unavailable")
    if progress:
        device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
        progress(ProgressEvent("startup", f"Checking CUDA & {device_name}"))
    _seed(spec.seed)
    if resume:
        run = Path(resume)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run = Path(spec.artifacts_dir) / f"{stamp}-pretrain-{uuid.uuid4().hex[:8]}"
        run.mkdir(parents=True, exist_ok=False)
        _write_json(run / "spec.resolved.json", spec.to_dict())
    store = AtomicCheckpointStore(run)
    resume_state: dict[str, Any] | None = None
    if resume:
        resume_state = store.load(map_location="cpu")
        saved_step = int(resume_state["step"])
        if resume_state.get("termination_reason") == "completed" or saved_step >= spec.max_steps:
            raise ValueError(f"run is already complete at step {saved_step}")
    total_started = time.perf_counter()
    telemetry = NVMLTelemetry(0.1, enable_gpu=device.type == "cuda")
    telemetry.start()
    startup_cooling_seconds = 0.0
    if device.type == "cuda" and spec.thermal_startup_max_c is not None:
        startup = wait_for_stable_thermal_headroom(
            telemetry.thermal_point,
            maximum_c=spec.thermal_startup_max_c,
            dwell_seconds=spec.thermal_startup_dwell_seconds,
            maximum_wait_seconds=spec.thermal_startup_timeout_seconds,
            notify=(
                lambda paused, current: progress(
                    ProgressEvent(
                        "startup",
                        message=(
                            f"Cooling to <= {spec.thermal_startup_max_c:.1f} C "
                            f"for {spec.thermal_startup_dwell_seconds:.0f}s "
                            f"({current if current is not None else 'no sensor'} C)"
                        ),
                        thermal_state="startup-cooling",
                        gpu_temperature_c=current,
                        thermal_pause_seconds=paused,
                    )
                )
                if progress
                else None
            ),
        )
        if startup.stop_reason is not None:
            telemetry.stop()
            raise RuntimeError(startup.stop_reason)
        startup_cooling_seconds = startup.pause_seconds
    power_limit_status = (
        manage_power_limit(spec.power_limit_watts, apply=False)
        if device.type == "cuda" and spec.power_limit_watts is not None else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = SmallCausalLM(spec.model).to(device)
    training_model = prepare_execution_model(model, spec.execution_backend)
    optimizer = (
        GaLoreAdamW(model.parameters(), lr=spec.learning_rate, rank=min(64, spec.model.width // 4))
        if use_galore
        else torch.optim.AdamW(
            model.parameters(), lr=spec.learning_rate, fused=device.type == "cuda"
        )
    )
    batcher = MMapTokenBatcher(spec.data, seed=spec.seed, device=device)
    if progress:
        progress(ProgressEvent("startup", "Memory-mapped dataset initialized"))
        if spec.execution_backend == "eager":
            progress(ProgressEvent("startup", f"NVML thermal safety guard armed ({spec.thermal_target_c:.1f}°C)"))
    validation_spec = (
        type(spec.data)(**{**asdict(spec.data), "path": spec.data.validation_path, "validation_path": None})
        if spec.data.validation_path
        else spec.data
    )
    validation_batcher = MMapTokenBatcher(validation_spec, seed=spec.seed + 1, device=device)
    step = tokens = 0
    initial_tokens = 0
    initial_step = 0
    if resume_state is not None:
        state = resume_state
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        batcher.load_state_dict(state["batcher"])
        step, tokens = int(state["step"]), int(state["tokens"])
        initial_tokens = tokens
        initial_step = step
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    initial_nll = _evaluate(model, validation_batcher)
    evaluations = [{"step": step, "nll": initial_nll, "perplexity": math.exp(initial_nll)}]
    training_started = time.perf_counter()
    thermal_abort = False
    thermal_pause_count = 0
    thermal_pause_seconds = 0.0
    thermal_phase_counts: dict[str, int] = {}
    regulator = build_thermal_controller(spec) if device.type == "cuda" else None
    compiler_announced = spec.execution_backend == "eager"
    while step < spec.max_steps:
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(spec.gradient_accumulation):
            x, y = batcher.batch(spec.batch_size)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = training_model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
                ) / spec.gradient_accumulation
            loss.backward()
            loss_sum += float(loss.detach())
            tokens += y.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step += 1
        if progress and not compiler_announced:
            progress(ProgressEvent("startup", "Windows Triton kernel compilation completed"))
            progress(ProgressEvent("startup", f"NVML thermal safety guard armed ({spec.thermal_target_c:.1f}°C)"))
            compiler_announced = True
        temperature = latest_temperature_c(telemetry.points)
        thermal_state = "full-speed"
        if regulator is not None:
            decision = regulator.update(
                latest_telemetry_point(telemetry.points),
                step_seconds=max(time.perf_counter() - step_started, 1e-6),
            )
            pause = decision.pause_seconds
            thermal_abort = decision.abort
            thermal_state = decision.phase
            thermal_phase_counts[decision.phase] = thermal_phase_counts.get(decision.phase, 0) + 1
        else:
            thermal_abort = temperature is not None and temperature >= spec.thermal_abort_c
            pause = duty_cycle_pause_seconds(
                temperature,
                target_c=spec.thermal_target_c,
                abort_c=spec.thermal_abort_c,
                nominal_seconds=spec.thermal_pause_seconds,
            )
        if thermal_abort:
            if progress:
                progress(ProgressEvent(
                    "thermal_abort",
                    message=f"Thermal boundary reached ({temperature}°C); saving checkpoint",
                    step=step,
                    total_steps=spec.max_steps,
                    gpu_temperature_c=temperature,
                    thermal_state="thermal-abort",
                ))
            break
        if pause:
            time.sleep(pause)
            thermal_pause_count += 1
            thermal_pause_seconds += pause
            post_pause_temperature = latest_temperature_c(telemetry.points)
            if post_pause_temperature is not None and post_pause_temperature >= spec.thermal_abort_c:
                thermal_abort = True
                if progress:
                    progress(ProgressEvent(
                        "thermal_abort",
                        message=(
                            f"Thermal boundary reached ({post_pause_temperature:.1f}°C); "
                            "saving checkpoint"
                        ),
                        step=step,
                        total_steps=spec.max_steps,
                        gpu_temperature_c=post_pause_temperature,
                        thermal_state="thermal-abort",
                        initial_step=initial_step,
                    ))
                break
        session_tokens = tokens - initial_tokens
        if progress:
            elapsed = time.perf_counter() - training_started
            progress(ProgressEvent(
                "step",
                step=step,
                total_steps=spec.max_steps,
                elapsed_seconds=elapsed,
                tokens_per_second=session_tokens / elapsed if elapsed else None,
                loss=loss_sum,
                vram_bytes=(torch.cuda.memory_allocated() if device.type == "cuda" else None),
                gpu_temperature_c=temperature,
                thermal_state=thermal_state,
                thermal_pause_seconds=pause,
                initial_step=initial_step,
            ))
        if step == spec.max_steps or step % max(1, spec.max_steps // 4) == 0:
            nll = _evaluate(model, validation_batcher)
            evaluations.append({"step": step, "nll": nll, "perplexity": math.exp(nll), "train_loss": loss_sum})
    training_seconds = time.perf_counter() - training_started
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "batcher": batcher.state_dict(), "step": step, "tokens": tokens,
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "termination_reason": "thermal_abort" if thermal_abort else "completed",
    }
    checkpoint_path = store.save(state)
    measured = telemetry.stop()
    seconds = time.perf_counter() - total_started
    session_tokens = tokens - initial_tokens
    conditioned_seconds = max(0.0, seconds - startup_cooling_seconds)
    summary = {
        "state": "thermal_abort" if thermal_abort else "completed",
        "thermal_abort": thermal_abort, "mode": spec.mode, "step": step, "tokens": tokens,
        "session_tokens": session_tokens,
        "seconds": seconds, "training_loop_seconds": training_seconds,
        "tokens_per_second": session_tokens / seconds if seconds else 0.0,
        "startup_cooling_seconds": startup_cooling_seconds,
        "conditioned_session_seconds": conditioned_seconds,
        "conditioned_end_to_end_tokens_per_second": (
            session_tokens / conditioned_seconds if conditioned_seconds else None
        ),
        "training_loop_tokens_per_second": session_tokens / training_seconds if training_seconds else None,
        "training_loop_compute_tokens_per_second": (
            session_tokens / (training_seconds - thermal_pause_seconds)
            if training_seconds > thermal_pause_seconds else None
        ),
        "parameter_count": model.parameter_count, "evaluations": evaluations,
        "perplexity_monotonic": all(
            right["perplexity"] <= left["perplexity"]
            for left, right in zip(evaluations, evaluations[1:])
        ),
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "thermal_policy": {
            "mode": spec.thermal_control_mode,
            "target_c": spec.thermal_target_c,
            "cruise_max_c": spec.thermal_cruise_max_c,
            "abort_c": spec.thermal_abort_c,
            "pause_count": thermal_pause_count,
            "total_pause_seconds": thermal_pause_seconds,
            "phase_counts": thermal_phase_counts,
        },
        "power_limit": power_limit_status,
        "checkpoint": str(checkpoint_path),
        "telemetry": {key: value for key, value in measured.items() if key != "points"},
    }
    energy = measured.get("gpu_board_energy_joules")
    summary["joules_per_token"] = float(energy) / tokens if energy is not None and tokens else None
    _write_json(run / "metrics.summary.json", summary)
    return run


def evaluate_run(run: str | Path) -> dict[str, float]:
    root = Path(run)
    spec = load_spec(root / "spec.resolved.json")
    spec.validate()
    if spec.mode == TrainingMode.QLORA:
        from molt_stream.training.qlora import evaluate_qlora

        return evaluate_qlora(spec, root)
    device = torch.device(spec.stream.device)
    model = SmallCausalLM(spec.model).to(device)
    state = AtomicCheckpointStore(root).load(map_location=device)
    model.load_state_dict(state["model"])
    validation_spec = (
        type(spec.data)(**{**asdict(spec.data), "path": spec.data.validation_path, "validation_path": None})
        if spec.data.validation_path
        else spec.data
    )
    batcher = MMapTokenBatcher(validation_spec, seed=spec.seed + 1, device=device)
    nll = _evaluate(model, batcher, batches=8)
    return {"validation_nll": nll, "validation_perplexity": math.exp(nll)}


@torch.no_grad()
def generate_run(run: str | Path, prompt: list[int], max_new_tokens: int = 32) -> list[int]:
    root = Path(run)
    spec = load_spec(root / "spec.resolved.json")
    spec.validate()
    if spec.mode == TrainingMode.QLORA:
        from molt_stream.training.qlora import generate_qlora

        return generate_qlora(spec, root, prompt, max_new_tokens)
    device = torch.device(spec.stream.device)
    model = SmallCausalLM(spec.model).to(device)
    model.load_state_dict(AtomicCheckpointStore(root).load(map_location=device)["model"])
    model.eval()
    tokens = list(prompt)
    for _ in range(max_new_tokens):
        window = tokens[-spec.model.context_length :]
        value = torch.tensor(window, device=device).unsqueeze(0)
        next_token = int(model(value)[0, -1].argmax())
        tokens.append(next_token)
    return tokens
