from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
from torch.nn import functional as F

from molt_stream.core.contracts import ProgressEvent
from molt_stream.core.errors import CapabilityError
from molt_stream.core.specs import TrainingSpec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.experiments.store import AtomicCheckpointStore, sha256
from molt_stream.measurement.telemetry import NVMLTelemetry, integrate_board_energy
from molt_stream.measurement.thermal import (
    build_thermal_controller,
    duty_cycle_pause_seconds,
    latest_telemetry_point,
    latest_temperature_c,
)


def _shifted_causal_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute loss against MOLT's already shifted targets exactly once."""
    logits = model(input_ids=inputs, use_cache=False).logits
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _local_checkpoint_parameter_count(base_model: str | None) -> int | None:
    """Count stored logical tensor elements without loading a local checkpoint."""
    if base_model is None:
        return None
    root = Path(base_model)
    if not root.is_dir():
        return None
    try:
        from safetensors import safe_open

        total = 0
        for path in sorted(root.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    total += math.prod(handle.get_slice(key).get_shape())
        return total or None
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _requirements() -> None:
    missing = [
        name
        for name in ("transformers", "peft", "bitsandbytes")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise CapabilityError(
            "QLoRA requires missing optional packages: " + ", ".join(missing)
            + ". No full-precision fallback is permitted."
        )
    if not torch.cuda.is_available():
        raise CapabilityError("QLoRA requires CUDA")


def _resolve_lora_target_modules(value: str) -> str | list[str]:
    """Convert the stable config representation into PEFT's accepted shape."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    return "all-linear" if names == ["all-linear"] else names


def _build_qlora_optimizer(
    model: torch.nn.Module,
    spec: TrainingSpec,
) -> torch.optim.Optimizer:
    if spec.stream.lora_plus_lr_ratio is not None:
        from peft.optimizers import create_loraplus_optimizer

        return create_loraplus_optimizer(
            model=model,
            optimizer_cls=torch.optim.AdamW,
            lr=spec.learning_rate,
            loraplus_lr_ratio=spec.stream.lora_plus_lr_ratio,
            loraplus_weight_decay=0.01,
        )
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=spec.learning_rate,
    )


def _build_qlora_model(spec: TrainingSpec) -> torch.nn.Module:
    _requirements()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if spec.base_model is None:
        raise ValueError("QLoRA requires base_model")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec.base_model,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=spec.activation_checkpointing,
    )
    return get_peft_model(
        model,
        LoraConfig(
            r=spec.stream.lora_rank,
            lora_alpha=int(spec.stream.lora_alpha),
            target_modules=_resolve_lora_target_modules(spec.stream.lora_target_modules),
            task_type="CAUSAL_LM",
        ),
    )


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def train_qlora(
    spec: TrainingSpec,
    *,
    resume: str | Path | None = None,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> Path:
    """Optional standard NF4 QLoRA engine.

    The verified custom streamed-layer path is not yet wired into arbitrary
    Hugging Face decoder graphs, so this function makes no 3.5 GB streaming claim.
    """
    _requirements()
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    assert spec.base_model is not None
    random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    run = Path(resume) if resume else Path(spec.artifacts_dir) / (
        f"{time.strftime('%Y%m%d-%H%M%S')}-qlora-{uuid.uuid4().hex[:8]}"
    )
    if not resume:
        run.mkdir(parents=True, exist_ok=False)
        _atomic_json(run / "spec.resolved.json", spec.to_dict())
    store = AtomicCheckpointStore(run)
    resume_state = store.load(map_location="cpu") if resume else None
    if resume_state is not None:
        saved_step = int(resume_state["step"])
        if resume_state.get("termination_reason") == "completed" or saved_step >= spec.max_steps:
            raise ValueError(f"run is already complete at step {saved_step}")
    telemetry = NVMLTelemetry(0.1)
    telemetry.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = _build_qlora_model(spec)
    optimizer = _build_qlora_optimizer(model, spec)
    batcher = MMapTokenBatcher(spec.data, seed=spec.seed, device="cuda")
    validation_spec = type(spec.data)(
        **{**asdict(spec.data), "path": spec.data.validation_path or spec.data.path, "validation_path": None}
    )
    validation = MMapTokenBatcher(validation_spec, seed=spec.seed + 1, device="cuda")
    setup_seconds = time.perf_counter() - started
    quantized_parameter_elements = sum(parameter.numel() for parameter in model.parameters())
    base_checkpoint_parameter_count = _local_checkpoint_parameter_count(spec.base_model)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    step = tokens = 0
    initial_tokens = 0
    initial_step = 0
    if resume_state is not None:
        state = resume_state
        set_peft_model_state_dict(model, state["adapters"])
        optimizer.load_state_dict(state["optimizer"])
        batcher.load_state_dict(state["batcher"])
        step, tokens = int(state["step"]), int(state["tokens"])
        initial_tokens = tokens
        initial_step = step
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    @torch.no_grad()
    def validation_nll() -> float:
        model.eval()
        state = validation.state_dict()
        losses = []
        for _ in range(4):
            x, y = validation.batch(1)
            losses.append(float(_shifted_causal_loss(model, x, y)))
        validation.load_state_dict(state)
        model.train()
        return sum(losses) / len(losses)

    validation_started = time.perf_counter()
    initial = validation_nll()
    validation_seconds = time.perf_counter() - validation_started
    def evaluation_record(nll: float) -> dict[str, float | int | None]:
        return {
            "step": step,
            "nll": nll,
            "perplexity": math.exp(nll),
            "end_to_end_seconds": time.perf_counter() - started,
            "board_energy_joules": integrate_board_energy(telemetry.points),
        }

    evaluations = [evaluation_record(initial)]
    thermal_abort = False
    thermal_pause_seconds = 0.0
    update_compute_seconds = 0.0
    regulator = build_thermal_controller(spec)
    evaluation_interval = spec.evaluation_interval or max(1, spec.max_steps // 4)
    training_started = time.perf_counter()
    while step < spec.max_steps:
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(spec.gradient_accumulation):
            x, y = batcher.batch(spec.batch_size)
            micro_loss = _shifted_causal_loss(model, x, y) / spec.gradient_accumulation
            micro_loss.backward()
            loss_sum += float(micro_loss.detach())
            tokens += y.numel()
        optimizer.step()
        torch.cuda.synchronize()
        update_compute_seconds += time.perf_counter() - step_started
        step += 1
        temperature = latest_temperature_c(telemetry.points)
        if regulator is not None:
            decision = regulator.update(
                latest_telemetry_point(telemetry.points),
                step_seconds=max(time.perf_counter() - step_started, 1e-6),
            )
            pause = decision.pause_seconds
            thermal_abort = decision.abort
            thermal_state = decision.phase
        else:
            thermal_abort = bool(temperature is not None and temperature >= spec.thermal_abort_c)
            pause = duty_cycle_pause_seconds(
                temperature,
                target_c=spec.thermal_target_c,
                abort_c=spec.thermal_abort_c,
                nominal_seconds=spec.thermal_pause_seconds,
            )
            thermal_state = "cooling" if pause > 0 else "full-speed"
        if thermal_abort:
            if progress:
                progress(ProgressEvent(
                    "thermal_abort",
                    message=f"Thermal boundary reached ({temperature}°C); saving checkpoint",
                    step=step,
                    total_steps=spec.max_steps,
                    gpu_temperature_c=temperature,
                    thermal_state="thermal-abort",
                    initial_step=initial_step,
                ))
            thermal_abort = True
            break
        if pause:
            if pause >= 2.0 and progress:
                remaining = pause
                recovery_floor = spec.thermal_cruise_max_c if spec.thermal_cruise_max_c is not None else 60.0
                while remaining > 0:
                    chunk = min(0.5, remaining)
                    time.sleep(chunk)
                    remaining -= chunk
                    curr_temp = latest_temperature_c(telemetry.points)
                    if curr_temp is not None and curr_temp <= recovery_floor:
                        break
                    progress(ProgressEvent(
                        "step",
                        step=step,
                        total_steps=spec.max_steps,
                        elapsed_seconds=time.perf_counter() - training_started,
                        tokens_per_second=(tokens - initial_tokens) / max(1e-6, time.perf_counter() - training_started),
                        loss=loss_sum,
                        gpu_temperature_c=curr_temp,
                        vram_bytes=int(torch.cuda.memory_allocated()),
                        thermal_pause_seconds=remaining,
                        thermal_state="pit-stop-cooldown",
                        initial_step=initial_step,
                    ))
            else:
                time.sleep(pause)
            thermal_pause_seconds += pause
            temperature = latest_temperature_c(telemetry.points)
            if temperature is not None and temperature >= spec.thermal_abort_c:
                thermal_abort = True
                if progress:
                    progress(ProgressEvent(
                        "thermal_abort",
                        message=f"Thermal boundary reached ({temperature:.1f}°C); saving checkpoint",
                        step=step,
                        total_steps=spec.max_steps,
                        gpu_temperature_c=temperature,
                        thermal_state="thermal-abort",
                        initial_step=initial_step,
                    ))
                break
        session_tokens = tokens - initial_tokens
        if progress:
            progress(ProgressEvent(
                "step",
                step=step,
                total_steps=spec.max_steps,
                elapsed_seconds=time.perf_counter() - started,
                tokens_per_second=session_tokens / max(time.perf_counter() - started, 1e-6),
                loss=loss_sum,
                vram_bytes=torch.cuda.memory_allocated(),
                gpu_temperature_c=temperature,
                thermal_state=thermal_state,
                thermal_pause_seconds=pause,
                initial_step=initial_step,
            ))
        if step == spec.max_steps or step % evaluation_interval == 0:
            evaluation_started = time.perf_counter()
            nll = validation_nll()
            validation_seconds += time.perf_counter() - evaluation_started
            evaluations.append(evaluation_record(nll))
    training_loop_seconds = time.perf_counter() - training_started
    state = {
        "adapters": get_peft_model_state_dict(model),
        "optimizer": optimizer.state_dict(), "batcher": batcher.state_dict(),
        "step": step, "tokens": tokens, "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "termination_reason": "thermal_abort" if thermal_abort else "completed",
    }
    checkpoint_started = time.perf_counter()
    checkpoint_path = store.save(state)
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    measured = telemetry.stop()
    seconds = time.perf_counter() - started
    session_tokens = tokens - initial_tokens
    _atomic_json(
        run / "metrics.summary.json",
        {
            "state": "thermal_abort" if thermal_abort else "completed",
            "mode": "qlora", "step": step, "tokens": tokens, "seconds": seconds,
            "session_tokens": session_tokens,
            "tokens_per_second": session_tokens / seconds if seconds else 0.0, "evaluations": evaluations,
            "joules_per_token": (
                float(measured["gpu_board_energy_joules"]) / session_tokens
                if measured["gpu_board_energy_joules"] is not None and session_tokens else None
            ),
            "perplexity_monotonic": all(
                right["perplexity"] <= left["perplexity"]
                for left, right in zip(evaluations, evaluations[1:])
            ),
            "setup_seconds": setup_seconds,
            "validation_seconds": validation_seconds,
            "training_loop_seconds": training_loop_seconds,
            "update_compute_seconds": update_compute_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "training_loop_tokens_per_second": (
                session_tokens / training_loop_seconds if training_loop_seconds else None
            ),
            "update_compute_tokens_per_second": (
                session_tokens / update_compute_seconds if update_compute_seconds else None
            ),
            "base_checkpoint_parameter_count": base_checkpoint_parameter_count,
            "quantized_parameter_elements": quantized_parameter_elements,
            "trainable_parameter_count": trainable_parameter_count,
            "optimizer_backend": (
                "peft-loraplus-torch-adamw"
                if spec.stream.lora_plus_lr_ratio is not None
                else "torch-adamw-default"
            ),
            "optimizer_learning_rates": sorted({
                float(group["lr"]) for group in optimizer.param_groups
            }),
            "lora_plus_lr_ratio": spec.stream.lora_plus_lr_ratio,
            "activation_checkpointing": spec.activation_checkpointing,
            "lora_target_modules": spec.stream.lora_target_modules,
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "telemetry": {key: value for key, value in measured.items() if key != "points"},
            "layer_streaming": False,
            "thermal_abort": thermal_abort,
            "thermal_pause_seconds": thermal_pause_seconds,
            "limitation": "Standard optional QLoRA path; arbitrary-decoder layer streaming is not integrated.",
        },
    )
    return run


@torch.no_grad()
def evaluate_qlora(spec: TrainingSpec, run: str | Path, *, batches: int = 8) -> dict[str, float]:
    from peft import set_peft_model_state_dict

    model = _build_qlora_model(spec)
    state = AtomicCheckpointStore(run).load(map_location="cpu")
    set_peft_model_state_dict(model, state["adapters"])
    validation_spec = type(spec.data)(
        **{
            **asdict(spec.data),
            "path": spec.data.validation_path or spec.data.path,
            "validation_path": None,
        }
    )
    validation = MMapTokenBatcher(validation_spec, seed=spec.seed + 1, device="cuda")
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = validation.batch(1)
        losses.append(float(_shifted_causal_loss(model, x, y)))
    nll = sum(losses) / len(losses)
    return {"validation_nll": nll, "validation_perplexity": math.exp(nll)}


@torch.no_grad()
def generate_qlora(
    spec: TrainingSpec,
    run: str | Path,
    prompt: list[int],
    max_new_tokens: int = 32,
) -> list[int]:
    from peft import set_peft_model_state_dict

    if not prompt:
        raise ValueError("prompt must contain at least one token ID")
    model = _build_qlora_model(spec)
    state = AtomicCheckpointStore(run).load(map_location="cpu")
    set_peft_model_state_dict(model, state["adapters"])
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    window = prompt[-spec.model.context_length :]
    inputs = torch.tensor(window, device="cuda").unsqueeze(0)
    generated = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return [*prompt[:-len(window)], *generated[0].tolist()]


def benchmark_qlora_adapter(
    spec: TrainingSpec,
    run: str | Path,
    *,
    batches: int = 8,
    split: str = "validation",
) -> dict[str, object]:
    """Compare zero-initialized LoRA and a saved adapter on identical windows."""
    from peft import set_peft_model_state_dict

    if batches <= 0:
        raise ValueError("batches must be positive")
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    root = Path(run)
    store = AtomicCheckpointStore(root)
    checkpoint = store.resolve()
    telemetry = NVMLTelemetry(0.05)
    telemetry.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = _build_qlora_model(spec)
    benchmark_path = (
        spec.data.path
        if split == "train"
        else spec.data.validation_path or spec.data.path
    )
    validation_spec = type(spec.data)(
        **{
            **asdict(spec.data),
            "path": benchmark_path,
            "validation_path": None,
        }
    )

    @torch.no_grad()
    def measure() -> tuple[float, float]:
        batcher = MMapTokenBatcher(validation_spec, seed=spec.seed + 1, device="cuda")
        model.eval()
        torch.cuda.synchronize()
        phase_started = time.perf_counter()
        losses = []
        for _ in range(batches):
            x, y = batcher.batch(1)
            losses.append(float(_shifted_causal_loss(model, x, y)))
        torch.cuda.synchronize()
        return sum(losses) / len(losses), time.perf_counter() - phase_started

    baseline_nll, baseline_seconds = measure()
    state = store.load(map_location="cpu")
    set_peft_model_state_dict(model, state["adapters"])
    adapter_nll, adapter_seconds = measure()
    measured = telemetry.stop()
    total_seconds = time.perf_counter() - started
    result: dict[str, object] = {
        "schema_version": 1,
        "benchmark": "molt-qlora-adapter-quality",
        "run": str(root),
        "base_model": spec.base_model,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "batches_per_arm": batches,
        "split": split,
        "data_path": benchmark_path,
        "context_length": spec.data.context_length,
        "tokens_per_arm": batches * spec.data.context_length,
        "baseline": {
            "validation_nll": baseline_nll,
            "validation_perplexity": math.exp(baseline_nll),
            "seconds": baseline_seconds,
        },
        "adapter": {
            "validation_nll": adapter_nll,
            "validation_perplexity": math.exp(adapter_nll),
            "seconds": adapter_seconds,
        },
        "nll_improvement_percent": (1.0 - adapter_nll / baseline_nll) * 100.0,
        "quality_improved": adapter_nll < baseline_nll,
        "total_seconds_including_load": total_seconds,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "telemetry": {key: value for key, value in measured.items() if key != "points"},
        "workload_control": "Same model, selected data file, token windows, context, and batch count.",
    }
    output = (
        Path(spec.artifacts_dir).parent
        / "benchmarks"
        / f"qwen-qlora-{split}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    result["artifact_path"] = str(output)
    return result
