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

from molt_stream.core.contracts import ProgressEvent
from molt_stream.core.errors import CapabilityError
from molt_stream.core.specs import TrainingSpec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.experiments.store import AtomicCheckpointStore
from molt_stream.measurement.telemetry import NVMLTelemetry
from molt_stream.measurement.thermal import (
    build_thermal_controller,
    latest_telemetry_point,
    latest_temperature_c,
)


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
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
    telemetry = NVMLTelemetry(0.1)
    telemetry.start()
    started = time.perf_counter()
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
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=spec.stream.lora_rank,
            lora_alpha=int(spec.stream.lora_alpha),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=spec.learning_rate,
    )
    batcher = MMapTokenBatcher(spec.data, seed=spec.seed, device="cuda")
    validation_spec = type(spec.data)(
        **{**asdict(spec.data), "path": spec.data.validation_path or spec.data.path, "validation_path": None}
    )
    validation = MMapTokenBatcher(validation_spec, seed=spec.seed + 1, device="cuda")
    step = tokens = 0
    if resume:
        state = store.load(map_location="cpu")
        set_peft_model_state_dict(model, state["adapters"])
        optimizer.load_state_dict(state["optimizer"])
        batcher.load_state_dict(state["batcher"])
        step, tokens = int(state["step"]), int(state["tokens"])
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
            losses.append(float(model(input_ids=x, labels=y).loss))
        validation.load_state_dict(state)
        model.train()
        return sum(losses) / len(losses)

    initial = validation_nll()
    evaluations = [{"step": step, "nll": initial, "perplexity": math.exp(initial)}]
    thermal_abort = False
    thermal_pause_seconds = 0.0
    regulator = build_thermal_controller(spec)
    while step < spec.max_steps:
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(spec.gradient_accumulation):
            x, y = batcher.batch(spec.batch_size)
            loss = model(input_ids=x, labels=y).loss / spec.gradient_accumulation
            loss.backward()
            tokens += y.numel()
        optimizer.step()
        torch.cuda.synchronize()
        step += 1
        decision = regulator.update(
            latest_telemetry_point(telemetry.points),
            step_seconds=max(time.perf_counter() - step_started, 1e-6),
        ) if regulator is not None else None
        temperature = latest_temperature_c(telemetry.points)
        pause = decision.pause_seconds if decision is not None else 0.0
        thermal_abort = bool(decision.abort) if decision is not None else bool(
            temperature is not None and temperature >= spec.thermal_abort_c
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
            thermal_abort = True
            break
        if pause:
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
                    ))
                break
        if progress:
            progress(ProgressEvent(
                "step",
                step=step,
                total_steps=spec.max_steps,
                elapsed_seconds=time.perf_counter() - started,
                tokens_per_second=tokens / max(time.perf_counter() - started, 1e-6),
                loss=float(loss.detach()) * spec.gradient_accumulation,
                vram_bytes=torch.cuda.memory_allocated(),
                gpu_temperature_c=temperature,
                thermal_state=decision.phase if decision is not None else "full-speed",
                thermal_pause_seconds=pause,
            ))
        if step == spec.max_steps or step % max(1, spec.max_steps // 4) == 0:
            nll = validation_nll()
            evaluations.append({"step": step, "nll": nll, "perplexity": math.exp(nll)})
    state = {
        "adapters": get_peft_model_state_dict(model),
        "optimizer": optimizer.state_dict(), "batcher": batcher.state_dict(),
        "step": step, "tokens": tokens, "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "termination_reason": "thermal_abort" if thermal_abort else "completed",
    }
    store.save(state)
    measured = telemetry.stop()
    seconds = time.perf_counter() - started
    _atomic_json(
        run / "metrics.summary.json",
        {
            "state": "thermal_abort" if thermal_abort else "completed",
            "mode": "qlora", "step": step, "tokens": tokens, "seconds": seconds,
            "tokens_per_second": tokens / seconds, "evaluations": evaluations,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "telemetry": {key: value for key, value in measured.items() if key != "points"},
            "layer_streaming": False,
            "thermal_abort": thermal_abort,
            "thermal_pause_seconds": thermal_pause_seconds,
            "limitation": "Standard optional QLoRA path; arbitrary-decoder layer streaming is not integrated.",
        },
    )
    return run
