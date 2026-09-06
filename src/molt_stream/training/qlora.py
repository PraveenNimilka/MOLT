from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
from torch.nn import functional as F

from molt_stream.core.contracts import ProgressEvent
from molt_stream.core.errors import CapabilityError, IntegrityError
from molt_stream.core.specs import TrainingSpec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.experiments.store import AtomicCheckpointStore, sha256
from molt_stream.kernels.partitioned_loss import exact_partitioned_linear_cross_entropy
from molt_stream.measurement.telemetry import (
    NVMLTelemetry,
    integrate_board_energy,
    manage_power_limit,
)
from molt_stream.measurement.step_windows import step_window_rates
from molt_stream.measurement.update_accounting import UpdateAccounting
from molt_stream.measurement.thermal import (
    build_thermal_controller,
    cooling_pause,
    duty_cycle_pause_seconds,
    latest_telemetry_point,
    latest_temperature_c,
    passive_cooldown,
    postrun_thermal_violation,
    wait_for_thermal_headroom,
)
from molt_stream.training.update_transaction import UpdateTransaction
from molt_stream.training.evaluation_guard import EvaluationThermalStop, guarded_decoder_evaluation
from molt_stream.training.layer_thermal_guard import (
    TrainingThermalStop,
    guarded_decoder_training,
)
from molt_stream.training.activation_offload import activation_offload_context

_FROZEN_HEAD_COMPUTE_CACHE = "_molt_frozen_head_bf16"


def _activation_storage_context(spec: TrainingSpec):
    if spec.qlora_activation_compression_bits == 4:
        from molt_stream.methods.activation_compression import CompressedSavedActivations

        return CompressedSavedActivations(
            minimum_bytes=spec.qlora_activation_compression_minimum_bytes
        )
    return activation_offload_context(spec.qlora_activation_offload)


def _shifted_causal_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    autocast: bool = False,
    chunk_size: int | None = None,
    precompute_head_gradient: bool = False,
    loss_backend: str = "analytical",
) -> torch.Tensor:
    """Compute loss against MOLT's already shifted targets exactly once."""
    with torch.autocast(inputs.device.type, dtype=torch.bfloat16, enabled=autocast):
        if chunk_size is not None:
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            head = base.get_output_embeddings()
            if (getattr(base.config, "model_type", None) != "qwen2"
                    or type(head) is not torch.nn.Linear or head.bias is not None
                    or head.weight.requires_grad):
                raise CapabilityError("Partitioned QLoRA loss requires Qwen2 with a frozen, bias-free Linear head")
            hidden = base.model(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state
            projection_weight = (
                getattr(model, _FROZEN_HEAD_COMPUTE_CACHE)
                if precompute_head_gradient
                and autocast
                and hasattr(model, _FROZEN_HEAD_COMPUTE_CACHE)
                else head.weight
            )
            if not torch.is_grad_enabled() and hidden.dtype != projection_weight.dtype:
                hidden = hidden.to(projection_weight.dtype)
            return exact_partitioned_linear_cross_entropy(hidden, projection_weight, targets, chunk_size,
                precompute_frozen_gradient=precompute_head_gradient, backend=loss_backend)
        logits = model(input_ids=inputs, use_cache=False).logits
    # AMP changes GEMM precision, not the precision of the loss reduction.
    loss_logits = logits.float() if autocast else logits
    return F.cross_entropy(loss_logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


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
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_options: dict[str, object] = {}
    if spec.qlora_fused_optimizer:
        if not parameters or any(p.device.type != "cuda" or p.dtype != torch.float32 for p in parameters):
            raise CapabilityError("Fused QLoRA AdamW requires CUDA FP32 trainable parameters")
        optimizer_options["fused"] = True
    if spec.stream.lora_plus_lr_ratio is not None:
        from peft.optimizers import create_loraplus_optimizer

        return create_loraplus_optimizer(
            model=model,
            optimizer_cls=torch.optim.AdamW,
            lr=spec.learning_rate,
            loraplus_lr_ratio=spec.stream.lora_plus_lr_ratio,
            loraplus_weight_decay=0.01,
            **optimizer_options,
        )
    return torch.optim.AdamW(
        parameters,
        lr=spec.learning_rate,
        **optimizer_options,
    )


def _validate_loading_info(info: dict[str, object]) -> None:
    """Refuse partial base-model loads before attaching or updating adapters.

    Transformers already handles its architecture's documented tied-weight
    exclusions. Do not hide remaining mismatches with a MOLT allowlist.
    """
    failures = {
        key: info[key]
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if info.get(key)
    }
    if failures:
        raise IntegrityError(
            "Base checkpoint does not match its configured architecture. "
            "Training was refused; restore the original model configuration "
            "and compatible implementation, not just a different model_type. "
            f"Loading details: {failures}"
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
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        spec.base_model,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation=spec.qlora_attention_backend,
        output_loading_info=True,
    )
    _validate_loading_info(loading_info)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=spec.activation_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if spec.qlora_restore_tied_embedding_bf16:
        input_embedding = model.get_input_embeddings()
        output_embedding = model.get_output_embeddings()
        if (
            input_embedding is None
            or output_embedding is None
            or input_embedding.weight.requires_grad
            or output_embedding.weight.requires_grad
            or input_embedding.weight.data_ptr() != output_embedding.weight.data_ptr()
        ):
            raise CapabilityError(
                "BF16 embedding restoration requires one frozen, tied input/output weight"
            )
        input_embedding.weight.data = input_embedding.weight.data.to(torch.bfloat16)
    if spec.qlora_frozen_rmsnorm_bf16:
        from molt_stream.kernels.frozen_rmsnorm import enable_qwen2_frozen_rmsnorm_bf16

        enable_qwen2_frozen_rmsnorm_bf16(model)
    if spec.qlora_checkpoint_stride != 1:
        from molt_stream.training.selective_checkpoint import configure_checkpoint_stride

        configure_checkpoint_stride(model, spec.qlora_checkpoint_stride)
    lora_kwargs: dict[str, object] = {}
    target_modules = _resolve_lora_target_modules(spec.stream.lora_target_modules)
    if spec.qlora_train_last_layers is not None and spec.qlora_full_warmup_steps == 0:
        from molt_stream.training.truncated_backprop import QWEN2_LINEAR_MODULES, upper_layer_indices

        if getattr(model.config, "model_type", None) != "qwen2":
            raise CapabilityError("Upper-layer QLoRA currently requires Qwen2")
        if target_modules == "all-linear":
            target_modules = QWEN2_LINEAR_MODULES
        lora_kwargs = {
            "layers_to_transform": upper_layer_indices(
                int(model.config.num_hidden_layers), spec.qlora_train_last_layers
            ),
            "layers_pattern": "layers",
        }
    model = get_peft_model(
        model,
        LoraConfig(
            r=spec.stream.lora_rank,
            lora_alpha=int(spec.stream.lora_alpha),
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            **lora_kwargs,
        ),
    )
    if spec.qlora_scheduled_nf4_down_projection:
        from molt_stream.methods.nf4_lora import enable_scheduled_nf4_lora

        enable_scheduled_nf4_lora(model, module_suffixes=("down_proj",))
    if spec.qlora_bf16_adapter_shadows:
        from molt_stream.methods.lora_shadow import enable_bf16_lora_shadows

        enable_bf16_lora_shadows(model)
    if spec.qlora_train_last_layers is not None and spec.qlora_full_warmup_steps == 0:
        from molt_stream.training.truncated_backprop import truncate_before_decoder_layer

        truncate_before_decoder_layer(
            model, int(model.config.num_hidden_layers) - spec.qlora_train_last_layers
        )
    if (
        spec.qlora_autocast
        and spec.qlora_precompute_head_gradient
        and spec.qlora_cache_frozen_head
        and spec.qlora_loss_chunk_size is not None
    ):
        head = model.get_base_model().get_output_embeddings()
        if type(head) is not torch.nn.Linear or head.weight.requires_grad:
            raise CapabilityError("Frozen-head compute cache requires a frozen Linear output head")
        model.register_buffer(
            _FROZEN_HEAD_COMPUTE_CACHE,
            head.weight.detach().to(dtype=torch.bfloat16),
            persistent=False,
        )
    return model


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
        if resume_state.get("termination_reason") == "completed" or saved_step > spec.max_steps:
            raise ValueError(f"run is already complete at step {saved_step}")
        # Preserve each session's measurements before a resumed session writes
        # new summaries. A final-step thermal stop may still need validation.
        archive = run / "sessions" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        archive.mkdir(parents=True, exist_ok=False)
        for name in ("metrics.summary.json", "telemetry.samples.json", "step.intervals.json"):
            if (run / name).is_file():
                shutil.copy2(run / name, archive / name)
    telemetry = NVMLTelemetry(0.1)
    telemetry.start()
    try:
        power_limit_status = (
            manage_power_limit(spec.power_limit_watts, apply=False)
            if spec.power_limit_watts is not None
            else None
        )
    except BaseException:
        telemetry.stop()
        raise
    if power_limit_status is not None and not power_limit_status["verified"]:
        telemetry.stop()
        raise CapabilityError(
            f"Requested {spec.power_limit_watts:g} W is not enforced. Apply it explicitly "
            "from an Administrator terminal; MOLT will never silently continue an invalid "
            "power-limit experiment."
        )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        model = _build_qlora_model(spec)
    except BaseException:
        telemetry.stop()
        raise
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

    upper_layers_activated = bool(
        spec.qlora_train_last_layers is not None and spec.qlora_full_warmup_steps == 0
    )
    if (spec.qlora_train_last_layers is not None
            and step >= spec.qlora_full_warmup_steps and not upper_layers_activated):
        from molt_stream.training.truncated_backprop import activate_upper_layer_training
        activate_upper_layer_training(model, spec.qlora_train_last_layers)
        upper_layers_activated = True

    thermal_pause_seconds = 0.0
    thermal_stop_reason: str | None = None
    microbatch_peak_temperature: float | None = None
    loss_sum = 0.0
    recovery_count = 0
    recovery_checkpoint_seconds = 0.0

    def fresh_temperature():
        nonlocal microbatch_peak_temperature
        point = telemetry.thermal_point()
        if point is not None and point.gpu_temperature_c is not None and math.isfinite(point.gpu_temperature_c):
            microbatch_peak_temperature = max(microbatch_peak_temperature or point.gpu_temperature_c,
                                             point.gpu_temperature_c)
        return point

    def committed_state(reason: str) -> dict[str, object]:
        return {
            "adapters": get_peft_model_state_dict(model),
            "optimizer": optimizer.state_dict(), "batcher": batcher.state_dict(),
            "step": step, "tokens": tokens, "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "termination_reason": reason,
        }

    def recover_thermal() -> bool:
        nonlocal recovery_count, recovery_checkpoint_seconds
        nonlocal thermal_pause_seconds, thermal_stop_reason
        if (spec.thermal_recovery_mode != "cool-and-continue"
                or recovery_count >= spec.thermal_max_recoveries):
            return False
        checkpoint_started = time.perf_counter()
        store.save(committed_state("thermal_recovery"))
        recovery_checkpoint_seconds += time.perf_counter() - checkpoint_started
        decision = passive_cooldown(
            fresh_temperature,
            recovery_c=spec.thermal_recovery_c or spec.thermal_target_c,
            notify=(lambda paused, current: progress(ProgressEvent(
                "step", step=step, total_steps=spec.max_steps, initial_step=initial_step,
                elapsed_seconds=time.perf_counter() - started, loss=loss_sum,
                gpu_temperature_c=current, thermal_pause_seconds=paused,
                thermal_state="passive-recovery",
            ))) if progress else None,
        )
        recovery_count += 1
        thermal_pause_seconds += decision.pause_seconds
        thermal_stop_reason = decision.stop_reason
        return decision.stop_reason is None

    def microbatch_gate() -> bool:
        nonlocal thermal_pause_seconds, thermal_stop_reason
        guard_c = spec.thermal_microbatch_guard_c or spec.thermal_target_c
        decision = wait_for_thermal_headroom(
            fresh_temperature, target_c=guard_c, abort_c=spec.thermal_abort_c,
            notify=(lambda paused, current: progress(ProgressEvent(
                "step", step=step, total_steps=spec.max_steps, initial_step=initial_step,
                elapsed_seconds=time.perf_counter() - started, loss=loss_sum,
                gpu_temperature_c=current, thermal_pause_seconds=paused,
                thermal_state="microbatch-cooling",
            ))) if progress else None,
        )
        thermal_pause_seconds += decision.pause_seconds
        thermal_stop_reason = decision.stop_reason
        return decision.stop_reason is None

    def intra_step_checkpoint() -> bool:
        nonlocal thermal_pause_seconds
        if not microbatch_gate():
            return False
        pause = spec.qlora_intra_step_pause_ms / 1000.0
        if pause:
            time.sleep(pause)
            thermal_pause_seconds += pause
        return True

    @torch.no_grad()
    def validation_nll() -> float | None:
        model.eval()
        state = validation.state_dict()
        losses = []
        try:
            for _ in range(spec.qlora_validation_batches):
                if not microbatch_gate():
                    return None
                x, y = validation.batch(1)
                guard = guarded_decoder_evaluation(model, microbatch_gate) if spec.qlora_validation_layer_guard else nullcontext()
                with guard:
                    losses.append(float(_shifted_causal_loss(model, x, y, chunk_size=spec.qlora_loss_chunk_size)))
                if not microbatch_gate():
                    return None
        except EvaluationThermalStop:
            return None
        finally:
            validation.load_state_dict(state)
            model.train()
        return sum(losses) / len(losses)

    validation_started = time.perf_counter()
    initial = validation_nll()
    while initial is None and recover_thermal():
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

    evaluations = [evaluation_record(initial)] if initial is not None else []
    thermal_abort = initial is None
    update_accounting = UpdateAccounting()
    regulator = build_thermal_controller(spec)
    evaluation_interval = spec.evaluation_interval or max(1, spec.max_steps // 4)
    training_started = time.perf_counter()
    step_intervals: list[float] = []
    discarded_tokens = 0
    activation_compression_tensors = 0
    activation_compression_original_bytes = 0
    activation_compression_stored_bytes = 0
    activation_compression_sizes: dict[int, int] = {}
    while step < spec.max_steps and not thermal_abort:
        if (spec.qlora_train_last_layers is not None
                and step >= spec.qlora_full_warmup_steps and not upper_layers_activated):
            from molt_stream.training.truncated_backprop import activate_upper_layer_training
            activate_upper_layer_training(model, spec.qlora_train_last_layers)
            upper_layers_activated = True
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        transaction = UpdateTransaction.capture(batcher, cuda=True)
        loss_sum = 0.0
        pending_tokens = 0
        pause_before_update = thermal_pause_seconds

        for _ in range(spec.gradient_accumulation):
            if not microbatch_gate():
                break
            x, y = batcher.batch(spec.batch_size)
            pending_tokens += y.numel()
            layer_guard = (
                guarded_decoder_training(
                    model,
                    boundary_after_layers=spec.qlora_intra_step_boundary_layer,
                    checkpoint=intra_step_checkpoint,
                )
                if spec.qlora_intra_step_boundary_layer is not None
                else nullcontext()
            )
            micro_loss = None
            try:
                with layer_guard:
                    activation_storage = _activation_storage_context(spec)
                    with activation_storage:
                        micro_loss = _shifted_causal_loss(
                            model,
                            x,
                            y,
                            autocast=spec.qlora_autocast,
                            chunk_size=spec.qlora_loss_chunk_size,
                            precompute_head_gradient=spec.qlora_precompute_head_gradient,
                            loss_backend=spec.qlora_loss_backend,
                        ) / spec.gradient_accumulation
                    activation_compression_tensors += int(
                        getattr(activation_storage, "compressed_tensor_count", 0)
                    )
                    activation_compression_original_bytes += int(
                        getattr(activation_storage, "original_bytes", 0)
                    )
                    activation_compression_stored_bytes += int(
                        getattr(activation_storage, "stored_bytes", 0)
                    )
                    for size, count in getattr(activation_storage, "size_histogram", {}).items():
                        activation_compression_sizes[size] = (
                            activation_compression_sizes.get(size, 0) + count
                        )
                    torch.cuda.synchronize()
                    if not microbatch_gate():
                        del micro_loss
                        micro_loss = None
                        break
                    micro_loss.backward()
            except TrainingThermalStop:
                if micro_loss is not None:
                    del micro_loss
                break
            assert micro_loss is not None
            loss_sum += float(micro_loss.detach())
            del micro_loss
            if not microbatch_gate():
                break
        if thermal_stop_reason is not None:
            transaction.rollback(batcher, optimizer)
            discarded_tokens += pending_tokens
            update_accounting.record(
                elapsed_seconds=time.perf_counter() - step_started,
                pause_seconds=thermal_pause_seconds - pause_before_update,
                tokens=pending_tokens, committed=False,
            )
            if recover_thermal():
                continue
            thermal_abort = True
            if progress:
                progress(ProgressEvent("thermal_abort", message=thermal_stop_reason,
                    step=step, total_steps=spec.max_steps, initial_step=initial_step,
                    thermal_state="thermal-abort", gpu_temperature_c=latest_temperature_c(telemetry.points)))
            break
        optimizer.step()
        if spec.qlora_bf16_adapter_shadows:
            from molt_stream.methods.lora_shadow import invalidate_bf16_lora_shadows

            invalidate_bf16_lora_shadows(model)
        tokens += pending_tokens
        torch.cuda.synchronize()
        update_accounting.record(
            elapsed_seconds=time.perf_counter() - step_started,
            pause_seconds=thermal_pause_seconds - pause_before_update,
            tokens=pending_tokens, committed=True,
        )
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
            def notify_cooling(remaining: float, current: float | None) -> None:
                if progress:
                    progress(ProgressEvent(
                        "step",
                        step=step,
                        total_steps=spec.max_steps,
                        elapsed_seconds=time.perf_counter() - training_started,
                        tokens_per_second=(tokens - initial_tokens) / max(1e-6, time.perf_counter() - training_started),
                        loss=loss_sum,
                        gpu_temperature_c=current,
                        vram_bytes=int(torch.cuda.memory_allocated()),
                        thermal_pause_seconds=remaining,
                        thermal_state="pit-stop-cooldown",
                        initial_step=initial_step,
                    ))

            thermal_pause_seconds += cooling_pause(
                pause,
                lambda: latest_temperature_c(telemetry.points),
                recovery_c=spec.thermal_cruise_max_c if spec.thermal_cruise_max_c is not None else 60.0,
                abort_c=spec.thermal_abort_c,
                notify=notify_cooling if progress else None,
            )
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
            while nll is None and recover_thermal():
                nll = validation_nll()
            validation_seconds += time.perf_counter() - evaluation_started
            if nll is None:
                thermal_abort = True
                break
            evaluations.append(evaluation_record(nll))
        step_intervals.append(time.perf_counter() - step_started)
    training_loop_seconds = time.perf_counter() - training_started
    state = committed_state("thermal_abort" if thermal_abort else "completed")
    checkpoint_started = time.perf_counter()
    checkpoint_path = store.save(state)
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    measured = telemetry.stop()
    if microbatch_peak_temperature is not None:
        measured["peak_gpu_temperature_c"] = max(
            measured.get("peak_gpu_temperature_c") or microbatch_peak_temperature, microbatch_peak_temperature)
    measured_peak_temperature = measured.get("peak_gpu_temperature_c")
    delayed_thermal_violation = postrun_thermal_violation(
        None if measured_peak_temperature is None else float(measured_peak_temperature),
        abort_c=spec.thermal_abort_c,
    )
    if delayed_thermal_violation and not thermal_abort:
        # NVML temperature can peak after the last synchronous guard because
        # heat reaches the sensor with delay. Never publish such a session as
        # completed: preserve its committed state as resumable thermal stop.
        thermal_abort = True
        thermal_stop_reason = (
            f"delayed thermal peak reached {float(measured_peak_temperature):.1f} C "
            f"at or above the {spec.thermal_abort_c:.1f} C boundary"
        )
        corrective_checkpoint_started = time.perf_counter()
        checkpoint_path = store.save(committed_state("thermal_abort"))
        checkpoint_seconds += time.perf_counter() - corrective_checkpoint_started
    _atomic_json(run / "telemetry.samples.json", {"points": measured["points"], "errors": measured["errors"]})
    _atomic_json(run / "step.intervals.json", {"seconds": step_intervals,
                 "tokens_per_update": spec.batch_size * spec.gradient_accumulation * spec.data.context_length,
                 "includes": "training, evaluation and thermal pacing; completed steps only"})
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
            ) if len(evaluations) >= 2 else None,
            "setup_seconds": setup_seconds,
            "validation_seconds": validation_seconds,
            "training_loop_seconds": training_loop_seconds,
            "step_window_rates": step_window_rates(
                step_intervals, spec.batch_size * spec.gradient_accumulation * spec.data.context_length
            ),
            **update_accounting.summary(),
            "checkpoint_seconds": checkpoint_seconds + recovery_checkpoint_seconds,
            "thermal_recoveries": recovery_count,
            "training_loop_tokens_per_second": (
                session_tokens / training_loop_seconds if training_loop_seconds else None
            ),
            "base_checkpoint_parameter_count": base_checkpoint_parameter_count,
            "quantized_parameter_elements": quantized_parameter_elements,
            "trainable_parameter_count": trainable_parameter_count,
            "active_trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "optimizer_backend": (
                "peft-loraplus-torch-adamw"
                if spec.stream.lora_plus_lr_ratio is not None
                else "torch-adamw-fused" if spec.qlora_fused_optimizer else "torch-adamw-default"
            ),
            "qlora_autocast": spec.qlora_autocast,
            "qlora_loss_chunk_size": spec.qlora_loss_chunk_size,
            "qlora_loss_backend": spec.qlora_loss_backend,
            "qlora_attention_backend": spec.qlora_attention_backend,
            "qlora_activation_offload": spec.qlora_activation_offload,
            "qlora_activation_compression_bits": spec.qlora_activation_compression_bits,
            "qlora_activation_compression_minimum_bytes": (
                spec.qlora_activation_compression_minimum_bytes
            ),
            "activation_compression": {
                "tensor_count": activation_compression_tensors,
                "original_bytes": activation_compression_original_bytes,
                "stored_bytes": activation_compression_stored_bytes,
                "size_histogram": {
                    str(size): activation_compression_sizes[size]
                    for size in sorted(activation_compression_sizes)
                },
            } if spec.qlora_activation_compression_bits is not None else None,
            "frozen_head_compute_cache_bytes": (
                getattr(model, _FROZEN_HEAD_COMPUTE_CACHE).numel()
                * getattr(model, _FROZEN_HEAD_COMPUTE_CACHE).element_size()
                if hasattr(model, _FROZEN_HEAD_COMPUTE_CACHE) else 0
            ),
            "frozen_head_compute_cache_unique_bytes": (
                0
                if hasattr(model, _FROZEN_HEAD_COMPUTE_CACHE)
                and getattr(model, _FROZEN_HEAD_COMPUTE_CACHE).data_ptr()
                == model.get_base_model().get_output_embeddings().weight.data_ptr()
                else (
                    getattr(model, _FROZEN_HEAD_COMPUTE_CACHE).numel()
                    * getattr(model, _FROZEN_HEAD_COMPUTE_CACHE).element_size()
                    if hasattr(model, _FROZEN_HEAD_COMPUTE_CACHE) else 0
                )
            ),
            "evaluation_precision": (
                "autocast-disabled; frozen tied embedding/head restored to BF16"
                if spec.qlora_restore_tied_embedding_bf16
                else "autocast-disabled; NF4 base compute remains BF16"
            ),
            "qlora_fused_optimizer": spec.qlora_fused_optimizer,
            "optimizer_learning_rates": sorted({
                float(group["lr"]) for group in optimizer.param_groups
            }),
            "lora_plus_lr_ratio": spec.stream.lora_plus_lr_ratio,
            "activation_checkpointing": spec.activation_checkpointing,
            "qlora_checkpoint_stride": spec.qlora_checkpoint_stride,
            "qlora_precompute_head_gradient": spec.qlora_precompute_head_gradient,
            "qlora_cache_frozen_head": spec.qlora_cache_frozen_head,
            "qlora_restore_tied_embedding_bf16": spec.qlora_restore_tied_embedding_bf16,
            "qlora_frozen_rmsnorm_bf16": spec.qlora_frozen_rmsnorm_bf16,
            "qlora_scheduled_nf4_down_projection": (
                spec.qlora_scheduled_nf4_down_projection
            ),
            "qlora_bf16_adapter_shadows": spec.qlora_bf16_adapter_shadows,
            "qlora_validation_layer_guard": spec.qlora_validation_layer_guard,
            "qlora_intra_step_boundary_layer": spec.qlora_intra_step_boundary_layer,
            "qlora_intra_step_pause_ms": spec.qlora_intra_step_pause_ms,
            "qlora_train_last_layers": spec.qlora_train_last_layers,
            "qlora_full_warmup_steps": spec.qlora_full_warmup_steps,
            "qlora_validation_batches": spec.qlora_validation_batches,
            "lora_target_modules": spec.stream.lora_target_modules,
            "checkpoint": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "telemetry": {key: value for key, value in measured.items() if key != "points"},
            "power_limit": power_limit_status,
            "layer_streaming": False,
            "thermal_abort": thermal_abort,
            "thermal_stop_reason": thermal_stop_reason,
            "postrun_thermal_violation": delayed_thermal_violation,
            "discarded_uncommitted_tokens": discarded_tokens,
            "thermal_pause_seconds": thermal_pause_seconds,
            "thermal_microbatch_guard_c": (
                spec.thermal_microbatch_guard_c or spec.thermal_target_c
            ),
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
        losses.append(float(_shifted_causal_loss(model, x, y, chunk_size=spec.qlora_loss_chunk_size)))
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
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=spec.qlora_autocast):
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
            losses.append(float(_shifted_causal_loss(model, x, y, chunk_size=spec.qlora_loss_chunk_size)))
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
