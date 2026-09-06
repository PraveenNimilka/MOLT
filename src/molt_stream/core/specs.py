from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any


_ENVIRONMENT_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand_config_path(value: str, field_name: str) -> str:
    missing = sorted({
        name
        for match in _ENVIRONMENT_VARIABLE.finditer(value)
        if (name := (match.group(1) or match.group(2))) not in os.environ
    })
    if missing:
        raise ValueError(
            f"{field_name} references unset environment variable(s): " + ", ".join(missing)
        )
    return os.path.expandvars(value)


class TrainingMode(StrEnum):
    PRETRAIN = "pretrain"
    QLORA = "qlora"


@dataclass(frozen=True)
class StreamSpec:
    device: str = "cuda"
    compute_dtype: str = "bfloat16"
    quant_block_size: int = 64
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_target_modules: str = "all-linear"
    lora_plus_lr_ratio: float | None = None
    double_buffer: bool = True
    bundle_size: int = 4
    cuda_graphs: bool = False
    pin_host_memory: bool = True

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.compute_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported compute dtype")
        if self.quant_block_size < 16 or self.quant_block_size % 2:
            raise ValueError("quant_block_size must be even and >=16")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if self.lora_plus_lr_ratio is not None and self.lora_plus_lr_ratio < 1:
            raise ValueError("lora_plus_lr_ratio must be >= 1 when provided")
        target_modules = [
            name.strip() for name in self.lora_target_modules.split(",") if name.strip()
        ]
        if not target_modules:
            raise ValueError("lora_target_modules must name at least one module")
        if "all-linear" in target_modules and target_modules != ["all-linear"]:
            raise ValueError("all-linear cannot be combined with explicit LoRA target modules")
        if self.bundle_size <= 0:
            raise ValueError("bundle_size must be positive")


@dataclass(frozen=True)
class DataSpec:
    path: str
    context_length: int
    validation_path: str | None = None
    storage_dtype: str = "int32"
    sequential: bool = True
    packing: str = "contiguous"

    def validate(self) -> None:
        if not Path(self.path).is_file():
            raise FileNotFoundError(self.path)
        if self.validation_path is not None and not Path(self.validation_path).is_file():
            raise FileNotFoundError(self.validation_path)
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.storage_dtype not in {"uint8", "int32"}:
            raise ValueError("storage_dtype must be uint8 or int32")
        if self.packing not in {"contiguous", "indexed"}:
            raise ValueError("packing must be contiguous or indexed")


@dataclass(frozen=True)
class ModelSpec:
    vocab_size: int = 256
    context_length: int = 128
    layers: int = 6
    width: int = 512
    heads: int = 8
    hidden_width: int = 1536

    def validate(self) -> None:
        if min(self.vocab_size, self.context_length, self.layers, self.width, self.heads, self.hidden_width) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


@dataclass(frozen=True)
class TrainingSpec:
    mode: TrainingMode
    data: DataSpec
    model: ModelSpec = ModelSpec()
    stream: StreamSpec = StreamSpec()
    batch_size: int = 1
    gradient_accumulation: int = 1
    max_steps: int = 10
    evaluation_interval: int | None = None
    learning_rate: float = 2e-4
    seed: int = 1337
    artifacts_dir: str = "artifacts/molt-stream"
    base_model: str | None = None
    execution_backend: str = "eager"
    activation_checkpointing: bool = True
    qlora_fused_optimizer: bool = False
    qlora_autocast: bool = False
    qlora_loss_chunk_size: int | None = None
    qlora_loss_backend: str = "analytical"
    qlora_attention_backend: str = "sdpa"
    qlora_activation_offload: bool = False
    qlora_activation_compression_bits: int | None = None
    qlora_activation_compression_minimum_bytes: int = 1 << 20
    qlora_checkpoint_stride: int = 1
    qlora_precompute_head_gradient: bool = False
    qlora_cache_frozen_head: bool = True
    qlora_restore_tied_embedding_bf16: bool = False
    qlora_frozen_rmsnorm_bf16: bool = False
    qlora_scheduled_nf4_down_projection: bool = False
    qlora_bf16_adapter_shadows: bool = False
    qlora_validation_layer_guard: bool = False
    qlora_intra_step_boundary_layer: int | None = None
    qlora_intra_step_pause_ms: float = 0.0
    qlora_train_last_layers: int | None = None
    qlora_full_warmup_steps: int = 0
    qlora_validation_batches: int = 4
    thermal_target_c: float = 60.0
    thermal_abort_c: float = 72.0
    thermal_microbatch_guard_c: float | None = None
    thermal_pause_seconds: float = 0.04
    thermal_control_mode: str = "reactive"
    thermal_lookahead_seconds: float = 1.5
    thermal_max_pause_seconds: float = 0.25
    thermal_initial_pause_seconds: float = 0.0
    thermal_stability_band_c: float = 1.5
    thermal_power_target_watts: float | None = None
    thermal_cruise_max_c: float | None = None
    min_pause_ms: float = 10.0
    max_pause_ms: float = 25.0
    thermal_protective_pause_ms: float = 100.0
    thermal_protective_hysteresis_c: float = 2.0
    thermal_steady_pause_ms: float = 40.0
    power_limit_watts: float | None = None
    thermal_recovery_mode: str = "stop"
    thermal_recovery_c: float | None = None
    thermal_max_recoveries: int = 3

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.stream.validate()
        if type(self.qlora_validation_batches) is not int or self.qlora_validation_batches < 1:
            raise ValueError("qlora_validation_batches must be a positive integer")
        if self.qlora_validation_batches != 4 and self.mode != TrainingMode.QLORA:
            raise ValueError("qlora_validation_batches requires QLoRA")
        if type(self.qlora_full_warmup_steps) is not int or self.qlora_full_warmup_steps < 0:
            raise ValueError("qlora_full_warmup_steps must be a non-negative integer")
        if self.qlora_full_warmup_steps and self.qlora_train_last_layers is None:
            raise ValueError("qlora_full_warmup_steps requires qlora_train_last_layers")
        if self.qlora_full_warmup_steps > self.max_steps:
            raise ValueError("qlora_full_warmup_steps cannot exceed max_steps")
        if self.qlora_train_last_layers is not None and (
            type(self.qlora_train_last_layers) is not int or self.qlora_train_last_layers < 1
        ):
            raise ValueError("qlora_train_last_layers must be a positive integer")
        if self.qlora_train_last_layers is not None and self.mode != TrainingMode.QLORA:
            raise ValueError("qlora_train_last_layers requires QLoRA")
        if type(self.qlora_validation_layer_guard) is not bool:
            raise ValueError("qlora_validation_layer_guard must be a boolean")
        if self.qlora_validation_layer_guard and self.mode != TrainingMode.QLORA:
            raise ValueError("qlora_validation_layer_guard requires QLoRA")
        if self.qlora_intra_step_boundary_layer is not None:
            if (
                type(self.qlora_intra_step_boundary_layer) is not int
                or self.qlora_intra_step_boundary_layer < 1
            ):
                raise ValueError("qlora_intra_step_boundary_layer must be a positive integer")
            if self.mode != TrainingMode.QLORA or self.stream.device != "cuda":
                raise ValueError("intra-step layer thermal guarding requires QLoRA on CUDA")
        if not 0.0 <= self.qlora_intra_step_pause_ms <= 100.0:
            raise ValueError("qlora_intra_step_pause_ms must be between 0 and 100")
        if self.qlora_intra_step_pause_ms and self.qlora_intra_step_boundary_layer is None:
            raise ValueError("qlora_intra_step_pause_ms requires a boundary layer")
        if type(self.qlora_precompute_head_gradient) is not bool:
            raise ValueError("qlora_precompute_head_gradient must be a boolean")
        if self.qlora_precompute_head_gradient and (
            self.mode != TrainingMode.QLORA or self.qlora_loss_chunk_size is None
        ):
            raise ValueError("qlora_precompute_head_gradient requires partitioned QLoRA loss")
        if type(self.qlora_cache_frozen_head) is not bool:
            raise ValueError("qlora_cache_frozen_head must be a boolean")
        if type(self.qlora_restore_tied_embedding_bf16) is not bool:
            raise ValueError("qlora_restore_tied_embedding_bf16 must be a boolean")
        if self.qlora_restore_tied_embedding_bf16 and (
            self.mode != TrainingMode.QLORA or not self.qlora_autocast
        ):
            raise ValueError("BF16 tied embeddings require autocast QLoRA")
        if self.qlora_frozen_rmsnorm_bf16 and (
            self.mode != TrainingMode.QLORA or not self.qlora_autocast
        ):
            raise ValueError("BF16 frozen RMSNorm requires autocast QLoRA")
        if self.qlora_scheduled_nf4_down_projection and (
            self.mode != TrainingMode.QLORA
            or self.stream.device != "cuda"
            or not self.qlora_autocast
        ):
            raise ValueError(
                "scheduled NF4 down projection requires autocast QLoRA on CUDA"
            )
        if self.qlora_bf16_adapter_shadows and (
            self.mode != TrainingMode.QLORA
            or self.stream.device != "cuda"
            or not self.qlora_autocast
        ):
            raise ValueError("BF16 adapter shadows require autocast QLoRA on CUDA")
        if self.qlora_bf16_adapter_shadows and self.qlora_scheduled_nf4_down_projection:
            raise ValueError(
                "BF16 adapter shadows and scheduled NF4 down projection are mutually exclusive"
            )
        if type(self.qlora_checkpoint_stride) is not int or self.qlora_checkpoint_stride < 1:
            raise ValueError("qlora_checkpoint_stride must be a positive integer")
        if self.qlora_checkpoint_stride != 1 and (
            self.mode != TrainingMode.QLORA or not self.activation_checkpointing
        ):
            raise ValueError("qlora_checkpoint_stride requires QLoRA activation checkpointing")
        if self.qlora_loss_chunk_size is not None:
            if type(self.qlora_loss_chunk_size) is not int or self.qlora_loss_chunk_size <= 0:
                raise ValueError("qlora_loss_chunk_size must be a positive integer")
            if self.mode != TrainingMode.QLORA:
                raise ValueError("qlora_loss_chunk_size requires QLoRA")
        if self.qlora_loss_backend not in {"analytical", "triton", "auto"}:
            raise ValueError("qlora_loss_backend must be analytical, triton, or auto")
        if self.qlora_loss_backend != "analytical" and not self.qlora_precompute_head_gradient:
            raise ValueError(
                "non-analytical qlora_loss_backend requires qlora_precompute_head_gradient"
            )
        if self.qlora_attention_backend not in {"sdpa", "eager"}:
            raise ValueError("qlora_attention_backend must be sdpa or eager")
        if self.qlora_attention_backend != "sdpa" and self.mode != TrainingMode.QLORA:
            raise ValueError("qlora_attention_backend requires QLoRA")
        if type(self.qlora_activation_offload) is not bool:
            raise ValueError("qlora_activation_offload must be a boolean")
        if self.qlora_activation_offload and (
            self.mode != TrainingMode.QLORA or self.stream.device != "cuda"
        ):
            raise ValueError("qlora_activation_offload requires QLoRA on CUDA")
        if self.qlora_activation_offload and self.activation_checkpointing:
            raise ValueError(
                "qlora_activation_offload and activation_checkpointing are mutually exclusive"
            )
        if self.qlora_activation_compression_bits not in {None, 4}:
            raise ValueError("qlora_activation_compression_bits must be null or 4")
        if self.qlora_activation_compression_bits is not None and (
            self.mode != TrainingMode.QLORA or self.stream.device != "cuda"
        ):
            raise ValueError("QLoRA activation compression requires QLoRA on CUDA")
        if self.qlora_activation_compression_bits is not None and (
            self.activation_checkpointing or self.qlora_activation_offload
        ):
            raise ValueError(
                "QLoRA activation compression excludes checkpointing and CPU activation offload"
            )
        if (
            type(self.qlora_activation_compression_minimum_bytes) is not int
            or self.qlora_activation_compression_minimum_bytes < 0
        ):
            raise ValueError("QLoRA activation compression minimum bytes must be non-negative")
        if not isinstance(self.qlora_fused_optimizer, bool):
            raise ValueError("qlora_fused_optimizer must be a boolean")
        if self.qlora_fused_optimizer and (
            self.mode != TrainingMode.QLORA or self.stream.device != "cuda"
        ):
            raise ValueError("qlora_fused_optimizer requires QLoRA on CUDA")
        if not isinstance(self.qlora_autocast, bool):
            raise ValueError("qlora_autocast must be a boolean")
        if self.qlora_autocast and (
            self.mode != TrainingMode.QLORA or self.stream.device != "cuda"
        ):
            raise ValueError("qlora_autocast requires QLoRA on CUDA")
        if self.mode == TrainingMode.PRETRAIN and self.data.context_length != self.model.context_length:
            raise ValueError("data and model context lengths must match for pretraining")
        if min(self.batch_size, self.gradient_accumulation, self.max_steps) <= 0:
            raise ValueError("batch, accumulation and max_steps must be positive")
        if self.evaluation_interval is not None and self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive when provided")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.execution_backend not in {
            "eager", "compile-max-autotune", "compile-max-autotune-no-cudagraphs"
        }:
            raise ValueError(
                "execution_backend must be eager, compile-max-autotune, or "
                "compile-max-autotune-no-cudagraphs"
            )
        if self.mode == TrainingMode.QLORA and self.execution_backend != "eager":
            raise ValueError(
                "compiled execution backends are not yet checkpoint-safe for QLoRA; "
                "use eager rather than silently running an uncompiled model"
            )
        if not (0 < self.thermal_target_c < self.thermal_abort_c):
            raise ValueError("thermal_target_c must be positive and below thermal_abort_c")
        microbatch_guard_c = self.thermal_microbatch_guard_c or self.thermal_target_c
        if not self.thermal_target_c <= microbatch_guard_c < self.thermal_abort_c:
            raise ValueError(
                "thermal_microbatch_guard_c must be at or above thermal_target_c "
                "and below thermal_abort_c"
            )
        if self.thermal_recovery_mode not in {"stop", "cool-and-continue"}:
            raise ValueError("thermal_recovery_mode must be stop or cool-and-continue")
        recovery_c = self.thermal_recovery_c or self.thermal_target_c
        if not 0 < recovery_c < self.thermal_abort_c:
            raise ValueError("thermal_recovery_c must be below thermal_abort_c")
        if type(self.thermal_max_recoveries) is not int or self.thermal_max_recoveries < 0:
            raise ValueError("thermal_max_recoveries must be a non-negative integer")
        if self.thermal_control_mode in ("dual-gear", "intercooler"):
            if not 0.05 <= self.thermal_pause_seconds <= 30.0:
                raise ValueError("dual-gear thermal_pause_seconds must be between 0.05 and 30.0")
        elif not 0.03 <= self.thermal_pause_seconds <= 0.05:
            raise ValueError("thermal_pause_seconds must be between 0.03 and 0.05")
        if self.thermal_control_mode not in {
            "reactive", "predictive-cruise", "zone-cruise", "steady-duty", "intercooler", "dual-gear"
        }:
            raise ValueError(
                "thermal_control_mode must be reactive, predictive-cruise, zone-cruise, "
                "steady-duty, intercooler, or dual-gear"
            )
        if self.thermal_lookahead_seconds <= 0:
            raise ValueError("thermal_lookahead_seconds must be positive")
        if not 0 <= self.thermal_initial_pause_seconds <= self.thermal_max_pause_seconds:
            raise ValueError("thermal_initial_pause_seconds must be between zero and the maximum pause")
        if self.thermal_max_pause_seconds <= 0:
            raise ValueError("thermal_max_pause_seconds must be positive")
        if not 0 < self.thermal_stability_band_c < (self.thermal_abort_c - self.thermal_target_c):
            raise ValueError("thermal_stability_band_c must fit between target and abort")
        if self.power_limit_watts is not None and self.power_limit_watts <= 0:
            raise ValueError("power_limit_watts must be positive when provided")
        if self.thermal_power_target_watts is not None and self.thermal_power_target_watts <= 0:
            raise ValueError("thermal_power_target_watts must be positive when provided")
        if self.thermal_control_mode == "zone-cruise":
            if self.thermal_cruise_max_c is None:
                raise ValueError("zone-cruise requires thermal_cruise_max_c")
            if not self.thermal_target_c < self.thermal_cruise_max_c < self.thermal_abort_c:
                raise ValueError("thermal zones must satisfy target < cruise max < abort")
            if not 0 <= self.min_pause_ms <= self.max_pause_ms:
                raise ValueError("min_pause_ms must be between zero and max_pause_ms")
            if self.thermal_protective_pause_ms < self.max_pause_ms:
                raise ValueError("thermal_protective_pause_ms must be >= max_pause_ms")
            assert self.thermal_cruise_max_c is not None
            if not 0 < self.thermal_protective_hysteresis_c < (
                self.thermal_cruise_max_c - self.thermal_target_c
            ):
                raise ValueError("thermal_protective_hysteresis_c must fit in cruise window")
        if self.thermal_control_mode == "steady-duty":
            if self.thermal_cruise_max_c is None:
                raise ValueError("steady-duty requires thermal_cruise_max_c")
            if not self.thermal_target_c < self.thermal_cruise_max_c < self.thermal_abort_c:
                raise ValueError("steady-duty requires target < protective < abort")
            if not 0 <= self.thermal_steady_pause_ms <= self.thermal_protective_pause_ms:
                raise ValueError("steady pause must fit below the protective pause")
        if self.mode == TrainingMode.QLORA and not self.base_model:
            raise ValueError("QLoRA mode requires base_model")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingSpec":
        known_fields = {field.name for field in fields(cls)}
        unknown_fields = sorted(set(value) - known_fields)
        if unknown_fields:
            raise ValueError(
                "unknown training configuration field(s): " + ", ".join(unknown_fields)
            )
        if not isinstance(value.get("qlora_fused_optimizer", False), bool):
            raise ValueError("qlora_fused_optimizer must be a boolean")
        if not isinstance(value.get("qlora_autocast", False), bool):
            raise ValueError("qlora_autocast must be a boolean")
        data_value = dict(value["data"])
        data_value["path"] = _expand_config_path(str(data_value["path"]), "data.path")
        if data_value.get("validation_path") is not None:
            data_value["validation_path"] = _expand_config_path(
                str(data_value["validation_path"]), "data.validation_path"
            )
        base_model = value.get("base_model")
        if base_model is not None:
            base_model = _expand_config_path(str(base_model), "base_model")
        return cls(
            mode=TrainingMode(value["mode"]),
            data=DataSpec(**data_value),
            model=ModelSpec(**value.get("model", {})),
            stream=StreamSpec(**value.get("stream", {})),
            batch_size=int(value.get("batch_size", 1)),
            gradient_accumulation=int(value.get("gradient_accumulation", 1)),
            max_steps=int(value.get("max_steps", 10)),
            evaluation_interval=(
                int(value["evaluation_interval"])
                if value.get("evaluation_interval") is not None else None
            ),
            learning_rate=float(value.get("learning_rate", 2e-4)),
            seed=int(value.get("seed", 1337)),
            artifacts_dir=str(value.get("artifacts_dir", "artifacts/molt-stream")),
            base_model=base_model,
            execution_backend=str(value.get("execution_backend", "eager")),
            activation_checkpointing=bool(value.get("activation_checkpointing", True)),
            qlora_fused_optimizer=value.get("qlora_fused_optimizer", False),
            qlora_autocast=value.get("qlora_autocast", False),
            qlora_loss_chunk_size=value.get("qlora_loss_chunk_size"),
            qlora_loss_backend=str(value.get("qlora_loss_backend", "analytical")),
            qlora_attention_backend=str(value.get("qlora_attention_backend", "sdpa")),
            qlora_activation_offload=value.get("qlora_activation_offload", False),
            qlora_activation_compression_bits=value.get("qlora_activation_compression_bits"),
            qlora_activation_compression_minimum_bytes=value.get(
                "qlora_activation_compression_minimum_bytes", 1 << 20
            ),
            qlora_checkpoint_stride=value.get("qlora_checkpoint_stride", 1),
            qlora_precompute_head_gradient=value.get("qlora_precompute_head_gradient", False),
            qlora_cache_frozen_head=value.get("qlora_cache_frozen_head", True),
            qlora_restore_tied_embedding_bf16=value.get(
                "qlora_restore_tied_embedding_bf16", False
            ),
            qlora_frozen_rmsnorm_bf16=value.get("qlora_frozen_rmsnorm_bf16", False),
            qlora_scheduled_nf4_down_projection=value.get(
                "qlora_scheduled_nf4_down_projection", False
            ),
            qlora_bf16_adapter_shadows=value.get("qlora_bf16_adapter_shadows", False),
            qlora_validation_layer_guard=value.get("qlora_validation_layer_guard", False),
            qlora_intra_step_boundary_layer=value.get("qlora_intra_step_boundary_layer"),
            qlora_intra_step_pause_ms=float(value.get("qlora_intra_step_pause_ms", 0.0)),
            qlora_train_last_layers=value.get("qlora_train_last_layers"),
            qlora_full_warmup_steps=value.get("qlora_full_warmup_steps", 0),
            qlora_validation_batches=value.get("qlora_validation_batches", 4),
            thermal_target_c=float(value.get("thermal_target_c", 60.0)),
            thermal_abort_c=float(value.get("thermal_abort_c", 72.0)),
            thermal_microbatch_guard_c=(
                float(value["thermal_microbatch_guard_c"])
                if value.get("thermal_microbatch_guard_c") is not None else None
            ),
            thermal_pause_seconds=float(value.get("thermal_pause_seconds", 0.04)),
            thermal_control_mode=str(value.get("thermal_control_mode", "reactive")),
            thermal_lookahead_seconds=float(value.get("thermal_lookahead_seconds", 1.5)),
            thermal_max_pause_seconds=float(value.get("thermal_max_pause_seconds", 0.25)),
            thermal_initial_pause_seconds=float(value.get("thermal_initial_pause_seconds", 0.0)),
            thermal_stability_band_c=float(value.get("thermal_stability_band_c", 1.5)),
            thermal_power_target_watts=(
                float(value["thermal_power_target_watts"])
                if value.get("thermal_power_target_watts") is not None else None
            ),
            thermal_cruise_max_c=(
                float(value["thermal_cruise_max_c"])
                if value.get("thermal_cruise_max_c") is not None else None
            ),
            min_pause_ms=float(value.get("min_pause_ms", 10.0)),
            max_pause_ms=float(value.get("max_pause_ms", 25.0)),
            thermal_protective_pause_ms=float(
                value.get("thermal_protective_pause_ms", 100.0)
            ),
            thermal_protective_hysteresis_c=float(
                value.get("thermal_protective_hysteresis_c", 2.0)
            ),
            thermal_steady_pause_ms=float(value.get("thermal_steady_pause_ms", 40.0)),
            power_limit_watts=(
                float(value["power_limit_watts"])
                if value.get("power_limit_watts") is not None else None
            ),
            thermal_recovery_mode=str(value.get("thermal_recovery_mode", "stop")),
            thermal_recovery_c=(
                float(value["thermal_recovery_c"])
                if value.get("thermal_recovery_c") is not None else None
            ),
            thermal_max_recoveries=value.get("thermal_max_recoveries", 3),
        )


@dataclass(frozen=True)
class GateSpec:
    maximum_vram_bytes: int = 3_500_000_000
    maximum_incremental_rss_bytes: int = 1_500_000_000
    minimum_small_model_tokens_per_second: float = 35_000.0
    maximum_temperature_c: float = 72.0
    require_zero_thermal_throttle: bool = True


def load_spec(path: str | Path) -> TrainingSpec:
    """Load a TrainingSpec from a JSON configuration file."""
    target = Path(path)
    data = json.loads(target.read_text("utf-8"))
    return TrainingSpec.from_dict(data)
