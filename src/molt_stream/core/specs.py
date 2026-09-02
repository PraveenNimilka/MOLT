from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


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
    learning_rate: float = 2e-4
    seed: int = 1337
    artifacts_dir: str = "artifacts/molt-stream"
    base_model: str | None = None
    execution_backend: str = "eager"
    thermal_target_c: float = 60.0
    thermal_abort_c: float = 72.0
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

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.stream.validate()
        if min(self.batch_size, self.gradient_accumulation, self.max_steps) <= 0:
            raise ValueError("batch, accumulation and max_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.execution_backend not in {
            "eager", "compile-max-autotune", "compile-max-autotune-no-cudagraphs"
        }:
            raise ValueError(
                "execution_backend must be eager, compile-max-autotune, or "
                "compile-max-autotune-no-cudagraphs"
            )
        if not (0 < self.thermal_target_c < self.thermal_abort_c):
            raise ValueError("thermal_target_c must be positive and below thermal_abort_c")
        if not 0.03 <= self.thermal_pause_seconds <= 0.05:
            raise ValueError("thermal_pause_seconds must be between 0.03 and 0.05")
        if self.thermal_control_mode not in {
            "reactive", "predictive-cruise", "zone-cruise", "steady-duty"
        }:
            raise ValueError(
                "thermal_control_mode must be reactive, predictive-cruise, zone-cruise, "
                "or steady-duty"
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
        return cls(
            mode=TrainingMode(value["mode"]),
            data=DataSpec(**value["data"]),
            model=ModelSpec(**value.get("model", {})),
            stream=StreamSpec(**value.get("stream", {})),
            batch_size=int(value.get("batch_size", 1)),
            gradient_accumulation=int(value.get("gradient_accumulation", 1)),
            max_steps=int(value.get("max_steps", 10)),
            learning_rate=float(value.get("learning_rate", 2e-4)),
            seed=int(value.get("seed", 1337)),
            artifacts_dir=str(value.get("artifacts_dir", "artifacts/molt-stream")),
            base_model=value.get("base_model"),
            execution_backend=str(value.get("execution_backend", "eager")),
            thermal_target_c=float(value.get("thermal_target_c", 60.0)),
            thermal_abort_c=float(value.get("thermal_abort_c", 72.0)),
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
        )


@dataclass(frozen=True)
class GateSpec:
    maximum_vram_bytes: int = 3_500_000_000
    maximum_incremental_rss_bytes: int = 1_500_000_000
    minimum_small_model_tokens_per_second: float = 35_000.0
    maximum_temperature_c: float = 72.0
    require_zero_thermal_throttle: bool = True
