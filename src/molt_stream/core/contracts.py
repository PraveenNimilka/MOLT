from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TelemetryPoint:
    monotonic_seconds: float
    process_rss_bytes: int
    gpu_power_watts: float | None
    gpu_used_bytes: int | None
    gpu_temperature_c: float | None
    gpu_utilization_percent: float | None
    gpu_enforced_power_limit_watts: float | None
    gpu_clock_event_reasons: int | None
    system_cpu_percent: float | None = None
    system_available_memory_bytes: int | None = None
    gpu_graphics_clock_mhz: int | None = None
    gpu_memory_clock_mhz: int | None = None
    gpu_process_used_bytes: int | None = None


class TelemetrySampler(Protocol):
    points: list[TelemetryPoint]

    def start(self) -> None: ...

    def stop(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class ProgressEvent:
    kind: str
    message: str = ""
    step: int = 0
    total_steps: int = 0
    elapsed_seconds: float = 0.0
    tokens_per_second: float | None = None
    loss: float | None = None
    vram_bytes: int | None = None
    gpu_temperature_c: float | None = None
    thermal_state: str = "full-speed"
    thermal_pause_seconds: float = 0.0
    initial_step: int = 0
