from __future__ import annotations

import threading
import time
from dataclasses import asdict

import psutil

from molt_stream.core.contracts import TelemetryPoint


def manage_power_limit(
    target_watts: float = 65.0, *, apply: bool = False, device_index: int = 0
) -> dict[str, object]:
    """Query and optionally set an NVML power limit, returning verification.

    Setting is never implicit: callers must pass ``apply=True``. Laptop firmware
    commonly rejects this operation even when NVML reports constraint values.
    """
    if target_watts <= 0:
        raise ValueError("target_watts must be positive")
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        enforced_before = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        try:
            minimum_mw, maximum_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            minimum_watts, maximum_watts = minimum_mw / 1000.0, maximum_mw / 1000.0
            supported: bool | None = minimum_watts <= target_watts <= maximum_watts
        except Exception:
            minimum_watts = maximum_watts = None
            supported = None
        error = None
        applied = False
        if apply:
            if supported is False:
                error = f"target {target_watts}W is outside {minimum_watts}-{maximum_watts}W"
            else:
                try:
                    pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(target_watts * 1000))
                    applied = True
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
        enforced_after = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        return {
            "target_watts": target_watts,
            "apply_requested": apply,
            "set_call_succeeded": applied,
            "supported_range": supported,
            "minimum_watts": minimum_watts,
            "maximum_watts": maximum_watts,
            "enforced_before_watts": enforced_before,
            "enforced_after_watts": enforced_after,
            "verified": abs(enforced_after - target_watts) < 0.5,
            "error": error,
        }
    finally:
        pynvml.nvmlShutdown()


def integrate_board_energy(points: list[TelemetryPoint]) -> float | None:
    powered = [point for point in points if point.gpu_power_watts is not None]
    if len(powered) < 2:
        return None
    return sum(
        (right.monotonic_seconds - left.monotonic_seconds)
        * (float(left.gpu_power_watts) + float(right.gpu_power_watts)) / 2
        for left, right in zip(powered, powered[1:])
    )


class NVMLTelemetry:
    def __init__(self, interval_seconds: float = 0.1, *, enable_gpu: bool = True):
        if interval_seconds <= 0:
            raise ValueError("interval must be positive")
        self.interval_seconds = interval_seconds
        self.enable_gpu = enable_gpu
        self.points: list[TelemetryPoint] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = self._handle = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry already started")
        if self.enable_gpu:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as exc:
                self.errors.append(f"NVML unavailable: {type(exc).__name__}: {exc}")
        self._thread = threading.Thread(target=self._sample, daemon=True, name="molt-nvml")
        self._thread.start()

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            power = used = temp = util = limit = reasons = None
            if self._nvml is not None:
                try:
                    power = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000
                    used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
                    temp = float(self._nvml.nvmlDeviceGetTemperature(self._handle, 0))
                    util = float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
                    limit = self._nvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000
                    reasons = int(self._nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._handle))
                except Exception as exc:
                    self.errors.append(f"NVML sample failed: {type(exc).__name__}: {exc}")
            self.points.append(
                TelemetryPoint(
                    time.perf_counter(), process.memory_info().rss, power, used, temp, util,
                    limit, reasons,
                )
            )
            self._stop.wait(self.interval_seconds)

    def stop(self) -> dict[str, object]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, 3 * self.interval_seconds))
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception as exc:
                self.errors.append(f"NVML shutdown failed: {type(exc).__name__}: {exc}")
        energy = integrate_board_energy(self.points)
        powers = [float(p.gpu_power_watts) for p in self.points if p.gpu_power_watts is not None]
        temps = [float(p.gpu_temperature_c) for p in self.points if p.gpu_temperature_c is not None]
        used = [int(p.gpu_used_bytes) for p in self.points if p.gpu_used_bytes is not None]
        utilization = [float(p.gpu_utilization_percent) for p in self.points if p.gpu_utilization_percent is not None]
        reasons = [int(p.gpu_clock_event_reasons) for p in self.points if p.gpu_clock_event_reasons is not None]
        reason_mask = 0
        for reason in reasons:
            reason_mask |= reason
        limits = [float(p.gpu_enforced_power_limit_watts) for p in self.points if p.gpu_enforced_power_limit_watts is not None]
        return {
            "sample_count": len(self.points), "gpu_board_energy_joules": energy,
            "mean_gpu_power_watts": sum(powers) / len(powers) if powers else None,
            "mean_gpu_utilization_percent": sum(utilization) / len(utilization) if utilization else None,
            "peak_gpu_utilization_percent": max(utilization) if utilization else None,
            "minimum_enforced_power_limit_watts": min(limits) if limits else None,
            "maximum_enforced_power_limit_watts": max(limits) if limits else None,
            "peak_gpu_temperature_c": max(temps) if temps else None,
            "peak_gpu_used_bytes": max(used) if used else None,
            "thermal_throttle_observed": any(reason & (0x20 | 0x40) for reason in reasons),
            "gpu_clock_event_reason_mask": reason_mask if reasons else None,
            "peak_process_rss_bytes": max((p.process_rss_bytes for p in self.points), default=None),
            "errors": self.errors, "points": [asdict(point) for point in self.points],
        }
