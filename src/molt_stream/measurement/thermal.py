from __future__ import annotations

import time
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from molt_stream.core.contracts import TelemetryPoint

if TYPE_CHECKING:
    from molt_stream.core.specs import TrainingSpec


@dataclass(frozen=True)
class MicrobatchThermalDecision:
    pause_seconds: float
    temperature_c: float | None
    stop_reason: str | None = None


@dataclass(frozen=True)
class SystemStartupThermalDecision:
    pause_seconds: float
    gpu_temperature_c: float | None
    cpu_temperature_c: float | None
    cpu_sensor_available: bool
    stop_reason: str | None = None


def wait_for_stable_system_headroom(
    read: Callable[[], TelemetryPoint | None],
    *,
    maximum_gpu_c: float | None,
    maximum_cpu_c: float | None,
    dwell_seconds: float,
    maximum_wait_seconds: float = 300.0,
    maximum_sample_age_seconds: float = 1.0,
    notify: Callable[[float, float | None, float | None], None] | None = None,
) -> SystemStartupThermalDecision:
    """Wait passively for a stable GPU/CPU starting temperature.

    GPU telemetry remains fail-closed when a GPU limit is requested. CPU
    telemetry is best-effort because Windows has no universal CPU sensor API;
    an unavailable CPU sensor is explicitly reported and never fabricated.
    """
    if maximum_gpu_c is None and maximum_cpu_c is None:
        raise ValueError("at least one startup thermal limit is required")
    limits = [value for value in (maximum_gpu_c, maximum_cpu_c) if value is not None]
    if not all(math.isfinite(value) and 0 < value < 125 for value in limits):
        raise ValueError("startup thermal limits must be positive finite temperatures")
    if not (
        math.isfinite(dwell_seconds)
        and dwell_seconds > 0
        and math.isfinite(maximum_wait_seconds)
        and maximum_wait_seconds >= dwell_seconds
        and maximum_sample_age_seconds > 0
    ):
        raise ValueError("Invalid stable system headroom settings")

    started = time.perf_counter()
    stable_since: float | None = None
    saw_cpu_sensor = False
    last_gpu = last_cpu = None
    while True:
        sample = read()
        now = time.perf_counter()
        if sample is None or not 0 <= now - sample.monotonic_seconds <= maximum_sample_age_seconds:
            return SystemStartupThermalDecision(
                now - started, last_gpu, last_cpu, saw_cpu_sensor,
                "thermal telemetry missing or stale",
            )
        last_gpu = sample.gpu_temperature_c
        last_cpu = sample.cpu_temperature_c
        gpu_valid = last_gpu is not None and math.isfinite(last_gpu)
        cpu_valid = last_cpu is not None and math.isfinite(last_cpu)
        saw_cpu_sensor = saw_cpu_sensor or cpu_valid
        if maximum_gpu_c is not None and not gpu_valid:
            return SystemStartupThermalDecision(
                now - started, last_gpu, last_cpu, saw_cpu_sensor,
                "GPU thermal telemetry missing or stale",
            )
        gpu_cool = maximum_gpu_c is None or (gpu_valid and last_gpu <= maximum_gpu_c)
        # CPU is enforced whenever its sensor exists. An absent sensor cannot
        # safely be interpreted as either hot or cool, so it is disclosed and
        # the GPU gate continues independently.
        cpu_cool = maximum_cpu_c is None or not cpu_valid or last_cpu <= maximum_cpu_c
        if gpu_cool and cpu_cool:
            stable_since = now if stable_since is None else stable_since
            if now - stable_since >= dwell_seconds:
                return SystemStartupThermalDecision(
                    now - started, last_gpu, last_cpu, saw_cpu_sensor
                )
        else:
            stable_since = None
        if now - started >= maximum_wait_seconds:
            return SystemStartupThermalDecision(
                now - started, last_gpu, last_cpu, saw_cpu_sensor,
                "stable startup cooling timeout",
            )
        if notify is not None:
            notify(now - started, last_gpu, last_cpu)
        time.sleep(min(0.25, maximum_wait_seconds - (now - started)))


def postrun_thermal_violation(
    peak_temperature_c: float | None, *, abort_c: float
) -> bool:
    """Return whether delayed telemetry crossed the registered boundary.

    NVML temperature is sampled asynchronously and may report the physical
    peak only after the last CUDA synchronization.  Centralising this rule
    keeps the run state and the published metrics consistent.
    """
    if not math.isfinite(abort_c) or abort_c <= 0:
        raise ValueError("abort_c must be a positive finite temperature")
    return bool(
        peak_temperature_c is not None
        and math.isfinite(peak_temperature_c)
        and peak_temperature_c >= abort_c
    )


def passive_cooldown(
    read: Callable[[], TelemetryPoint | None], *, recovery_c: float,
    maximum_wait_seconds: float = 120.0,
    notify: Callable[[float, float], None] | None = None,
) -> MicrobatchThermalDecision:
    """Launch no compute while waiting for a fresh safe sample."""
    if recovery_c <= 0 or maximum_wait_seconds <= 0:
        raise ValueError("Invalid passive cooldown settings")
    started = time.perf_counter()
    while True:
        sample = read()
        now = time.perf_counter()
        temperature = None if sample is None else sample.gpu_temperature_c
        if (sample is None or temperature is None or not math.isfinite(temperature)
                or not 0 <= now - sample.monotonic_seconds <= 1.0):
            return MicrobatchThermalDecision(now - started, temperature,
                                             "thermal telemetry missing or stale")
        if temperature <= recovery_c:
            return MicrobatchThermalDecision(now - started, temperature)
        if now - started >= maximum_wait_seconds:
            return MicrobatchThermalDecision(now - started, temperature,
                                             "passive cooling timeout")
        if notify is not None:
            notify(now - started, temperature)
        time.sleep(min(0.25, maximum_wait_seconds - (now - started)))


def wait_for_stable_thermal_headroom(
    read: Callable[[], TelemetryPoint | None],
    *,
    maximum_c: float,
    dwell_seconds: float,
    maximum_wait_seconds: float = 300.0,
    maximum_sample_age_seconds: float = 1.0,
    notify: Callable[[float, float], None] | None = None,
) -> MicrobatchThermalDecision:
    """Require continuously cool telemetry before starting a workload.

    A single low die-temperature sample does not prove that a laptop heat pipe
    or vapor chamber has cooled. The dwell resets whenever temperature rises
    above ``maximum_c``. Missing or stale telemetry fails closed.
    """

    if not (
        math.isfinite(maximum_c)
        and maximum_c > 0
        and math.isfinite(dwell_seconds)
        and dwell_seconds > 0
        and math.isfinite(maximum_wait_seconds)
        and maximum_wait_seconds >= dwell_seconds
        and maximum_sample_age_seconds > 0
    ):
        raise ValueError("Invalid stable thermal headroom settings")
    started = time.perf_counter()
    stable_since: float | None = None
    while True:
        sample = read()
        now = time.perf_counter()
        temperature = None if sample is None else sample.gpu_temperature_c
        if (
            sample is None
            or temperature is None
            or not math.isfinite(temperature)
            or not 0 <= now - sample.monotonic_seconds <= maximum_sample_age_seconds
        ):
            return MicrobatchThermalDecision(
                now - started, temperature, "thermal telemetry missing or stale"
            )
        if temperature <= maximum_c:
            stable_since = now if stable_since is None else stable_since
            if now - stable_since >= dwell_seconds:
                return MicrobatchThermalDecision(now - started, temperature)
        else:
            stable_since = None
        if now - started >= maximum_wait_seconds:
            return MicrobatchThermalDecision(
                now - started, temperature, "stable startup cooling timeout"
            )
        if notify is not None:
            notify(now - started, temperature)
        time.sleep(min(0.25, maximum_wait_seconds - (now - started)))


def wait_for_thermal_headroom(
    read: Callable[[], TelemetryPoint | None], *, target_c: float, abort_c: float,
    maximum_wait_seconds: float = 30.0, maximum_sample_age_seconds: float = 1.0,
    notify: Callable[[float, float], None] | None = None,
) -> MicrobatchThermalDecision:
    """Gate the next microbatch; stale/missing telemetry fails closed.

    Does not preempt an executing CUDA kernel. A hot sample waits for one degree
    of hysteresis below target, with a bounded timeout. Notification is advisory.
    """
    if not (0 < target_c < abort_c and maximum_wait_seconds > 0 and maximum_sample_age_seconds > 0):
        raise ValueError("Invalid microbatch thermal boundaries")
    started = time.perf_counter()
    waited = False
    while True:
        sample = read()
        now = time.perf_counter()
        temperature = None if sample is None else sample.gpu_temperature_c
        if (sample is None or temperature is None or not math.isfinite(temperature)
                or not 0 <= now - sample.monotonic_seconds <= maximum_sample_age_seconds):
            return MicrobatchThermalDecision(now - started if waited else 0.0, temperature,
                                            "thermal telemetry missing or stale")
        if temperature >= abort_c:
            return MicrobatchThermalDecision(now - started if waited else 0.0, temperature,
                                            "thermal abort boundary reached")
        if temperature <= target_c - (1.0 if waited else 0.0):
            return MicrobatchThermalDecision(now - started if waited else 0.0, temperature)
        if now - started >= maximum_wait_seconds:
            return MicrobatchThermalDecision(now - started, temperature, "cooling timeout")
        if notify is not None:
            notify(now - started, temperature)
        time.sleep(min(0.1, maximum_wait_seconds - (now - started)))
        waited = True


def cooling_pause(
    seconds: float,
    temperature: Callable[[], float | None],
    *,
    recovery_c: float,
    abort_c: float,
    notify: Callable[[float, float | None], None] | None = None,
) -> float:
    """Return actual wall pause time; UI observers never select cooling policy.

    Long pauses check temperature every half second and return at recovery or
    abort so the caller can checkpoint. Short micro-pauses retain their duration.
    Missing telemetry never counts as recovery.
    """
    started = time.perf_counter()
    remaining = max(0.0, seconds)
    while remaining > 0:
        time.sleep(min(0.5, remaining) if seconds >= 2.0 else remaining)
        remaining = max(0.0, seconds - (time.perf_counter() - started))
        current = temperature()
        if seconds < 2.0 or (current is not None and (current <= recovery_c or current >= abort_c)):
            break
        if notify is not None:
            notify(remaining, current)
    return time.perf_counter() - started


def latest_temperature_c(points: Sequence[TelemetryPoint]) -> float | None:
    return next(
        (float(point.gpu_temperature_c) for point in reversed(points) if point.gpu_temperature_c is not None),
        None,
    )


def latest_cpu_temperature_c(points: Sequence[TelemetryPoint]) -> float | None:
    return next(
        (
            float(point.cpu_temperature_c)
            for point in reversed(points)
            if point.cpu_temperature_c is not None
        ),
        None,
    )


def latest_telemetry_point(points: Sequence[TelemetryPoint]) -> TelemetryPoint | None:
    """Return the newest GPU-temperature sample without copying telemetry."""
    return next((point for point in reversed(points) if point.gpu_temperature_c is not None), None)


@dataclass(frozen=True)
class ThermalDecision:
    pause_seconds: float
    temperature_c: float | None
    filtered_temperature_c: float | None
    heating_rate_c_per_second: float
    projected_temperature_c: float | None
    abort: bool
    phase: str


class ZonedThermalController:
    """Deterministic pacing plus an independent emergency safety boundary.

    The normal cruise window uses smoothstep interpolation so adjacent whole-
    degree NVML samples do not create large pause jumps. Above the cruise
    ceiling, a longer recovery pause is allowed; the advertised 10-25 ms bound
    applies only inside the normal cruise window.
    """

    def __init__(
        self,
        *,
        target_c: float,
        cruise_max_c: float,
        abort_c: float,
        minimum_pause_seconds: float = 0.010,
        maximum_pause_seconds: float = 0.025,
        protective_pause_seconds: float = 0.100,
        protective_hysteresis_c: float = 2.0,
    ) -> None:
        if not 0 < target_c < cruise_max_c < abort_c:
            raise ValueError("thermal zones must satisfy target < cruise max < abort")
        if not 0 <= minimum_pause_seconds <= maximum_pause_seconds:
            raise ValueError("minimum pause must not exceed maximum pause")
        if protective_pause_seconds < maximum_pause_seconds:
            raise ValueError("protective pause must be at least the cruise maximum")
        if not 0 < protective_hysteresis_c < (cruise_max_c - target_c):
            raise ValueError("protective hysteresis must fit inside the cruise window")
        self.target_c = target_c
        self.cruise_max_c = cruise_max_c
        self.abort_c = abort_c
        self.minimum_pause_seconds = minimum_pause_seconds
        self.maximum_pause_seconds = maximum_pause_seconds
        self.protective_pause_seconds = protective_pause_seconds
        self.protective_hysteresis_c = protective_hysteresis_c
        self._protective_latched = False

    def update(
        self,
        point: TelemetryPoint | None,
        *,
        step_seconds: float,
    ) -> ThermalDecision:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if point is None or point.gpu_temperature_c is None:
            return ThermalDecision(0.0, None, None, 0.0, None, False, "no-telemetry")
        temperature = float(point.gpu_temperature_c)
        if temperature >= self.abort_c:
            return ThermalDecision(
                0.0, temperature, temperature, 0.0, temperature, True, "thermal-abort"
            )
        if self._protective_latched:
            if temperature <= self.cruise_max_c - self.protective_hysteresis_c:
                self._protective_latched = False
            else:
                return ThermalDecision(
                    self.protective_pause_seconds,
                    temperature,
                    temperature,
                    0.0,
                    temperature,
                    False,
                    "protective-cooling",
                )
        if temperature < self.target_c:
            return ThermalDecision(
                0.0, temperature, temperature, 0.0, temperature, False, "full-speed"
            )
        if temperature <= self.cruise_max_c:
            fraction = (temperature - self.target_c) / (
                self.cruise_max_c - self.target_c
            )
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            pause = self.minimum_pause_seconds + smooth * (
                self.maximum_pause_seconds - self.minimum_pause_seconds
            )
            return ThermalDecision(
                pause, temperature, temperature, 0.0, temperature, False, "cooling"
            )
        self._protective_latched = True
        return ThermalDecision(
            self.protective_pause_seconds,
            temperature,
            temperature,
            0.0,
            temperature,
            False,
            "protective-cooling",
        )


class SteadyDutyThermalController:
    """Constant proactive pacing with a hotter protective recovery latch."""

    def __init__(
        self,
        *,
        steady_pause_seconds: float,
        protective_c: float,
        abort_c: float,
        protective_pause_seconds: float = 0.100,
        protective_hysteresis_c: float = 2.0,
    ) -> None:
        if steady_pause_seconds < 0:
            raise ValueError("steady pause must be non-negative")
        if not 0 < protective_c < abort_c:
            raise ValueError("protective temperature must be below abort")
        if protective_pause_seconds < steady_pause_seconds:
            raise ValueError("protective pause must be at least the steady pause")
        if not 0 < protective_hysteresis_c < protective_c:
            raise ValueError("protective hysteresis must be positive and bounded")
        self.steady_pause_seconds = steady_pause_seconds
        self.protective_c = protective_c
        self.abort_c = abort_c
        self.protective_pause_seconds = protective_pause_seconds
        self.protective_hysteresis_c = protective_hysteresis_c
        self._protective_latched = False

    def update(
        self,
        point: TelemetryPoint | None,
        *,
        step_seconds: float,
    ) -> ThermalDecision:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if point is None or point.gpu_temperature_c is None:
            return ThermalDecision(
                self.steady_pause_seconds, None, None, 0.0, None, False, "cooling"
            )
        temperature = float(point.gpu_temperature_c)
        if temperature >= self.abort_c:
            return ThermalDecision(
                0.0, temperature, temperature, 0.0, temperature, True, "thermal-abort"
            )
        if self._protective_latched:
            if temperature <= self.protective_c - self.protective_hysteresis_c:
                self._protective_latched = False
            else:
                return ThermalDecision(
                    self.protective_pause_seconds,
                    temperature,
                    temperature,
                    0.0,
                    temperature,
                    False,
                    "protective-cooling",
                )
        if temperature >= self.protective_c:
            self._protective_latched = True
            return ThermalDecision(
                self.protective_pause_seconds,
                temperature,
                temperature,
                0.0,
                temperature,
                False,
                "protective-cooling",
            )
        return ThermalDecision(
            self.steady_pause_seconds,
            temperature,
            temperature,
            0.0,
            temperature,
            False,
            "cooling",
        )


class DualGearThermalController:
    """Two pacing states with temperature hysteresis; throughput is workload-dependent."""

    def __init__(
        self,
        *,
        shift_down_c: float = 74.0,
        shift_up_c: float = 65.0,
        gear1_pause_seconds: float = 0.0,
        gear2_pause_seconds: float = 0.22,
        abort_c: float = 85.0,
    ) -> None:
        if not 0 < shift_up_c < shift_down_c < abort_c:
            raise ValueError("Thermal gears must satisfy shift_up < shift_down < abort")
        if gear1_pause_seconds < 0 or gear2_pause_seconds < 0:
            raise ValueError("Gear pauses must be non-negative")
        self.shift_down_c = shift_down_c
        self.shift_up_c = shift_up_c
        self.gear1_pause_seconds = gear1_pause_seconds
        self.gear2_pause_seconds = gear2_pause_seconds
        self.abort_c = abort_c
        self._gear = 1

    def update(
        self,
        point: TelemetryPoint | None,
        *,
        step_seconds: float,
    ) -> ThermalDecision:
        if point is None or point.gpu_temperature_c is None:
            pause = self.gear1_pause_seconds if self._gear == 1 else self.gear2_pause_seconds
            phase = "gear-1-sprint" if self._gear == 1 else "gear-2-cooldown"
            return ThermalDecision(pause, None, None, 0.0, None, False, phase)

        temp = float(point.gpu_temperature_c)
        if temp >= self.abort_c:
            return ThermalDecision(0.0, temp, temp, 0.0, temp, True, "thermal-abort")

        if self._gear == 2:
            if temp <= self.shift_up_c:
                self._gear = 1
                return ThermalDecision(
                    self.gear1_pause_seconds,
                    temp,
                    temp,
                    0.0,
                    temp,
                    False,
                    "gear-1-sprint",
                )
            return ThermalDecision(
                self.gear2_pause_seconds,
                temp,
                temp,
                0.0,
                temp,
                False,
                "gear-2-cooldown",
            )

        if temp >= self.shift_down_c:
            self._gear = 2
            return ThermalDecision(
                self.gear2_pause_seconds,
                temp,
                temp,
                0.0,
                temp,
                False,
                "gear-2-cooldown",
            )

        return ThermalDecision(
            self.gear1_pause_seconds,
            temp,
            temp,
            0.0,
            temp,
            False,
            "gear-1-sprint",
        )


IntercoolerThermalController = DualGearThermalController


class ThermalCruiseController:
    """Predictive host-side pacing for a thermally constrained training loop.

    This is deliberately a controller, not a claim of a new optimization
    algorithm. It filters NVML's integer temperature samples, estimates the
    heating slope, projects that slope over a short horizon, and controls idle
    time before the measured temperature crosses the target. All pauses are
    outside CUDA work and therefore included in end-to-end throughput.
    """

    def __init__(
        self,
        *,
        target_c: float,
        abort_c: float,
        lookahead_seconds: float = 1.5,
        stability_band_c: float = 1.5,
        initial_pause_seconds: float = 0.0,
        maximum_pause_seconds: float = 0.25,
        power_target_watts: float | None = None,
    ) -> None:
        if not 0 < target_c < abort_c:
            raise ValueError("target_c must be positive and below abort_c")
        if lookahead_seconds <= 0 or stability_band_c <= 0:
            raise ValueError("lookahead and stability band must be positive")
        if power_target_watts is not None and power_target_watts <= 0:
            raise ValueError("power target must be positive when provided")
        if not 0 <= initial_pause_seconds <= maximum_pause_seconds:
            raise ValueError("initial pause must be between zero and maximum pause")
        self.target_c = target_c
        self.abort_c = abort_c
        self.lookahead_seconds = lookahead_seconds
        self.stability_band_c = stability_band_c
        self.initial_pause_seconds = initial_pause_seconds
        self.maximum_pause_seconds = maximum_pause_seconds
        self.power_target_watts = power_target_watts
        self._filtered_temperature_c: float | None = None
        self._filtered_slope = 0.0
        self._last_sample_time: float | None = None
        self._last_pause_seconds = initial_pause_seconds
        self._pause_to_compute_ratio: float | None = None
        self._cruise_ratio: float | None = None
        self._filtered_power_watts: float | None = None

    def update(
        self,
        point: TelemetryPoint | None,
        *,
        step_seconds: float,
    ) -> ThermalDecision:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if point is None or point.gpu_temperature_c is None:
            pause = self._last_pause_seconds
            return ThermalDecision(pause, None, None, 0.0, None, False, "no-telemetry")

        temperature = float(point.gpu_temperature_c)
        if point.gpu_power_watts is not None:
            power = float(point.gpu_power_watts)
            if self._filtered_power_watts is None:
                self._filtered_power_watts = power
            else:
                self._filtered_power_watts += 0.30 * (power - self._filtered_power_watts)
        previous_filtered = self._filtered_temperature_c
        if previous_filtered is None:
            self._filtered_temperature_c = temperature
        else:
            self._filtered_temperature_c += 0.20 * (temperature - self._filtered_temperature_c)

        if (
            self._last_sample_time is not None
            and previous_filtered is not None
            and point.monotonic_seconds > self._last_sample_time
        ):
            elapsed = point.monotonic_seconds - self._last_sample_time
            raw_slope = (self._filtered_temperature_c - previous_filtered) / elapsed
            raw_slope = max(-2.0, min(2.0, raw_slope))
            self._filtered_slope += 0.20 * (raw_slope - self._filtered_slope)
        self._last_sample_time = point.monotonic_seconds

        projected = self._filtered_temperature_c + max(0.0, self._filtered_slope) * self.lookahead_seconds
        control_edge = self.target_c - self.stability_band_c / 2.0
        projected_error = projected - control_edge
        if self._pause_to_compute_ratio is None:
            self._pause_to_compute_ratio = self.initial_pause_seconds / step_seconds
            self._cruise_ratio = self._pause_to_compute_ratio
        if self._cruise_ratio is None:
            raise RuntimeError("thermal cruise controller failed to initialize")
        # Learn the equilibrium duty ratio slowly and symmetrically. Keeping
        # this state separate from fast feedback prevents integral wind-up.
        equilibrium_error = self._filtered_temperature_c - self.target_c
        self._cruise_ratio = max(
            0.0,
            min(4.0, self._cruise_ratio + max(-0.03, min(0.03, 0.01 * equilibrium_error))),
        )
        desired_ratio = max(
            0.0,
            min(
                4.0,
                self._cruise_ratio
                + 0.14 * max(0.0, projected_error)
                + 0.08 * max(0.0, self._filtered_slope),
            ),
        )
        if self.power_target_watts is not None and self._filtered_power_watts is not None:
            # Approximate the idle board draw conservatively. For active power
            # P and target average T, pause/compute ~= (P-T)/(T-P_idle).
            idle_power_watts = min(15.0, 0.5 * self.power_target_watts)
            denominator = max(self.power_target_watts - idle_power_watts, 1.0)
            power_feed_forward = max(
                0.0,
                (self._filtered_power_watts - self.power_target_watts) / denominator,
            )
            desired_ratio = max(desired_ratio, min(4.0, power_feed_forward))
        self._pause_to_compute_ratio += 0.18 * (
            desired_ratio - self._pause_to_compute_ratio
        )
        if projected_error > 0:
            phase = "braking" if projected >= self.target_c else "approach"
        else:
            phase = "cruise" if abs(self._filtered_temperature_c - self.target_c) <= self.stability_band_c else "warmup"
        pause = step_seconds * self._pause_to_compute_ratio
        pause = min(self.maximum_pause_seconds, pause)
        self._last_pause_seconds = pause
        return ThermalDecision(
            pause,
            temperature,
            self._filtered_temperature_c,
            self._filtered_slope,
            projected,
            temperature >= self.abort_c,
            phase,
        )


def build_thermal_controller(
    spec: TrainingSpec,
) -> ThermalCruiseController | ZonedThermalController | SteadyDutyThermalController | DualGearThermalController | None:
    """Build the explicitly configured controller without hidden fallback."""
    if spec.thermal_control_mode in ("dual-gear", "intercooler"):
        return DualGearThermalController(
            shift_down_c=spec.thermal_target_c,
            shift_up_c=spec.thermal_cruise_max_c if spec.thermal_cruise_max_c is not None else 65.0,
            gear1_pause_seconds=0.0,
            gear2_pause_seconds=spec.thermal_pause_seconds if spec.thermal_pause_seconds > 0 else 0.22,
            abort_c=spec.thermal_abort_c,
        )
    if spec.thermal_control_mode == "predictive-cruise":
        return ThermalCruiseController(
            target_c=spec.thermal_target_c,
            abort_c=spec.thermal_abort_c,
            lookahead_seconds=spec.thermal_lookahead_seconds,
            stability_band_c=spec.thermal_stability_band_c,
            initial_pause_seconds=spec.thermal_initial_pause_seconds,
            maximum_pause_seconds=spec.thermal_max_pause_seconds,
            power_target_watts=spec.thermal_power_target_watts,
        )
    if spec.thermal_control_mode == "zone-cruise":
        if spec.thermal_cruise_max_c is None:
            raise ValueError("zone-cruise requires thermal_cruise_max_c")
        return ZonedThermalController(
            target_c=spec.thermal_target_c,
            cruise_max_c=spec.thermal_cruise_max_c,
            abort_c=spec.thermal_abort_c,
            minimum_pause_seconds=spec.min_pause_ms / 1000.0,
            maximum_pause_seconds=spec.max_pause_ms / 1000.0,
            protective_pause_seconds=spec.thermal_protective_pause_ms / 1000.0,
            protective_hysteresis_c=spec.thermal_protective_hysteresis_c,
        )
    if spec.thermal_control_mode == "steady-duty":
        if spec.thermal_cruise_max_c is None:
            raise ValueError("steady-duty requires thermal_cruise_max_c")
        return SteadyDutyThermalController(
            steady_pause_seconds=spec.thermal_steady_pause_ms / 1000.0,
            protective_c=spec.thermal_cruise_max_c,
            abort_c=spec.thermal_abort_c,
            protective_pause_seconds=spec.thermal_protective_pause_ms / 1000.0,
            protective_hysteresis_c=spec.thermal_protective_hysteresis_c,
        )
    return None


def duty_cycle_pause_seconds(
    temperature_c: float | None,
    *,
    target_c: float,
    abort_c: float,
    nominal_seconds: float = 0.04,
) -> float:
    """Return a bounded 30-50 ms pause proportional to target overshoot."""
    if temperature_c is None or temperature_c <= target_c:
        return 0.0
    span = max(abort_c - target_c, 1e-6)
    fraction = min(1.0, (temperature_c - target_c) / span)
    return min(0.05, max(0.03, nominal_seconds + 0.01 * (2.0 * fraction - 1.0)))
