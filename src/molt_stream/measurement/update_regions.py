from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch


@dataclass(frozen=True)
class PowerWindow:
    joules: float | None
    covered_seconds: float
    window_seconds: float


class CapturedCudaRegionTimer:
    """Reusable timing events that are recorded as external CUDA-graph nodes."""

    def __init__(self, markers: Sequence[str]) -> None:
        names = tuple(dict.fromkeys(markers))
        if not names:
            raise ValueError("at least one CUDA region marker is required")
        self._events = {
            name: torch.cuda.Event(enable_timing=True, external=True) for name in names
        }

    def record(self, marker: str) -> None:
        try:
            event = self._events[marker]
        except KeyError as exc:
            raise ValueError(f"unknown CUDA region marker: {marker}") from exc
        event.record()

    def elapsed_seconds(self, start: str, end: str) -> float:
        try:
            return self._events[start].elapsed_time(self._events[end]) / 1000.0
        except KeyError as exc:
            raise ValueError(f"unknown CUDA region marker: {exc.args[0]}") from exc


def _point_value(point: object, name: str) -> float | None:
    value = point.get(name) if isinstance(point, Mapping) else getattr(point, name)
    return None if value is None else float(value)


def integrate_power_window(
    points: Iterable[object], start: float, end: float
) -> PowerWindow:
    """Integrate sampled board power, clipping and interpolating at boundaries."""

    if end < start:
        raise ValueError("power window end precedes its start")
    ordered = tuple(points)
    total = 0.0
    coverage = 0.0
    for left, right in pairwise(ordered):
        t0 = _point_value(left, "monotonic_seconds")
        t1 = _point_value(right, "monotonic_seconds")
        if t0 is None or t1 is None:
            continue
        lo, hi = max(start, t0), min(end, t1)
        p0 = _point_value(left, "gpu_power_watts")
        p1 = _point_value(right, "gpu_power_watts")
        if hi <= lo or t1 <= t0 or p0 is None or p1 is None:
            continue
        power_lo = p0 + (p1 - p0) * (lo - t0) / (t1 - t0)
        power_hi = p0 + (p1 - p0) * (hi - t0) / (t1 - t0)
        total += (power_lo + power_hi) * 0.5 * (hi - lo)
        coverage += hi - lo
    return PowerWindow(total if coverage else None, coverage, end - start)


_DEVICE_REGIONS = (
    "dataset_transfer",
    "graph_input_staging",
    "transformer_graph_replay",
    "frozen_vocabulary_loss",
    "optimizer_update",
)
_IDLE_HOST_REGIONS = (
    "dataset_transfer",
    "graph_input_staging",
    "telemetry",
    "thermal_pacing",
    "python_dispatch",
)


def attribute_update_energy(
    update: Mapping[str, object],
    telemetry_points: Iterable[object],
    *,
    idle_board_watts: float,
) -> dict[str, object]:
    """Partition measured update energy using CUDA time and an idle-board model.

    CUDA synchronization is an inclusive host wait and therefore receives zero
    additional joules. The returned region estimates sum to the measured update
    window whenever energy coverage is available.
    """

    if idle_board_watts < 0:
        raise ValueError("idle board power cannot be negative")
    start, end = float(update["start"]), float(update["end"])
    window = integrate_power_window(telemetry_points, start, end)
    regions_value = update.get("regions")
    if not isinstance(regions_value, Mapping):
        raise TypeError("update regions must be a mapping")
    regions = {name: dict(value) for name, value in regions_value.items()}
    if window.joules is None:
        for value in regions.values():
            value["estimated_board_joules"] = None
        return {
            "measured_board_joules": None,
            "energy_coverage_seconds": window.covered_seconds,
            "window_seconds": window.window_seconds,
            "regions": regions,
        }

    device_seconds = {
        name: max(0.0, float(regions.get(name, {}).get("cuda_seconds", 0.0) or 0.0))
        for name in _DEVICE_REGIONS
    }
    active_seconds = sum(device_seconds.values())
    idle_seconds = max(0.0, window.window_seconds - active_seconds)
    estimated_idle_joules = min(window.joules, idle_board_watts * idle_seconds)

    host_weights: dict[str, float] = {}
    for name in _IDLE_HOST_REGIONS:
        host = max(0.0, float(regions.get(name, {}).get("host_seconds", 0.0) or 0.0))
        device = device_seconds.get(name, 0.0)
        host_weights[name] = max(0.0, host - device)
    if not any(host_weights.values()):
        host_weights["python_dispatch"] = idle_seconds or 1.0

    active_joules = window.joules - estimated_idle_joules
    device_denominator = active_seconds or 1.0
    host_denominator = sum(host_weights.values()) or 1.0
    estimates = {name: 0.0 for name in regions}
    for name, seconds in device_seconds.items():
        estimates[name] = estimates.get(name, 0.0) + active_joules * seconds / device_denominator
    for name, weight in host_weights.items():
        estimates[name] = estimates.get(name, 0.0) + estimated_idle_joules * weight / host_denominator
    for name, value in regions.items():
        value["estimated_board_joules"] = estimates.get(name, 0.0)
    if "cuda_synchronization" in regions:
        regions["cuda_synchronization"]["estimated_board_joules"] = 0.0
        regions["cuda_synchronization"]["non_additive_overlap"] = True

    return {
        "measured_board_joules": window.joules,
        "energy_coverage_seconds": window.covered_seconds,
        "window_seconds": window.window_seconds,
        "idle_board_watts_assumption": idle_board_watts,
        "regions": regions,
    }
