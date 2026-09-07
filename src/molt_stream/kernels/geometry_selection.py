from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryMeasurement:
    name: str
    compute_tokens_per_second: float
    board_memory_bytes: int
    peak_temperature_c: float
    quality_error_percent: float
    late_to_early_rate: float | None
    completed: bool


@dataclass(frozen=True)
class GeometryGate:
    maximum_board_memory_bytes: int
    maximum_temperature_c: float
    maximum_quality_error_percent: float = 1.0
    minimum_late_to_early_rate: float = 0.95


def rejection_reasons(
    measurement: GeometryMeasurement, gate: GeometryGate
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not measurement.completed:
        reasons.append("run did not complete")
    if measurement.compute_tokens_per_second <= 0:
        reasons.append("throughput is not positive")
    if measurement.board_memory_bytes > gate.maximum_board_memory_bytes:
        reasons.append("board memory exceeds gate")
    if measurement.peak_temperature_c > gate.maximum_temperature_c:
        reasons.append("temperature exceeds gate")
    if measurement.quality_error_percent > gate.maximum_quality_error_percent:
        reasons.append("quality error exceeds gate")
    if measurement.late_to_early_rate is None:
        reasons.append("long-run stability is unmeasured")
    elif measurement.late_to_early_rate < gate.minimum_late_to_early_rate:
        reasons.append("late-run throughput stability is below gate")
    return tuple(reasons)


def select_fastest_eligible(
    measurements: tuple[GeometryMeasurement, ...], gate: GeometryGate
) -> GeometryMeasurement | None:
    """Select only among measured candidates that satisfy every hard gate."""

    eligible = [
        item for item in measurements if not rejection_reasons(item, gate)
    ]
    return max(eligible, key=lambda item: item.compute_tokens_per_second, default=None)
