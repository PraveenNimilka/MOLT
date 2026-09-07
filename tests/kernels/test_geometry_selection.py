from __future__ import annotations

from molt_stream.kernels.geometry_selection import (
    GeometryGate,
    GeometryMeasurement,
    rejection_reasons,
    select_fastest_eligible,
)


GIB = 1024 ** 3


def _measurement(name: str, speed: float, **overrides):
    values = {
        "name": name,
        "compute_tokens_per_second": speed,
        "board_memory_bytes": 2 * GIB,
        "peak_temperature_c": 70.0,
        "quality_error_percent": 0.2,
        "late_to_early_rate": 0.97,
        "completed": True,
    }
    values.update(overrides)
    return GeometryMeasurement(**values)


def test_selector_chooses_fastest_candidate_that_passes_every_gate() -> None:
    gate = GeometryGate(3 * GIB, 72.0)
    accepted = _measurement("stable", 1400.0)
    too_hot = _measurement("sprint", 2000.0, peak_temperature_c=75.0)
    assert select_fastest_eligible((accepted, too_hot), gate) == accepted


def test_short_screen_cannot_be_promoted_without_stability_evidence() -> None:
    gate = GeometryGate(3 * GIB, 72.0)
    short = _measurement("short", 1800.0, late_to_early_rate=None)
    assert "long-run stability is unmeasured" in rejection_reasons(short, gate)
    assert select_fastest_eligible((short,), gate) is None


def test_quality_memory_and_completion_fail_closed() -> None:
    gate = GeometryGate(3 * GIB, 72.0)
    failed = _measurement(
        "failed",
        1800.0,
        board_memory_bytes=4 * GIB,
        quality_error_percent=1.1,
        completed=False,
    )
    reasons = rejection_reasons(failed, gate)
    assert "run did not complete" in reasons
    assert "board memory exceeds gate" in reasons
    assert "quality error exceeds gate" in reasons
