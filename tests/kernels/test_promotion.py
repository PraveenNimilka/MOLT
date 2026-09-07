from __future__ import annotations

import pytest

from molt_stream.kernels.promotion import KernelMeasurement, evaluate_kernel_candidate


def test_candidate_must_pass_every_gate() -> None:
    reference = KernelMeasurement(10.0, 100)
    accepted = evaluate_kernel_candidate(
        reference,
        KernelMeasurement(8.0, 90),
        parity_passed=True,
    )
    assert accepted.promoted
    assert accepted.speedup_percent == pytest.approx(20.0)
    assert accepted.memory_change_percent == pytest.approx(-10.0)


@pytest.mark.parametrize(
    ("candidate", "parity", "reason"),
    [
        (KernelMeasurement(8.0, 90), False, "parity"),
        (KernelMeasurement(9.8, 90), True, "speedup"),
        (KernelMeasurement(8.0, 101), True, "memory growth"),
    ],
)
def test_candidate_fails_closed(candidate, parity: bool, reason: str) -> None:
    decision = evaluate_kernel_candidate(
        KernelMeasurement(10.0, 100), candidate, parity_passed=parity
    )
    assert not decision.promoted
    assert reason in decision.reason


def test_invalid_measurements_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        KernelMeasurement(0.0, 1)
    with pytest.raises(ValueError, match="negative"):
        KernelMeasurement(1.0, -1)
