import pytest

from molt_stream.measurement.closed_loop_pacing import ClosedLoopPacer
from molt_stream.training.memory_tiers import plan_resident_tier


def test_hot_rising_load_reduces_duty_without_touching_training():
    pacer = ClosedLoopPacer()
    cold = pacer.delay(1, 60, 65, .13, .14)
    hot = pacer.delay(2, 74, 90, .13, .14)
    assert hot > cold >= 0


def test_elapsed_host_work_is_credited():
    assert ClosedLoopPacer().delay(1, 60, 50, .13, 10) == 0


def test_invalid_telemetry_fails_closed():
    for temperature in (float("nan"), 84):
        with pytest.raises(ValueError):
            ClosedLoopPacer().delay(1, temperature, 70, .13, .14)


def test_power_feedback_and_bounds():
    pacer = ClosedLoopPacer()
    for now in range(1, 100):
        pacer.delay(now, 65, 120, .13, .14)
    assert 0.05 <= pacer.power_duty < 0.1
    assert 0.05 <= pacer.thermal_duty <= 1


def test_bad_time_and_power_rejected():
    pacer = ClosedLoopPacer()
    pacer.delay(1, 60, 70, .13, .14)
    with pytest.raises(ValueError):
        pacer.delay(1, 60, 70, .13, .14)
    with pytest.raises(ValueError):
        ClosedLoopPacer().delay(1, 60, float("nan"), .13, .14)


def test_small_tier_never_enables_checkpointing():
    assert not plan_resident_tier(1_500_000_000, 8 * 2**30).activation_checkpointing
    assert not plan_resident_tier(3_000_000_000, 8 * 2**30).activation_checkpointing
    plan = plan_resident_tier(8_000_000_000, 8 * 2**30)
    assert plan.activation_checkpointing and not plan.qualified
    assert plan.reserve_bytes >= 512 * 2**20


def test_tier_rejects_missing_reserve():
    with pytest.raises(ValueError):
        plan_resident_tier(1_500_000_000, 256 * 2**20)
