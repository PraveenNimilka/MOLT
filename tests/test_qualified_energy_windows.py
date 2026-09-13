import pytest

from benchmarks.qualified_screen_summary import window_energy


def test_energy_clips_and_interpolates_phase_boundaries():
    points = [{"monotonic_seconds": t, "gpu_power_watts": p} for t, p in [(0, 10), (2, 30), (4, 10)]]
    value = window_energy(points, 1, 3)
    assert value["joules"] == pytest.approx(50)
    assert value["covered_seconds"] == 2


def test_missing_power_is_not_zero_energy():
    points = [{"monotonic_seconds": t, "gpu_power_watts": p} for t, p in [(0, None), (1, 10)]]
    assert window_energy(points, 0, 1)["joules"] is None
