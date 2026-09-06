import pytest

from molt_stream.measurement.target import interpolate_nll_crossing


def test_interpolates_time_energy_and_step_to_target():
    evaluations = [
        {"step": 0, "nll": 2.0, "end_to_end_seconds": 2.0, "board_energy_joules": 10.0},
        {"step": 4, "nll": 1.0, "end_to_end_seconds": 10.0, "board_energy_joules": 50.0},
    ]
    assert interpolate_nll_crossing(evaluations, 1.5) == {
        "seconds": 6.0, "joules": 30.0, "step": 2.0,
    }


def test_rejects_nonmonotonic_measurement_axis():
    evaluations = [
        {"step": 0, "nll": 2.0, "end_to_end_seconds": 2.0, "board_energy_joules": 10.0},
        {"step": 1, "nll": 1.0, "end_to_end_seconds": 1.0, "board_energy_joules": 20.0},
    ]
    with pytest.raises(ValueError, match="nondecreasing"):
        interpolate_nll_crossing(evaluations, 1.5)


def test_returns_none_when_target_is_not_reached():
    evaluations = [
        {"step": 0, "nll": 2.0, "end_to_end_seconds": 2.0, "board_energy_joules": 10.0},
    ]
    assert interpolate_nll_crossing(evaluations, 1.5) is None
