import pytest

from molt_stream.measurement.step_windows import step_window_rates


def test_window_speed_captures_late_slowdown():
    result = step_window_rates([1.0] * 4 + [2.0] * 4, 100)
    assert result["first_quarter_tokens_per_second"] == 100
    assert result["last_quarter_tokens_per_second"] == 50
    assert result["last_to_first_rate_ratio"] == 0.5


def test_too_short_window_does_not_claim_stability():
    assert all(value is None for value in step_window_rates([1.0], 100).values())


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_bad_intervals_rejected(interval):
    with pytest.raises(ValueError):
        step_window_rates([interval], 100)
