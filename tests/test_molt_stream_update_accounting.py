import pytest

from molt_stream.measurement.update_accounting import UpdateAccounting


def test_rollback_work_reduces_useful_compute_rate():
    accounting = UpdateAccounting()
    accounting.record(elapsed_seconds=5, pause_seconds=2, tokens=512, committed=False)
    accounting.record(elapsed_seconds=4, pause_seconds=1, tokens=2048, committed=True)
    result = accounting.summary()
    assert result["update_attempts"] == 2
    assert result["discarded_update_compute_seconds"] == 3
    assert result["committed_update_compute_seconds"] == 3
    assert result["update_compute_seconds"] == 6
    assert result["update_compute_tokens_per_second"] == pytest.approx(2048 / 6)
    assert result["committed_update_compute_tokens_per_second"] == pytest.approx(2048 / 3)
    assert accounting.committed_tokens == 2048
    assert accounting.discarded_tokens == 512


def test_empty_accounting_does_not_manufacture_speed():
    assert UpdateAccounting().summary()["update_compute_tokens_per_second"] is None


@pytest.mark.parametrize("elapsed,pause,tokens", [
    (float("nan"), 0, 1), (float("inf"), 0, 1), (1, 2, 1),
    (1, -1, 1), (1, 0, -1), (1, 0, True),
])
def test_invalid_accounting_is_rejected(elapsed, pause, tokens):
    with pytest.raises(ValueError):
        UpdateAccounting().record(elapsed_seconds=elapsed, pause_seconds=pause,
                                  tokens=tokens, committed=True)
