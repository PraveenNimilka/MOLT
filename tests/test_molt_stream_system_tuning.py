"""Host tuning tests never modify actual Windows settings."""
import json
import pytest

from molt_stream.core import system_tuning
from molt_stream.core.system_tuning import prioritized_execution
from molt_stream.measurement.thermal import cooling_pause
from molt_stream.cli import main
from molt_stream.core.specs import load_spec
from molt_stream.measurement.thermal import DualGearThermalController, TelemetryPoint


def point(temperature: float) -> TelemetryPoint:
    return TelemetryPoint(0.0, 1, 50.0, None, temperature, None, None, None)


def test_dual_gear_thermal_controller_behavior():
    ctrl = DualGearThermalController(
        shift_down_c=74.0,
        shift_up_c=65.0,
        gear1_pause_seconds=0.0,
        gear2_pause_seconds=0.22,
        abort_c=85.0,
    )
    # Below 74°C: Gear 1 (0ms pause, ~3,000 tok/s)
    d1 = ctrl.update(point(70.0), step_seconds=0.34)
    assert d1.pause_seconds == 0.0
    assert d1.phase == "gear-1-sprint"
    assert not d1.abort

    # At 74°C: Shift down to Gear 2 (220ms pause, ~1,800 tok/s)
    d2 = ctrl.update(point(74.0), step_seconds=0.34)
    assert d2.pause_seconds == 0.22
    assert d2.phase == "gear-2-cooldown"
    assert not d2.abort

    # At 70°C: Still cooling in Gear 2
    d3 = ctrl.update(point(70.0), step_seconds=0.56)
    assert d3.pause_seconds == 0.22
    assert d3.phase == "gear-2-cooldown"

    # Drops below 65°C: Shift back up to Gear 1 (0ms pause, ~3,000 tok/s!)
    d4 = ctrl.update(point(64.5), step_seconds=0.56)
    assert d4.pause_seconds == 0.0
    assert d4.phase == "gear-1-sprint"

    # Emergency abort if ever at abort_c
    d5 = ctrl.update(point(85.0), step_seconds=0.34)
    assert d5.abort
    assert d5.phase == "thermal-abort"


@pytest.mark.parametrize("fails", [False, True])
def test_priority_is_restored(monkeypatch, fails):
    calls = []
    class Process:
        def nice(self, value=None):
            if value is None:
                return 32
            calls.append(value)
    monkeypatch.setattr(system_tuning.sys, "platform", "win32")
    monkeypatch.setattr(system_tuning.psutil, "Process", Process)
    monkeypatch.setattr(system_tuning.psutil, "ABOVE_NORMAL_PRIORITY_CLASS", 32768, raising=False)
    try:
        with prioritized_execution(True) as changed:
            assert changed
            if fails:
                raise ValueError("training failed")
    except ValueError:
        assert fails
    assert calls == [32768, 32]


def test_normal_mode_does_not_touch_process(monkeypatch):
    def forbidden():
        pytest.fail("Unexpected host tuning")
    monkeypatch.setattr(system_tuning.psutil, "Process", forbidden)
    with prioritized_execution(False) as changed:
        assert not changed


def test_denied_priority_still_runs(monkeypatch):
    monkeypatch.setattr(system_tuning.sys, "platform", "win32")
    def denied():
        raise system_tuning.psutil.AccessDenied()
    monkeypatch.setattr(system_tuning.psutil, "Process", denied)
    with pytest.warns(RuntimeWarning, match="priority unavailable"):
        with prioritized_execution(True) as changed:
            assert not changed


def test_prioritized_dry_run_never_activates_host_tuning(monkeypatch, capsys, smoke_config):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run activated host tuning")
    monkeypatch.setattr("molt_stream.cli.prioritized_execution", forbidden)
    assert main(["--json", "train", "--config", str(smoke_config),
                 "--mode-select", "prioritize", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run_success"
    assert result["spec"] == load_spec(smoke_config).to_dict()


def test_cancelled_training_never_activates_tuning(monkeypatch, capsys, smoke_config):
    def forbidden(*args, **kwargs):
        pytest.fail("Cancelled training activated tuning")
    monkeypatch.setattr("molt_stream.cli.prioritized_execution", forbidden)
    monkeypatch.setattr("molt_stream.cli._verify_cuda_or_prompt_install", lambda *args: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *args: "n")
    assert main(["--ui", "train", "--config", str(smoke_config),
                 "--mode-select", "prioritize"]) == 0
    assert "cancelled" in capsys.readouterr().out


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("temperatures,expected", [
    ([70.0, 60.0], 1.0), ([70.0, 85.0], 1.0), ([None] * 8, 4.0)
])
def test_cooling_policy_and_actual_accounting(monkeypatch, interactive, temperatures, expected):
    clock = [0.0]
    samples = iter(temperatures)
    notices = []
    monkeypatch.setattr("molt_stream.measurement.thermal.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("molt_stream.measurement.thermal.time.sleep",
                        lambda delay: clock.__setitem__(0, clock[0] + delay))
    elapsed = cooling_pause(4.0, lambda: next(samples), recovery_c=65.0, abort_c=85.0,
                            notify=(lambda *args: notices.append(args)) if interactive else None)
    assert elapsed == expected
    assert bool(notices) == interactive


def test_micro_pause_reports_sleep_overshoot(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("molt_stream.measurement.thermal.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("molt_stream.measurement.thermal.time.sleep",
                        lambda delay: clock.__setitem__(0, clock[0] + delay + 0.01))
    assert cooling_pause(0.04, lambda: 60.0, recovery_c=65.0, abort_c=85.0) == pytest.approx(0.05)
