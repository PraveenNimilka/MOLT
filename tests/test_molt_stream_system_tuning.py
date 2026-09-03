import sys
from pathlib import Path
import pytest

from molt_stream.core.specs import TrainingSpec, ModelSpec, DataSpec
from molt_stream.core.system_tuning import (
    elevate_process_priority,
    apply_defender_exclusion,
    detect_background_gpu_processes,
    suspend_background_apps,
    resume_background_apps,
    apply_prioritized_spec,
    activate_prioritized_tuning,
)
from molt_stream.measurement.thermal import DualGearThermalController, TelemetryPoint
from molt_stream.cli import _resolve_execution_mode, TerminalUI


def point(temperature: float) -> TelemetryPoint:
    return TelemetryPoint(0.0, 1, 50.0, None, temperature, None, None, None)


def test_elevate_process_priority_runs_safely():
    result = elevate_process_priority()
    assert isinstance(result, bool)


def test_apply_defender_exclusion_runs_safely(tmp_path):
    result = apply_defender_exclusion(tmp_path)
    assert isinstance(result, bool)


def test_detect_background_gpu_processes_returns_list():
    procs = detect_background_gpu_processes()
    assert isinstance(procs, list)


def test_suspend_and_resume_background_apps_safely():
    suspended = suspend_background_apps()
    assert isinstance(suspended, list)
    resume_background_apps(suspended)


def test_apply_prioritized_spec_configures_intercooler():
    spec = TrainingSpec(
        mode="qlora",
        data=DataSpec(path="fake.bin", context_length=256),
        model=ModelSpec(layers=1, width=64, context_length=256),
        batch_size=2,
        gradient_accumulation=2,
        max_steps=10,
    )
    tuned = apply_prioritized_spec(spec)
    assert tuned.thermal_control_mode == "dual-gear"
    assert tuned.thermal_target_c == 74.0
    assert tuned.thermal_cruise_max_c == 65.0
    assert tuned.thermal_pause_seconds == 0.22
    assert tuned.thermal_abort_c == 85.0
    assert tuned.batch_size == 4
    assert tuned.gradient_accumulation == 1


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


def test_activate_prioritized_tuning_returns_structured_dict(tmp_path):
    train_bin = tmp_path / "train.bin"
    train_bin.write_bytes(b"0" * 1024)
    spec = TrainingSpec(
        mode="qlora",
        data=DataSpec(path=str(train_bin), context_length=256),
        model=ModelSpec(layers=1, width=64, context_length=256),
        batch_size=1,
        gradient_accumulation=1,
        max_steps=10,
        artifacts_dir=str(tmp_path / "runs"),
    )
    res = activate_prioritized_tuning(spec)
    assert "priority_elevated" in res
    assert "background_gpu" in res
    assert "suspended_apps" in res
    assert "defender_exclusions" in res
    assert "tuned_spec" in res
    assert res["tuned_spec"].thermal_control_mode == "dual-gear"
    assert res["tuned_spec"].thermal_target_c == 74.0


def test_resolve_execution_mode_non_interactive():
    ui = TerminalUI(enabled=False, color=False)
    assert _resolve_execution_mode(None, ui) == "normal"
    assert _resolve_execution_mode("prioritize", ui) == "prioritize"
    assert _resolve_execution_mode("normal", ui) == "normal"
