"""Tests for modernized MOLT CLI, workspace discovery, and profiles."""
import json
import subprocess
from pathlib import Path

import pytest

from molt_stream import cli as cli_module
from molt_stream.cli import TerminalUI, main
from molt_stream.core.discovery import (
    find_datasets,
    find_models,
    find_runs,
    get_hardware_info,
)
from molt_stream.core.profiles import apply_profile, get_profile
from molt_stream.core.specs import load_spec
from molt_stream.measurement.thermal import (
    MicroPauseThermalController,
    build_thermal_controller,
)


def test_cuda_repair_uses_uv_when_project_venv_has_no_pip(monkeypatch):
    commands = []
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *args: "y")
    monkeypatch.setattr(cli_module, "get_hardware_info", lambda: {"gpu_name": "RTX"})
    monkeypatch.setattr(cli_module.shutil, "which", lambda name: r"C:\tools\uv.exe")
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command)
        or subprocess.CompletedProcess(command, 0),
    )

    assert cli_module._verify_cuda_or_prompt_install(
        TerminalUI(enabled=True, color=False), "cuda"
    ) is False
    assert commands[0][:5] == [
        r"C:\tools\uv.exe", "pip", "install", "--python", cli_module.sys.executable,
    ]
    assert "torch==2.8.0" in commands[0]


def test_hardware_discovery_returns_expected_keys():
    hw = get_hardware_info()
    assert "os" in hw
    assert "python" in hw
    assert "cpu" in hw
    assert "total_ram_gb" in hw
    assert hw["suggested_profile"] in ("SPEED", "BALANCED", "COOL", "ENERGY")
    assert hw["suggested_memory_budget_gb"] > 0


def test_workspace_discovery_finds_assets(tmp_path: Path):
    models = find_models(search_roots=[tmp_path])
    assert isinstance(models, list)

    # Verify discovery finds binary token datasets
    (tmp_path / "sample.bin").write_bytes(b"\x00" * 1024)
    datasets = find_datasets(search_roots=[tmp_path])
    assert isinstance(datasets, list)
    assert len(datasets) == 1
    assert datasets[0]["name"] == "sample.bin"

    runs = find_runs(search_roots=[tmp_path])
    assert isinstance(runs, list)


def test_profile_mappings_and_application():
    for name in ("speed", "balanced", "cool", "energy", "micro"):
        prof = get_profile(name)
        assert "thermal_target_c" in prof
        assert "thermal_pause_seconds" in prof

    with pytest.raises(ValueError, match="Unknown training profile"):
        get_profile("nonexistent_profile")

    spec = load_spec("configs/molt-stream-smoke.json")
    cooled_spec = apply_profile(spec, "cool")
    assert cooled_spec.thermal_target_c == 68.0
    assert cooled_spec.thermal_pause_seconds == 0.30

    speed_spec = apply_profile(spec, "speed")
    assert speed_spec.thermal_target_c == 82.0
    assert speed_spec.thermal_pause_seconds == 0.05
    micro_spec = apply_profile(spec, "micro")
    assert micro_spec.thermal_control_mode == "micro-guard"
    assert micro_spec.thermal_microbatch_guard_c == 83.0
    assert micro_spec.thermal_target_c == 78.0
    assert micro_spec.thermal_cruise_max_c == 82.0
    assert micro_spec.min_pause_ms == 1.0
    assert micro_spec.thermal_power_target_watts == 48.0
    micro_controller = build_thermal_controller(micro_spec)
    assert isinstance(micro_controller, MicroPauseThermalController)
    assert micro_controller.power_control_start_c == 65.0
    assert micro_controller.target_average_watts == 48.0
    assert micro_controller.pauses[-1] == pytest.approx(0.1)


@pytest.mark.parametrize("output_mode", ["--json", "--ui"])
def test_cli_inspect_runs_real_capability_probe(capsys, output_mode):
    assert main([output_mode, "inspect"]) == 0
    captured = capsys.readouterr()
    if output_mode == "--json":
        result = json.loads(captured.out)
        assert "cuda_available" in result
        assert "python" in result
    else:
        assert "Workspace info" in captured.out
    assert "ImportError" not in captured.out + captured.err


def test_cli_info_json(capsys):
    ret = main(["--json", "info"])
    assert ret == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "os" in data
    assert "python" in data


def test_cli_train_dry_run(capsys, smoke_config):
    ret = main([
        "--json", "train",
        "--config", str(smoke_config),
        "--profile", "balanced",
        "--dry-run",
    ])
    assert ret == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["status"] == "dry_run_success"
    assert data["spec"]["thermal_target_c"] == 74.0


@pytest.mark.parametrize("state,expected", [("completed", 0), ("thermal_abort", 1), ("interrupted", 1)])
def test_cli_train_exit_code_tracks_run_state(
    tmp_path, smoke_config, monkeypatch, capsys, state, expected
):
    root = tmp_path / "run"
    root.mkdir()
    (root / "metrics.summary.json").write_text(
        json.dumps({
            "state": state,
            "session_tokens": 0,
            "telemetry": {"gpu_board_energy_joules": None},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_module, "_verify_cuda_or_prompt_install", lambda *_args: True)
    monkeypatch.setattr("molt_stream.training.engine.train", lambda *_args, **_kwargs: root)
    assert main(["--json", "train", "--config", str(smoke_config), "-y"]) == expected
    assert json.loads(capsys.readouterr().out)["run"] == str(root)


def test_cli_config_list(capsys):
    ret = main(["--json", "config", "--list"])
    assert ret == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "models" in data
    assert "datasets" in data
    assert "profiles" in data


def test_cli_dry_run_rejects_missing_dataset(capsys, smoke_config):
    config = json.loads(smoke_config.read_text("utf-8"))
    missing = smoke_config.parent / "missing.bin"
    config["data"]["path"] = str(missing)
    smoke_config.write_text(json.dumps(config), encoding="utf-8")
    assert main(["--json", "train", "--config", str(smoke_config), "--dry-run"]) == 1
    assert "missing.bin" in capsys.readouterr().err


def test_cli_benchmark_smoke_execution(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("CUDA device required for physical smoke benchmark")
    monkeypatch.chdir(tmp_path)

    ret = main(["--json", "benchmark", "--smoke"])
    assert ret == 0


@pytest.mark.parametrize("state,expected", [("completed", 0), ("thermal_abort", 1)])
def test_guided_smoke_is_self_contained(tmp_path, monkeypatch, capsys, state, expected):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *args: "3")
    def fake_train(spec, **kwargs):
        spec.validate()
        assert spec.max_steps == 20
        assert Path(spec.data.path).read_bytes() != Path(spec.data.validation_path).read_bytes()
        root = Path(spec.artifacts_dir)
        root.mkdir()
        (root / "metrics.summary.json").write_text(json.dumps({
            "state": state, "session_tokens": 5120, "telemetry": {"gpu_board_energy_joules": None}
        }), encoding="utf-8")
        return root
    monkeypatch.setattr("molt_stream.training.engine.train", fake_train)
    assert main(["--ui"]) == expected
    output = capsys.readouterr().out
    assert ("INCOMPLETE" if expected else "PASS") in output
    assert "Unavailable" in output


def test_guided_errors_use_normal_error_handler(monkeypatch, capsys):
    def failed_menu(ui):
        raise FileNotFoundError("example missing file")
    monkeypatch.setattr("molt_stream.cli.guided_landing", failed_menu)
    assert main(["--ui"]) == 1
    assert "example missing file" in capsys.readouterr().out


def test_unsloth_strictly_excluded_from_repository_and_dependencies():
    pyproject_text = Path("pyproject.toml").read_text("utf-8").lower()
    assert "unsloth" not in pyproject_text

    # Verify unsloth is not imported in python runtime
    import sys
    assert "unsloth" not in sys.modules


def test_cli_guided_landing_menu(monkeypatch):
    from io import StringIO

    monkeypatch.setattr("sys.stdin", StringIO("6\n"))
    ret = main(["--ui"])
    assert ret == 0
