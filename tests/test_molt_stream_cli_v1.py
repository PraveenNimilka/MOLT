"""Tests for modernized MOLT CLI, workspace discovery, and profiles."""
import json
from pathlib import Path
import pytest

from molt_stream.cli import main
from molt_stream.core.discovery import (
    find_datasets,
    find_models,
    find_runs,
    get_hardware_info,
    init_workspace,
)
from molt_stream.core.profiles import PROFILES, apply_profile, get_profile
from molt_stream.core.specs import load_spec


def test_hardware_discovery_returns_expected_keys():
    hw = get_hardware_info()
    assert "os" in hw
    assert "python" in hw
    assert "cpu" in hw
    assert "total_ram_gb" in hw
    assert hw["suggested_profile"] in ("SPEED", "BALANCED", "COOL", "ENERGY")
    assert hw["suggested_memory_budget_gb"] > 0


def test_workspace_discovery_finds_assets():
    models = find_models()
    assert isinstance(models, list)
    datasets = find_datasets()
    assert isinstance(datasets, list)
    assert len(datasets) > 0  # At least smoke datasets exist in data/prepared
    runs = find_runs()
    assert isinstance(runs, list)


def test_profile_mappings_and_application():
    for name in ("speed", "balanced", "cool", "energy"):
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


def test_cli_info_json(capsys):
    ret = main(["--json", "info"])
    assert ret == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "os" in data
    assert "python" in data


def test_cli_train_dry_run(capsys):
    ret = main([
        "--json", "train",
        "--config", "configs/molt-stream-smoke.json",
        "--profile", "balanced",
        "--dry-run",
    ])
    assert ret == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["status"] == "dry_run_success"
    assert data["spec"]["thermal_target_c"] == 74.0


def test_cli_config_list(capsys):
    ret = main(["--json", "config", "--list"])
    assert ret == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "models" in data
    assert "datasets" in data
    assert "profiles" in data


def test_cli_benchmark_smoke_execution():
    ret = main(["--json", "benchmark", "--smoke"])
    assert ret == 0


def test_unsloth_strictly_excluded_from_repository_and_dependencies():
    pyproject_text = Path("pyproject.toml").read_text("utf-8").lower()
    assert "unsloth" not in pyproject_text

    # Verify unsloth is not imported in python runtime
    import sys
    assert "unsloth" not in sys.modules
