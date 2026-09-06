from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from molt_stream import setup_check


@pytest.mark.parametrize("eager", [False, True])
def test_installer_plan_is_locked_and_scoped(eager):
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell required")
    command = [powershell, "-NoProfile", "-File", "install.ps1", "-Plan"]
    if eager:
        command.append("-EagerOnly")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert "--locked" in plan["sync"]
    assert "qlora" in plan["sync"]
    assert ("windows-fusion" in plan["sync"]) is not eager
    assert ("--compile" in plan["check"]) is not eager
    assert plan["legacy_distribution"] == "molt-ai-infrastructure"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_installer_propagates_failure_without_real_downloads(tmp_path, exit_code):
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell required")
    shutil.copyfile("install.ps1", tmp_path / "install.ps1")
    (tmp_path / "uv.lock").write_text("# mock", encoding="utf-8")
    (tmp_path / "uv.cmd").write_text(
        f"@echo off\necho %*>>calls.log\nexit /b {exit_code}\n", encoding="ascii"
    )
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
                             "-File", str(tmp_path / "install.ps1")],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert (result.returncode == 0) is (exit_code == 0)
    calls = (tmp_path / "calls.log").read_text().splitlines()
    assert len(calls) == (2 if exit_code == 0 else 1)
    assert "sync --locked" in calls[0]


def test_installer_migrates_legacy_distribution_without_deleting_source(tmp_path):
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        pytest.skip("Windows PowerShell required")
    shutil.copyfile("install.ps1", tmp_path / "install.ps1")
    (tmp_path / "uv.lock").write_text("# mock", encoding="utf-8")
    python = tmp_path / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    (tmp_path / "uv.cmd").write_text(
        "@echo off\n"
        "echo %*>>calls.log\n"
        "if \"%1 %2\"==\"pip show\" echo Name: molt-ai-infrastructure\n"
        "exit /b 0\n",
        encoding="ascii",
    )
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(tmp_path / "install.ps1")],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    calls = (tmp_path / "calls.log").read_text().splitlines()
    assert calls[0].startswith("pip show --python")
    assert calls[1].startswith("pip uninstall --python")
    assert "molt-ai-infrastructure" in calls[1]
    assert calls[2].startswith("sync --locked")
    assert "--reinstall-package moltengine" in calls[2]
    assert calls[3].startswith("run --no-sync")
    assert (tmp_path / "install.ps1").is_file()


def test_setup_reports_errors_without_claiming_success(monkeypatch, capsys):
    def fail(**kwargs):
        raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(setup_check, "check_runtime", fail)
    assert setup_check.main([]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_setup_forwards_compile_choice(monkeypatch, capsys):
    def check(*, compile_enabled):
        assert compile_enabled
        return {"compile_backward": "passed"}
    monkeypatch.setattr(setup_check, "check_runtime", check)
    assert setup_check.main(["--compile"]) == 0
    assert json.loads(capsys.readouterr().out)["compile_backward"] == "passed"
