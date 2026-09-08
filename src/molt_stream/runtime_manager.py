"""Invoke the packaged per-user Windows runtime manager."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_ACTIONS = {"update": "Update", "repair": "Repair", "uninstall": "Uninstall"}


def runtime_home() -> Path:
    override = os.environ.get("MOLT_HOME")
    if override:
        return Path(override).resolve()
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise RuntimeError("LOCALAPPDATA is unavailable")
    return Path(local) / "MOLT"


def packaged_installer() -> Path:
    return Path(__file__).with_name("resources") / "install-global.ps1"


def stage_installer() -> Path:
    source = packaged_installer()
    if not source.is_file():
        raise RuntimeError(f"Packaged runtime installer is missing: {source}")
    destination = runtime_home() / "installer.ps1"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def run_action(action: str, *, plan: bool = False) -> int:
    """Run one management action synchronously through Windows PowerShell."""

    if action not in _ACTIONS:
        raise ValueError(f"unsupported runtime action: {action}")
    if sys.platform != "win32":
        raise RuntimeError("MOLT runtime management currently supports Windows only")
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        raise RuntimeError("Windows PowerShell is unavailable")
    command = [
        powershell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(stage_installer()),
        "-Action",
        _ACTIONS[action],
    ]
    if plan:
        command.append("-Plan")
    return subprocess.run(command, check=False).returncode
