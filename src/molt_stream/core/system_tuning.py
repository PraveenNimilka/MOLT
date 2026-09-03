from __future__ import annotations

import atexit
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil

from molt_stream.core.specs import TrainingSpec

SUSPENDABLE_TARGETS = {
    "discord.exe",
    "spotify.exe",
    "steam.exe",
    "epicgameslauncher.exe",
}


def elevate_process_priority() -> bool:
    """Elevate process priority safely on Windows without driving CPU into thermal runaway."""
    if sys.platform != "win32":
        return False
    try:
        process = psutil.Process()
        # Above Normal gives priority over background apps without forcing all CPU cores into maximum PL2 turbo voltage
        process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        return True
    except Exception:
        return False


def apply_defender_exclusion(path: str | Path) -> bool:
    """Attempt to add a directory to Windows Defender exclusions."""
    if sys.platform != "win32":
        return False
    target = str(Path(path).resolve())
    cmd = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        f"Add-MpPreference -ExclusionPath '{target}' -ErrorAction Stop",
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def detect_background_gpu_processes() -> list[str]:
    """Detect non-MOLT applications currently holding GPU context."""
    if sys.platform != "win32":
        return []
    current_pid = os.getpid()
    found: list[str] = []
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            procs = []
            try:
                procs.extend(pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
            except Exception:
                pass
            try:
                procs.extend(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
            except Exception:
                pass
            for proc in procs:
                if proc.pid != current_pid:
                    try:
                        name = psutil.Process(proc.pid).name()
                        if name not in found:
                            found.append(name)
                    except Exception:
                        pass
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return found


def suspend_background_apps() -> list[tuple[int, str]]:
    """Suspend non-critical background apps during prioritized training."""
    if sys.platform != "win32":
        return []
    suspended: list[tuple[int, str]] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = proc.info.get("name")
            if name and name.lower() in SUSPENDABLE_TARGETS:
                p = psutil.Process(proc.info["pid"])
                p.suspend()
                suspended.append((p.pid, name))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return suspended


def resume_background_apps(suspended: list[tuple[int, str]]) -> None:
    """Resume any suspended background apps."""
    for pid, _ in suspended:
        try:
            p = psutil.Process(pid)
            p.resume()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def apply_prioritized_spec(spec: TrainingSpec) -> TrainingSpec:
    """Optimize the training specification for full ~3,000 tok/s with Pit-Stop Cooldown."""
    batch_size = spec.batch_size
    gradient_accumulation = spec.gradient_accumulation

    if batch_size == 2 and gradient_accumulation == 2 and spec.mode == "qlora":
        batch_size = 4
        gradient_accumulation = 1

    return replace(
        spec,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        thermal_control_mode="dual-gear",
        thermal_target_c=74.0,
        thermal_cruise_max_c=65.0,
        thermal_pause_seconds=0.22,
        thermal_abort_c=85.0,
    )


def activate_prioritized_tuning(spec: TrainingSpec) -> dict[str, Any]:
    """Apply all hardware, OS, and specification optimizations for Prioritized Mode."""
    priority_elevated = elevate_process_priority()
    background_gpu = detect_background_gpu_processes()
    suspended = suspend_background_apps()

    if suspended:
        atexit.register(resume_background_apps, suspended)

    defender_paths = [Path(spec.data.path).parent]
    if spec.artifacts_dir:
        defender_paths.append(Path(spec.artifacts_dir))

    defender_results = {str(p): apply_defender_exclusion(p) for p in defender_paths}
    tuned_spec = apply_prioritized_spec(spec)

    return {
        "priority_elevated": priority_elevated,
        "background_gpu": background_gpu,
        "suspended_apps": suspended,
        "defender_exclusions": defender_results,
        "tuned_spec": tuned_spec,
    }
