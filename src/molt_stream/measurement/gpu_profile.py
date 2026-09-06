"""Explicit, reversible NVIDIA graphics-clock profiles for measured runs."""
from __future__ import annotations

import contextlib
import ctypes
import subprocess
from dataclasses import dataclass
from typing import Iterator

from molt_stream.core.errors import CapabilityError


@dataclass(frozen=True)
class GPUClockProfile:
    name: str
    minimum_mhz: int
    maximum_mhz: int

    def __post_init__(self) -> None:
        if self.minimum_mhz <= 0 or self.maximum_mhz < self.minimum_mhz:
            raise ValueError("GPU clock profile requires 0 < minimum <= maximum")


PROFILES = {"endurance": GPUClockProfile("endurance", 1500, 1650)}


def is_windows_administrator() -> bool:
    """Return whether the current Windows process is elevated."""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


def _nvidia_smi(*arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["nvidia-smi", *arguments], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise CapabilityError(f"nvidia-smi is unavailable: {exc}") from exc


def _require_success(result: subprocess.CompletedProcess[str], operation: str) -> None:
    message = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    refused = "not supported" in message.lower() or "insufficient permission" in message.lower()
    if result.returncode or refused:
        raise CapabilityError(f"NVIDIA {operation} failed: {message or 'unknown error'}")


@contextlib.contextmanager
def temporary_graphics_clock(profile: GPUClockProfile) -> Iterator[GPUClockProfile]:
    """Apply a clock range and unconditionally restore automatic clocks."""
    if not is_windows_administrator():
        raise CapabilityError("optimize-gpu requires an Administrator PowerShell")
    applied = False
    try:
        result = _nvidia_smi("-lgc", f"{profile.minimum_mhz},{profile.maximum_mhz}")
        _require_success(result, "graphics-clock lock")
        applied = True
        yield profile
    finally:
        if applied:
            reset = _nvidia_smi("-rgc")
            _require_success(reset, "graphics-clock restoration")


def verify_measured_clock_profile(
    telemetry: dict[str, object], profile: GPUClockProfile
) -> dict[str, object]:
    """Verify the completed run actually sampled only the requested clock range."""
    minimum = telemetry.get("minimum_gpu_graphics_clock_mhz")
    maximum = telemetry.get("maximum_gpu_graphics_clock_mhz")
    verified = (
        isinstance(minimum, (int, float))
        and isinstance(maximum, (int, float))
        and minimum >= profile.minimum_mhz
        and maximum <= profile.maximum_mhz
    )
    return {
        "profile": profile.name,
        "requested_minimum_mhz": profile.minimum_mhz,
        "requested_maximum_mhz": profile.maximum_mhz,
        "measured_minimum_mhz": minimum,
        "measured_maximum_mhz": maximum,
        "verified": verified,
    }
