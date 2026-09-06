import subprocess

import pytest

from molt_stream.core.errors import CapabilityError
from molt_stream.measurement import gpu_profile


def _result(code: int = 0, output: str = "All done.") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, output, "")


def test_clock_profile_is_always_restored(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_profile, "is_windows_administrator", lambda: True)
    monkeypatch.setattr(
        gpu_profile,
        "_nvidia_smi",
        lambda *args: calls.append(args) or _result(),
    )
    with pytest.raises(RuntimeError):
        with gpu_profile.temporary_graphics_clock(gpu_profile.PROFILES["endurance"]):
            raise RuntimeError("training failed")
    assert calls == [("-lgc", "1500,1650"), ("-rgc",)]


def test_clock_profile_requires_administrator(monkeypatch):
    monkeypatch.setattr(gpu_profile, "is_windows_administrator", lambda: False)
    with pytest.raises(CapabilityError, match="Administrator"):
        with gpu_profile.temporary_graphics_clock(gpu_profile.PROFILES["endurance"]):
            pass


def test_driver_refusal_is_not_treated_as_success(monkeypatch):
    monkeypatch.setattr(gpu_profile, "is_windows_administrator", lambda: True)
    monkeypatch.setattr(
        gpu_profile,
        "_nvidia_smi",
        lambda *args: _result(output="Changing clocks is not supported in current scope"),
    )
    with pytest.raises(CapabilityError, match="failed"):
        with gpu_profile.temporary_graphics_clock(gpu_profile.PROFILES["endurance"]):
            pass


def test_measured_clock_verification():
    result = gpu_profile.verify_measured_clock_profile(
        {"minimum_gpu_graphics_clock_mhz": 1500, "maximum_gpu_graphics_clock_mhz": 1650},
        gpu_profile.PROFILES["endurance"],
    )
    assert result["verified"] is True
