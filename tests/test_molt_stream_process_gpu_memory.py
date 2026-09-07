from __future__ import annotations

from types import SimpleNamespace

from molt_stream.measurement.telemetry import _process_gpu_memory_bytes


def test_process_gpu_memory_deduplicates_compute_and_graphics_lists() -> None:
    current = SimpleNamespace(pid=42, usedGpuMemory=800)
    other = SimpleNamespace(pid=7, usedGpuMemory=2_000)
    nvml = SimpleNamespace(
        nvmlDeviceGetComputeRunningProcesses_v3=lambda _handle: [current, other],
        nvmlDeviceGetGraphicsRunningProcesses_v3=lambda _handle: [
            SimpleNamespace(pid=42, usedGpuMemory=900)
        ],
    )
    assert _process_gpu_memory_bytes(nvml, object(), 42) == 900


def test_process_gpu_memory_rejects_wddm_sentinel_and_unsupported_queries() -> None:
    def unavailable(_handle):
        raise RuntimeError("not supported")

    nvml = SimpleNamespace(
        nvmlDeviceGetComputeRunningProcesses_v3=lambda _handle: [
            SimpleNamespace(pid=42, usedGpuMemory=2**64 - 1)
        ],
        nvmlDeviceGetGraphicsRunningProcesses_v3=unavailable,
    )
    assert _process_gpu_memory_bytes(nvml, object(), 42) is None
