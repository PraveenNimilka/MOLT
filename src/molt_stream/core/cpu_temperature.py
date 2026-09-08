from __future__ import annotations

import math
import sys
import threading
import time
from collections.abc import Callable

import psutil


_CPU_LABELS = ("cpu", "package", "tctl", "tdie", "core")


def _valid_temperature(value: object) -> float | None:
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return None
    return (
        temperature
        if math.isfinite(temperature) and 0.0 < temperature < 125.0
        else None
    )


def _psutil_cpu_temperature() -> float | None:
    query = getattr(psutil, "sensors_temperatures", None)
    if query is None:
        return None
    try:
        groups = query(fahrenheit=False)
    except Exception:  # sensor APIs are optional and platform-specific  # noqa: BLE001
        return None
    preferred: list[float] = []
    for group_name, entries in groups.items():
        group_is_cpu = any(token in group_name.lower() for token in _CPU_LABELS)
        for entry in entries:
            value = _valid_temperature(getattr(entry, "current", None))
            if value is None:
                continue
            label = str(getattr(entry, "label", "")).lower()
            if group_is_cpu or any(token in label for token in _CPU_LABELS):
                preferred.append(value)
    # Never describe an arbitrary motherboard/ACPI thermal zone as the CPU.
    return max(preferred) if preferred else None


def _windows_wmi_reader() -> tuple[Callable[[], float | None] | None, str | None]:
    if sys.platform != "win32":
        return None, None
    try:
        import wmi  # type: ignore[import-not-found]
    except Exception:  # optional integration  # noqa: BLE001
        return None, None
    for namespace, source in (
        (r"root\LibreHardwareMonitor", "LibreHardwareMonitor"),
        (r"root\OpenHardwareMonitor", "OpenHardwareMonitor"),
    ):
        try:
            probe = wmi.WMI(namespace=namespace)
            probe.Sensor(SensorType="Temperature")
        except Exception:  # optional provider  # noqa: BLE001, S112
            continue
        local = threading.local()

        def read(namespace=namespace, local=local) -> float | None:
            try:
                connection = getattr(local, "connection", None)
                if connection is None:
                    try:
                        import pythoncom  # type: ignore[import-not-found]

                        pythoncom.CoInitialize()
                    except ImportError:
                        pass
                    connection = wmi.WMI(namespace=namespace)
                    local.connection = connection
                values: list[float] = []
                for sensor in connection.Sensor(SensorType="Temperature"):
                    identity = " ".join(
                        str(getattr(sensor, field, ""))
                        for field in ("Name", "Identifier", "Parent")
                    ).lower()
                    if not any(token in identity for token in _CPU_LABELS):
                        continue
                    value = _valid_temperature(getattr(sensor, "Value", None))
                    if value is not None:
                        values.append(value)
                return max(values) if values else None
            except Exception:  # provider can disappear while running  # noqa: BLE001
                return None

        return read, source
    return None, None


class CpuTemperatureReader:
    """Low-overhead, truthful CPU package temperature reader.

    Windows does not expose a universal CPU temperature API. When available,
    LibreHardwareMonitor/OpenHardwareMonitor is used through its WMI provider;
    otherwise the reading remains unavailable rather than using an unrelated
    ACPI thermal-zone value.
    """

    def __init__(self, *, cache_seconds: float = 1.0) -> None:
        self.cache_seconds = max(0.1, float(cache_seconds))
        self._reader: Callable[[], float | None] = _psutil_cpu_temperature
        self.source: str | None = "psutil"
        initial = self._reader()
        if initial is None:
            windows_reader, source = _windows_wmi_reader()
            if windows_reader is not None:
                self._reader = windows_reader
                self.source = source
                initial = self._reader()
            else:
                self.source = None
        self._value = initial
        self._read_at = time.perf_counter()

    @property
    def available(self) -> bool:
        return self._value is not None

    def read(self) -> float | None:
        now = time.perf_counter()
        if now - self._read_at >= self.cache_seconds:
            self._value = self._reader()
            self._read_at = now
            if self._value is None and self.source == "psutil":
                self.source = None
        return self._value
