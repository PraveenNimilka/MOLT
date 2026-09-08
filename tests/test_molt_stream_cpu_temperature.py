from molt_stream.core import cpu_temperature
from molt_stream.measurement.telemetry import NVMLTelemetry


def test_cpu_temperature_reader_uses_truthful_cpu_sensor(monkeypatch):
    monkeypatch.setattr(cpu_temperature, "_psutil_cpu_temperature", lambda: 63.5)
    reader = cpu_temperature.CpuTemperatureReader(cache_seconds=10.0)
    assert reader.available
    assert reader.read() == 63.5
    assert reader.source == "psutil"


def test_cpu_temperature_reader_reports_unavailable(monkeypatch):
    monkeypatch.setattr(cpu_temperature, "_psutil_cpu_temperature", lambda: None)
    monkeypatch.setattr(cpu_temperature, "_windows_wmi_reader", lambda: (None, None))
    reader = cpu_temperature.CpuTemperatureReader(cache_seconds=10.0)
    assert not reader.available
    assert reader.read() is None
    assert reader.source is None


def test_telemetry_records_cpu_temperature_summary():
    class Reader:
        source = "test-sensor"

        @staticmethod
        def read():
            return 62.0

    telemetry = NVMLTelemetry(
        interval_seconds=0.01,
        enable_gpu=False,
        cpu_temperature_reader=Reader(),
    )
    telemetry.start()
    telemetry._stop.wait(0.03)
    summary = telemetry.stop()
    assert summary["peak_cpu_temperature_c"] == 62.0
    assert summary["mean_cpu_temperature_c"] == 62.0
    assert summary["cpu_temperature_source"] == "test-sensor"
