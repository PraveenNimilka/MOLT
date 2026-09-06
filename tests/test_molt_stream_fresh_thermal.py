from types import SimpleNamespace

from molt_stream.measurement.telemetry import NVMLTelemetry


def test_fresh_thermal_read_does_not_pollute_energy_samples():
    telemetry = NVMLTelemetry(enable_gpu=False)
    telemetry._handle = object()
    telemetry._nvml = SimpleNamespace(nvmlDeviceGetTemperature=lambda handle, sensor: 63)
    point = telemetry.thermal_point()
    assert point.gpu_temperature_c == 63
    assert point.gpu_power_watts is None
    assert telemetry.points == []


def test_fresh_thermal_read_reports_failure_instead_of_stale_value():
    telemetry = NVMLTelemetry(enable_gpu=False)
    telemetry._handle = object()
    def fail(*args):
        raise RuntimeError("sensor unavailable")
    telemetry._nvml = SimpleNamespace(nvmlDeviceGetTemperature=fail)
    assert telemetry.thermal_point() is None
    assert "sensor unavailable" in telemetry.errors[0]
