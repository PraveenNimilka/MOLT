import pytest

from molt_stream.core.contracts import TelemetryPoint
from molt_stream.core.specs import (
    DataSpec,
    ModelSpec,
    StreamSpec,
    TrainingMode,
    TrainingSpec,
)
from molt_stream.measurement.thermal import (
    HeatSoakGuardEnvelope,
    MicroPauseThermalController,
    SteadyDutyThermalController,
    ThermalCruiseController,
    ZonedThermalController,
    duty_cycle_pause_seconds,
    latest_temperature_c,
)


def test_heat_soak_guard_envelope_switches_once_without_model_assumptions():
    envelope = HeatSoakGuardEnvelope(
        heat_soak_c=75.0,
        steady_c=76.0,
        heat_soak_seconds=150.0,
        abort_c=84.0,
    )
    assert envelope.target_c(0.0) == 75.0
    assert envelope.target_c(149.999) == 75.0
    assert envelope.target_c(150.0) == 76.0
    assert envelope.target_c(10_000.0) == 76.0


@pytest.mark.parametrize(
    "values",
    [
        (76.0, 75.0, 150.0, 84.0),
        (75.0, 76.0, 0.0, 84.0),
        (75.0, 84.0, 150.0, 84.0),
    ],
)
def test_heat_soak_guard_envelope_rejects_invalid_boundaries(values):
    with pytest.raises(ValueError):
        HeatSoakGuardEnvelope(*values)


def point(
    temperature: float | None, timestamp: float = 0.0, power_watts: float | None = None
) -> TelemetryPoint:
    return TelemetryPoint(timestamp, 1, power_watts, None, temperature, None, None, None)


def test_thermal_pause_is_zero_at_or_below_target():
    assert duty_cycle_pause_seconds(60.0, target_c=60.0, abort_c=72.0) == 0.0
    assert duty_cycle_pause_seconds(59.0, target_c=60.0, abort_c=72.0) == 0.0


def test_thermal_pause_is_bounded_and_increases_with_overshoot():
    low = duty_cycle_pause_seconds(61.0, target_c=60.0, abort_c=72.0)
    high = duty_cycle_pause_seconds(71.0, target_c=60.0, abort_c=72.0)
    assert 0.03 <= low < high <= 0.05


def test_latest_temperature_ignores_missing_samples():
    assert latest_temperature_c([point(58.0), point(None), point(61.0)]) == 61.0


def test_training_spec_round_trips_thermal_and_power_fields(tmp_path):
    data = tmp_path / "tokens.bin"
    data.write_bytes(bytes(range(64)))
    spec = TrainingSpec(
        mode=TrainingMode.PRETRAIN,
        data=DataSpec(str(data), context_length=8, storage_dtype="uint8"),
        model=ModelSpec(vocab_size=256, context_length=8, layers=1, width=8, heads=1, hidden_width=16),
        stream=StreamSpec(device="cpu"),
        thermal_target_c=60.0,
        thermal_abort_c=72.0,
        thermal_pause_seconds=0.04,
        power_limit_watts=65.0,
        thermal_power_target_watts=50.0,
        thermal_control_mode="zone-cruise",
        thermal_cruise_max_c=66.0,
        min_pause_ms=10.0,
        max_pause_ms=25.0,
        thermal_protective_pause_ms=100.0,
        thermal_startup_max_c=50.0,
        thermal_cpu_startup_max_c=65.0,
        thermal_startup_dwell_seconds=10.0,
        thermal_startup_timeout_seconds=60.0,
    )
    restored = TrainingSpec.from_dict(spec.to_dict())
    restored.validate()
    assert restored.thermal_target_c == 60.0
    assert restored.thermal_abort_c == 72.0
    assert restored.power_limit_watts == 65.0
    assert restored.thermal_power_target_watts == 50.0
    assert restored.thermal_control_mode == "zone-cruise"
    assert restored.thermal_cruise_max_c == 66.0
    assert restored.min_pause_ms == 10.0
    assert restored.max_pause_ms == 25.0
    assert restored.thermal_startup_max_c == 50.0
    assert restored.thermal_cpu_startup_max_c == 65.0
    assert restored.thermal_startup_dwell_seconds == 10.0
    assert restored.thermal_startup_timeout_seconds == 60.0


def test_startup_thermal_configuration_fails_closed(tmp_path):
    data = tmp_path / "tokens.bin"
    data.write_bytes(bytes(range(64)))
    spec = TrainingSpec(
        mode=TrainingMode.PRETRAIN,
        data=DataSpec(str(data), context_length=8, storage_dtype="uint8"),
        model=ModelSpec(
            vocab_size=256,
            context_length=8,
            layers=1,
            width=8,
            heads=1,
            hidden_width=16,
        ),
        stream=StreamSpec(device="cpu"),
    )
    with pytest.raises(ValueError, match="requires thermal_startup_max_c"):
        TrainingSpec.from_dict(
            {**spec.to_dict(), "thermal_startup_dwell_seconds": 1.0}
        ).validate()


def test_predictive_cruise_brakes_before_target_crossing():
    controller = ThermalCruiseController(
        target_c=68.0,
        abort_c=72.0,
        lookahead_seconds=4.0,
        stability_band_c=1.0,
        initial_pause_seconds=0.002,
    )
    controller.update(point(65.0, 1.0), step_seconds=0.1)
    controller.update(point(66.0, 1.1), step_seconds=0.1)
    decision = controller.update(point(67.0, 1.2), step_seconds=0.1)
    assert decision.temperature_c < 68.0
    assert decision.projected_temperature_c > 67.5
    assert decision.pause_seconds > 0.002
    assert decision.phase == "braking"


def test_predictive_cruise_pause_decays_when_cooling_and_aborts_safely():
    controller = ThermalCruiseController(target_c=68.0, abort_c=72.0)
    hot = controller.update(point(69.0, 1.0), step_seconds=0.1)
    cool = hot
    for timestamp in range(2, 16):
        cool = controller.update(point(65.0, float(timestamp)), step_seconds=0.1)
    abort = controller.update(point(73.0, 16.0), step_seconds=0.1)
    assert hot.pause_seconds > cool.pause_seconds
    assert abort.abort


def test_power_feed_forward_paces_before_temperature_rises():
    unbudgeted = ThermalCruiseController(target_c=68.0, abort_c=72.0)
    budgeted = ThermalCruiseController(
        target_c=68.0, abort_c=72.0, power_target_watts=50.0
    )
    cool_high_power = point(55.0, 1.0, power_watts=80.0)
    plain = unbudgeted.update(cool_high_power, step_seconds=0.1)
    controlled = budgeted.update(cool_high_power, step_seconds=0.1)
    assert controlled.pause_seconds > plain.pause_seconds


def test_zone_cruise_scales_smoothly_from_10_to_25_ms():
    controller = ZonedThermalController(
        target_c=65.0,
        cruise_max_c=75.0,
        abort_c=80.0,
        minimum_pause_seconds=0.010,
        maximum_pause_seconds=0.025,
        protective_pause_seconds=0.100,
    )
    full = controller.update(point(64.9), step_seconds=0.1)
    low = controller.update(point(65.0), step_seconds=0.1)
    middle = controller.update(point(70.0), step_seconds=0.1)
    high = controller.update(point(75.0), step_seconds=0.1)
    assert full.phase == "full-speed" and full.pause_seconds == 0.0
    assert low.pause_seconds == pytest.approx(0.010)
    assert middle.pause_seconds == pytest.approx(0.0175)
    assert high.pause_seconds == pytest.approx(0.025)
    assert low.pause_seconds < middle.pause_seconds < high.pause_seconds


def test_zone_cruise_protects_above_ceiling_and_aborts_at_boundary():
    controller = ZonedThermalController(
        target_c=65.0, cruise_max_c=75.0, abort_c=80.0
    )
    protective = controller.update(point(76.0), step_seconds=0.1)
    still_protective = controller.update(point(75.0), step_seconds=0.1)
    released = controller.update(point(73.0), step_seconds=0.1)
    abort = controller.update(point(80.0), step_seconds=0.1)
    assert protective.phase == "protective-cooling"
    assert protective.pause_seconds == pytest.approx(0.100)
    assert not protective.abort
    assert still_protective.phase == "protective-cooling"
    assert released.phase == "cooling"
    assert abort.phase == "thermal-abort"
    assert abort.abort and abort.pause_seconds == 0.0


def test_micro_pause_bands_and_abort_are_bounded():
    controller = MicroPauseThermalController()
    assert controller.update(point(77, 1), step_seconds=.13).pause_seconds == 0
    assert controller.update(point(78, 2), step_seconds=.13).pause_seconds == pytest.approx(.001)
    assert controller.update(point(79, 3), step_seconds=.13).pause_seconds == pytest.approx(.006)
    assert controller.update(point(80, 4), step_seconds=.13).pause_seconds == pytest.approx(.012)
    protective = controller.update(point(82, 5), step_seconds=.13)
    assert protective.pause_seconds == pytest.approx(.100)
    assert controller.update(point(84, 6), step_seconds=.13).abort


def test_micro_pause_latch_requires_three_distinct_cool_samples():
    controller = MicroPauseThermalController()
    controller.update(point(82, 1), step_seconds=.13)
    for timestamp in (2, 2, 3):
        assert controller.update(point(80, timestamp), step_seconds=.13).phase == "protective-micro-pause"
    assert controller.update(point(80, 4), step_seconds=.13).phase == "micro-pause-high"


def test_micro_pause_missing_telemetry_is_protective():
    decision = MicroPauseThermalController().update(None, step_seconds=.13)
    assert decision.phase == "telemetry-protective"
    assert decision.pause_seconds == pytest.approx(.100)


def test_micro_pause_computes_bounded_power_budget_delay():
    controller = MicroPauseThermalController()
    decision = controller.update(point(72, 1, 72), step_seconds=.12)
    expected = .12 * (72 - 48) / (48 - 12)
    assert decision.phase == "power-budget-micro-pause"
    assert decision.pause_seconds == pytest.approx(expected)
    bounded = controller.update(point(72, 2, 140), step_seconds=.12)
    assert bounded.pause_seconds == pytest.approx(.100)


def test_steady_duty_paces_from_start_and_latches_protection():
    controller = SteadyDutyThermalController(
        steady_pause_seconds=0.040,
        protective_c=72.0,
        abort_c=78.0,
        protective_pause_seconds=0.100,
        protective_hysteresis_c=2.0,
    )
    cool = controller.update(point(55.0), step_seconds=0.05)
    protective = controller.update(point(72.0), step_seconds=0.05)
    latched = controller.update(point(71.0), step_seconds=0.05)
    released = controller.update(point(70.0), step_seconds=0.05)
    abort = controller.update(point(78.0), step_seconds=0.05)
    assert cool.pause_seconds == pytest.approx(0.040)
    assert protective.pause_seconds == pytest.approx(0.100)
    assert latched.phase == "protective-cooling"
    assert released.pause_seconds == pytest.approx(0.040)
    assert abort.abort
