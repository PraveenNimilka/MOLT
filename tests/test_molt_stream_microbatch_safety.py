import copy
import random

import pytest
import torch

from molt_stream.core.contracts import TelemetryPoint
from molt_stream.core.specs import DataSpec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.experiments.store import AtomicCheckpointStore
from molt_stream.measurement.thermal import (
    passive_cooldown,
    postrun_thermal_violation,
    wait_for_stable_thermal_headroom,
    wait_for_stable_system_headroom,
    wait_for_thermal_headroom,
)
from molt_stream.training.update_transaction import UpdateTransaction


def test_microbatch_thermal_wait_and_hysteresis(monkeypatch):
    clock = [10.0]
    temperatures = iter([65., 61., 59.])
    monkeypatch.setattr("molt_stream.measurement.thermal.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("molt_stream.measurement.thermal.time.sleep", lambda x: clock.__setitem__(0, clock[0] + x))
    def read():
        return TelemetryPoint(clock[0], 0, None, None, next(temperatures), None, None, None)
    result = wait_for_thermal_headroom(read, target_c=60, abort_c=72)
    assert result.stop_reason is None
    assert result.pause_seconds == pytest.approx(0.2)


@pytest.mark.parametrize("temperature,age", [(73., 0.), (None, 0.), (50., 2.)])
def test_microbatch_gate_stops_on_heat_or_unreliable_sensor(monkeypatch, temperature, age):
    monkeypatch.setattr("molt_stream.measurement.thermal.time.perf_counter", lambda: 10.)
    point = TelemetryPoint(10. - age, 0, None, None, temperature, None, None, None)
    assert wait_for_thermal_headroom(lambda: point, target_c=60, abort_c=72).stop_reason


def test_synchronous_sample_timestamp_is_checked_after_read(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("molt_stream.measurement.thermal.time.perf_counter", lambda: clock[0])
    def read():
        clock[0] += 0.01
        return TelemetryPoint(clock[0], 0, None, None, 50., None, None, None)
    assert wait_for_thermal_headroom(read, target_c=60, abort_c=72).stop_reason is None


@pytest.mark.parametrize(
    ("peak", "expected"),
    [(None, False), (71.999, False), (72.0, True), (75.0, True), (float("nan"), False)],
)
def test_delayed_nvml_peak_is_reconciled_with_abort_boundary(peak, expected):
    assert postrun_thermal_violation(peak, abort_c=72.0) is expected


def test_delayed_peak_reconciliation_rejects_invalid_boundary():
    with pytest.raises(ValueError):
        postrun_thermal_violation(70.0, abort_c=float("nan"))


def test_passive_cooldown_waits_without_rejecting_hot_start(monkeypatch):
    clock = [10.0]
    temperatures = iter([74., 65., 55.])
    monkeypatch.setattr("molt_stream.measurement.thermal.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("molt_stream.measurement.thermal.time.sleep",
                        lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    result = passive_cooldown(
        lambda: TelemetryPoint(clock[0], 0, None, None, next(temperatures), None, None, None),
        recovery_c=55,
    )
    assert result.stop_reason is None
    assert result.temperature_c == 55
    assert result.pause_seconds == pytest.approx(0.5)


def test_stable_startup_gate_resets_dwell_after_reheating(monkeypatch):
    clock = [10.0]
    temperatures = iter([49.0, 51.0, 49.0, 49.0, 49.0, 49.0, 49.0])
    monkeypatch.setattr(
        "molt_stream.measurement.thermal.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setattr(
        "molt_stream.measurement.thermal.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    result = wait_for_stable_thermal_headroom(
        lambda: TelemetryPoint(
            clock[0], 0, None, None, next(temperatures), None, None, None
        ),
        maximum_c=50.0,
        dwell_seconds=1.0,
        maximum_wait_seconds=2.0,
    )
    assert result.stop_reason is None
    assert result.pause_seconds == pytest.approx(1.5)


def test_stable_startup_gate_times_out_when_chassis_never_cools(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(
        "molt_stream.measurement.thermal.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setattr(
        "molt_stream.measurement.thermal.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    result = wait_for_stable_thermal_headroom(
        lambda: TelemetryPoint(clock[0], 0, None, None, 51.0, None, None, None),
        maximum_c=50.0,
        dwell_seconds=0.5,
        maximum_wait_seconds=0.5,
    )
    assert result.stop_reason == "stable startup cooling timeout"


def test_system_startup_gate_waits_for_both_gpu_and_cpu(monkeypatch):
    clock = [10.0]
    temperatures = iter([(54.0, 75.0), (54.0, 69.0), (54.0, 69.0), (54.0, 69.0)])
    monkeypatch.setattr(
        "molt_stream.measurement.thermal.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setattr(
        "molt_stream.measurement.thermal.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    def read():
        gpu, cpu = next(temperatures)
        return TelemetryPoint(
            clock[0], 0, None, None, gpu, None, None, None,
            cpu_temperature_c=cpu,
        )

    result = wait_for_stable_system_headroom(
        read,
        maximum_gpu_c=55.0,
        maximum_cpu_c=70.0,
        dwell_seconds=0.5,
        maximum_wait_seconds=2.0,
    )
    assert result.stop_reason is None
    assert result.cpu_sensor_available
    assert result.pause_seconds == pytest.approx(0.75)


def test_system_startup_gate_discloses_optional_cpu_sensor_absence(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(
        "molt_stream.measurement.thermal.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setattr(
        "molt_stream.measurement.thermal.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    result = wait_for_stable_system_headroom(
        lambda: TelemetryPoint(clock[0], 0, None, None, 50.0, None, None, None),
        maximum_gpu_c=55.0,
        maximum_cpu_c=70.0,
        dwell_seconds=0.5,
        maximum_wait_seconds=1.0,
    )
    assert result.stop_reason is None
    assert not result.cpu_sensor_available


def test_partial_accumulation_rollback_checkpoint_resume_is_exact(tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(bytes(range(64)))
    batcher = MMapTokenBatcher(DataSpec(str(path), context_length=4, storage_dtype="uint8"), seed=4, device="cpu")
    torch.manual_seed(11)
    random.seed(11)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    def microbatch():
        x, y = batcher.batch(1)
        ((model(x.float() / 64) - y.float() / 64).square().mean() / 2).backward()
        random.random()

    microbatch()
    microbatch()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before_model = copy.deepcopy(model.state_dict())
    transaction = UpdateTransaction.capture(batcher, cuda=False)
    microbatch()
    transaction.rollback(batcher, optimizer)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before_model[key])
    assert all(p.grad is None for p in model.parameters())
    store = AtomicCheckpointStore(tmp_path / "run")
    store.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "batcher": batcher.state_dict(), "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate()})
    microbatch()
    microbatch()
    optimizer.step()
    reference = copy.deepcopy(model.state_dict())
    reference_rng = torch.get_rng_state()
    reference_python = random.getstate()
    state = store.load()
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    optimizer.zero_grad(set_to_none=True)
    batcher.load_state_dict(state["batcher"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    microbatch()
    microbatch()
    optimizer.step()
    for key, value in model.state_dict().items():
        assert torch.equal(value, reference[key])
    assert torch.equal(torch.get_rng_state(), reference_rng)
    assert random.getstate() == reference_python


def test_rollback_can_preserve_static_gradient_buffer_addresses(tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(bytes(range(32)))
    batcher = MMapTokenBatcher(
        DataSpec(str(path), context_length=4, storage_dtype="uint8"),
        seed=7,
        device="cpu",
    )
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = torch.optim.AdamW((parameter,), lr=0.001)
    parameter.grad = torch.ones_like(parameter)
    pointer = parameter.grad.data_ptr()
    transaction = UpdateTransaction.capture(batcher, cuda=False)
    batcher.batch(1)

    transaction.rollback(batcher, optimizer, preserve_grad_buffers=True)

    assert parameter.grad is not None
    assert parameter.grad.data_ptr() == pointer
    assert torch.count_nonzero(parameter.grad) == 0
    restored = batcher.state_dict()
    assert restored["cursor"] == transaction.batcher_state["cursor"]
    assert restored["samples"] == transaction.batcher_state["samples"]
    assert torch.equal(
        restored["generator_state"], transaction.batcher_state["generator_state"]
    )
