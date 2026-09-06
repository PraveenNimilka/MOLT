from types import SimpleNamespace

import pytest
import torch

from molt_stream.training.layer_thermal_guard import (
    TrainingThermalStop,
    guarded_decoder_training,
)


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen2")
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4) for _ in range(4)]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            value = layer(value).relu()
        return value


def test_guard_checks_forward_and_backward_without_changing_math() -> None:
    baseline = _Decoder()
    guarded = _Decoder()
    guarded.load_state_dict(baseline.state_dict())
    value = torch.randn(3, 4)
    expected = baseline(value).sum()
    expected.backward()
    checks = 0

    def checkpoint() -> bool:
        nonlocal checks
        checks += 1
        return True

    with guarded_decoder_training(
        guarded, boundary_after_layers=2, checkpoint=checkpoint
    ):
        actual = guarded(value).sum()
        actual.backward()

    assert checks == 2
    assert torch.equal(actual, expected)
    for left, right in zip(guarded.parameters(), baseline.parameters(), strict=True):
        assert torch.equal(left.grad, right.grad)
    assert all(not layer._forward_pre_hooks for layer in guarded.model.layers)
    assert all(not layer._backward_pre_hooks for layer in guarded.model.layers)


def test_guard_stops_and_always_removes_hooks() -> None:
    model = _Decoder()
    with pytest.raises(TrainingThermalStop):
        with guarded_decoder_training(
            model, boundary_after_layers=2, checkpoint=lambda: False
        ):
            model(torch.randn(1, 4))
    assert all(not layer._forward_pre_hooks for layer in model.model.layers)
    assert all(not layer._backward_pre_hooks for layer in model.model.layers)
