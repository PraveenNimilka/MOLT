from __future__ import annotations

import pytest
import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.methods.nf4_layer_cache import dense_nf4_layer_cache


class _QuantizedWeight:
    def __init__(self) -> None:
        self.data = torch.tensor([[99.0]])
        self.quant_state = object()
        self.requires_grad = False


def _fake_layer(monkeypatch: pytest.MonkeyPatch):
    import bitsandbytes.functional as bnb_functional
    import peft.tuners.lora.bnb as peft_bnb

    class FakeBase(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = _QuantizedWeight()
            self.bias = None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs + 7

    class FakeLinear4bit(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_layer = FakeBase()

    monkeypatch.setattr(peft_bnb, "Linear4bit", FakeLinear4bit)
    monkeypatch.setattr(
        bnb_functional,
        "dequantize_4bit",
        lambda data, quant_state: torch.tensor([[2.0]]),
    )
    layer = torch.nn.Sequential(FakeLinear4bit())
    return layer, layer[0].base_layer


def test_dense_nf4_cache_restores_forward_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer, base = _fake_layer(monkeypatch)
    original = base.forward
    inputs = torch.tensor([[3.0]], dtype=torch.bfloat16)

    with dense_nf4_layer_cache(layer) as projection_count:
        assert projection_count == 1
        torch.testing.assert_close(
            base(inputs), torch.tensor([[6.0]], dtype=torch.bfloat16)
        )

    assert base.forward == original
    torch.testing.assert_close(base(inputs), torch.tensor([[10.0]], dtype=torch.bfloat16))


def test_dense_nf4_cache_restores_forward_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer, base = _fake_layer(monkeypatch)
    original = base.forward

    with pytest.raises(RuntimeError, match="injected"):
        with dense_nf4_layer_cache(layer):
            raise RuntimeError("injected")

    assert base.forward == original


def test_dense_nf4_cache_rejects_layer_without_nf4_projection() -> None:
    with pytest.raises(CapabilityError, match="contains no PEFT NF4"):
        with dense_nf4_layer_cache(torch.nn.Sequential(torch.nn.Linear(2, 2))):
            pass


def test_dense_nf4_cache_rejects_biased_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer, base = _fake_layer(monkeypatch)
    base.bias = torch.tensor([0.5], dtype=torch.bfloat16)
    with pytest.raises(CapabilityError, match="biased NF4"):
        with dense_nf4_layer_cache(layer):
            pass
