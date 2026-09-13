from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from molt_stream.methods.gated_replay import asymmetric_gated_nf4_lora
from molt_stream.methods.nf4_lora import scheduled_nf4_lora


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("hidden,intermediate", [(64, 96), (128, 256)])
@pytest.mark.parametrize(
    "rematerialize_up,offload_up,offload_gate",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (False, True, True),
    ],
)
def test_asymmetric_gated_replay_matches_reference(
    hidden: int,
    intermediate: int,
    rematerialize_up: bool,
    offload_up: bool,
    offload_gate: bool,
) -> None:
    import bitsandbytes as bnb

    torch.manual_seed(1221)
    x_left = torch.randn(1, 32, hidden, device="cuda", dtype=torch.bfloat16).requires_grad_()
    x_right = x_left.detach().clone().requires_grad_()

    def projection(out_features: int, in_features: int) -> tuple[torch.Tensor, ...]:
        dense = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
        packed, state = bnb.functional.quantize_4bit(
            dense, blocksize=64, compress_statistics=True, quant_type="nf4"
        )
        a = torch.randn(8, in_features, device="cuda", dtype=torch.float32).requires_grad_()
        b = torch.randn(out_features, 8, device="cuda", dtype=torch.float32).requires_grad_()
        return packed, state, a, b

    gate_packed, gate_state, gate_a_left, gate_b_left = projection(intermediate, hidden)
    up_packed, up_state, up_a_left, up_b_left = projection(intermediate, hidden)
    down_packed, down_state, down_a_left, down_b_left = projection(hidden, intermediate)
    gate_a_right = gate_a_left.detach().clone().requires_grad_()
    gate_b_right = gate_b_left.detach().clone().requires_grad_()
    up_a_right = up_a_left.detach().clone().requires_grad_()
    up_b_right = up_b_left.detach().clone().requires_grad_()
    down_a_right = down_a_left.detach().clone().requires_grad_()
    down_b_right = down_b_left.detach().clone().requires_grad_()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        gate_left = scheduled_nf4_lora(
            x_left, gate_packed, gate_a_left, gate_b_left, gate_state, 2.0
        )
        up_left = scheduled_nf4_lora(
            x_left, up_packed, up_a_left, up_b_left, up_state, 2.0
        )
        expected = scheduled_nf4_lora(
            F.silu(gate_left) * up_left,
            down_packed,
            down_a_left,
            down_b_left,
            down_state,
            2.0,
        )
        gate_right = scheduled_nf4_lora(
            x_right, gate_packed, gate_a_right, gate_b_right, gate_state, 2.0
        )
        up_right = scheduled_nf4_lora(
            x_right, up_packed, up_a_right, up_b_right, up_state, 2.0
        )
        actual = asymmetric_gated_nf4_lora(
            x_right,
            gate_right,
            up_right,
            up_packed,
            up_a_right,
            up_b_right,
            down_packed,
            down_a_right,
            down_b_right,
            up_state,
            down_state,
            2.0,
            2.0,
            rematerialize_up,
            offload_up,
            torch.empty_like(up_right, device="cpu", pin_memory=True)
            if offload_up
            else None,
            offload_gate,
            torch.empty_like(gate_right, device="cpu", pin_memory=True)
            if offload_gate
            else None,
        )

    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    assert torch.equal(actual, expected)
    for left, right in (
        (x_left, x_right),
        (gate_a_left, gate_a_right),
        (gate_b_left, gate_b_right),
        (up_a_left, up_a_right),
        (up_b_left, up_b_right),
        (down_a_left, down_a_right),
        (down_b_left, down_b_right),
    ):
        torch.testing.assert_close(right.grad, left.grad, rtol=4e-3, atol=4e-3)
