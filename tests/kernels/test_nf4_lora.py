from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from molt_stream.methods.nf4_backward import (
    packed_nf4_backward_input,
    packed_nf4_lora_backward_input,
)
from molt_stream.methods.nf4_lora import enable_scheduled_nf4_lora, scheduled_nf4_lora


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width,out_width", [(64, 96), (1536, 256), (1536, 1536), (1536, 8960), (8960, 1536)])
@pytest.mark.parametrize("scale", [0.125, 0.3, 2.0])
def test_scheduled_nf4_lora_matches_peft_dataflow(width: int, out_width: int, scale: float) -> None:
    import bitsandbytes as bnb

    torch.manual_seed(941)
    x_left = torch.randn(1, 256, width, device="cuda", dtype=torch.bfloat16).requires_grad_()
    x_right = x_left.detach().clone().requires_grad_()
    dense_weight = torch.randn(out_width, width, device="cuda", dtype=torch.bfloat16)
    packed, quant_state = bnb.functional.quantize_4bit(
        dense_weight, blocksize=64, compress_statistics=True, quant_type="nf4"
    )
    a_left = torch.randn(8, width, device="cuda", dtype=torch.float32).requires_grad_()
    b_left = torch.randn(out_width, 8, device="cuda", dtype=torch.float32).requires_grad_()
    a_right = a_left.detach().clone().requires_grad_()
    b_right = b_left.detach().clone().requires_grad_()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = bnb.matmul_4bit(x_left, packed, quant_state=quant_state)
        expected = expected + F.linear(F.linear(x_left, a_left), b_left) * scale
        actual = scheduled_nf4_lora(
            x_right, packed, a_right, b_right, quant_state, scale
        )
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)

    assert torch.equal(actual, expected)
    assert torch.allclose(x_right.grad, x_left.grad, rtol=4e-3, atol=4e-3)
    assert torch.allclose(a_right.grad, a_left.grad, rtol=4e-3, atol=4e-3)
    assert torch.allclose(b_right.grad, b_left.grad, rtol=4e-3, atol=4e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cached_backward_weight_is_bitwise_equal_to_per_call_decode() -> None:
    import bitsandbytes as bnb

    torch.manual_seed(944)
    dense_weight = torch.randn(1536, 1536, device="cuda", dtype=torch.bfloat16)
    packed, quant_state = bnb.functional.quantize_4bit(
        dense_weight, blocksize=64, compress_statistics=True, quant_type="nf4"
    )
    decoded = bnb.functional.dequantize_4bit(packed, quant_state=quant_state).to(
        torch.bfloat16
    )
    x_left = torch.randn(1, 256, 1536, device="cuda", dtype=torch.bfloat16).requires_grad_()
    x_right = x_left.detach().clone().requires_grad_()
    a_left = torch.randn(8, 1536, device="cuda", dtype=torch.float32).requires_grad_()
    b_left = torch.randn(1536, 8, device="cuda", dtype=torch.float32).requires_grad_()
    a_right = a_left.detach().clone().requires_grad_()
    b_right = b_left.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        left = scheduled_nf4_lora(
            x_left, packed, a_left, b_left, quant_state, 2.0
        )
        right = scheduled_nf4_lora(
            x_right, packed, a_right, b_right, quant_state, 2.0, decoded
        )
    upstream = torch.randn_like(left)
    left.backward(upstream)
    right.backward(upstream)
    assert torch.equal(right, left)
    assert torch.equal(x_right.grad, x_left.grad)
    assert torch.equal(a_right.grad, a_left.grad)
    assert torch.equal(b_right.grad, b_left.grad)


def test_scheduled_nf4_lora_refuses_cpu() -> None:
    pytest.importorskip("bitsandbytes")

    class QuantState:
        pass

    x = torch.randn(2, 4, requires_grad=True)
    packed = torch.zeros(8, 1, dtype=torch.uint8)
    a = torch.randn(2, 4, requires_grad=True)
    b = torch.randn(4, 2, requires_grad=True)
    with pytest.raises(Exception, match="requires CUDA"):
        scheduled_nf4_lora(x, packed, a, b, QuantState(), 1.0)


def test_module_suffix_filter_refuses_when_no_compatible_projection() -> None:
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    with pytest.raises(Exception, match="down_proj"):
        enable_scheduled_nf4_lora(
            torch.nn.Sequential(torch.nn.Linear(4, 4)),
            module_suffixes=("down_proj",),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_nf4_backward_handles_dimension_tails() -> None:
    import bitsandbytes as bnb

    torch.manual_seed(942)
    dense = torch.randn(70, 65, device="cuda", dtype=torch.bfloat16)
    packed, quant_state = bnb.functional.quantize_4bit(
        dense, blocksize=64, compress_statistics=True, quant_type="nf4"
    )
    grad = torch.randn(5, 70, device="cuda", dtype=torch.bfloat16)
    reference = grad @ bnb.functional.dequantize_4bit(
        packed, quant_state=quant_state
    )
    actual = packed_nf4_backward_input(grad, packed, quant_state)
    assert torch.allclose(actual, reference, rtol=4e-3, atol=4e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_packed_nf4_lora_backward_handles_rank_and_dimension_tails() -> None:
    import bitsandbytes as bnb

    torch.manual_seed(943)
    dense = torch.randn(70, 65, device="cuda", dtype=torch.bfloat16)
    packed, quant_state = bnb.functional.quantize_4bit(
        dense, blocksize=64, compress_statistics=True, quant_type="nf4"
    )
    grad = torch.randn(5, 70, device="cuda", dtype=torch.bfloat16)
    lora_grad = torch.randn(5, 7, device="cuda", dtype=torch.bfloat16)
    lora_a = torch.randn(7, 65, device="cuda", dtype=torch.float32)
    scale = 0.125
    reference = grad @ bnb.functional.dequantize_4bit(
        packed, quant_state=quant_state
    )
    reference.add_(lora_grad @ lora_a.to(torch.bfloat16), alpha=scale)
    actual = packed_nf4_lora_backward_input(
        grad, packed, quant_state, lora_grad, lora_a, scale
    )
    assert torch.allclose(actual, reference, rtol=4e-3, atol=4e-3)
