from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from molt_stream.methods.nf4_lora import enable_scheduled_nf4_lora, scheduled_nf4_lora
from molt_stream.methods.nf4_backward import packed_nf4_backward_input


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scheduled_nf4_lora_matches_peft_dataflow() -> None:
    import bitsandbytes as bnb

    torch.manual_seed(941)
    x_left = torch.randn(2, 17, 64, device="cuda", dtype=torch.bfloat16).requires_grad_()
    x_right = x_left.detach().clone().requires_grad_()
    dense_weight = torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)
    packed, quant_state = bnb.functional.quantize_4bit(
        dense_weight, blocksize=64, compress_statistics=True, quant_type="nf4"
    )
    a_left = torch.randn(8, 64, device="cuda", dtype=torch.float32).requires_grad_()
    b_left = torch.randn(96, 8, device="cuda", dtype=torch.float32).requires_grad_()
    a_right = a_left.detach().clone().requires_grad_()
    b_right = b_left.detach().clone().requires_grad_()
    scale = 0.125

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


def test_scheduled_nf4_lora_refuses_cpu() -> None:
    class QuantState:
        pass

    x = torch.randn(2, 4, requires_grad=True)
    packed = torch.zeros(8, 1, dtype=torch.uint8)
    a = torch.randn(2, 4, requires_grad=True)
    b = torch.randn(4, 2, requires_grad=True)
    with pytest.raises(Exception, match="requires CUDA"):
        scheduled_nf4_lora(x, packed, a, b, QuantState(), 1.0)


def test_module_suffix_filter_refuses_when_no_compatible_projection() -> None:
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
