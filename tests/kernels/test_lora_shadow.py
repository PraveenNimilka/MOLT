from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from molt_stream.methods.lora_shadow import invalidate_bf16_lora_shadows, shadow_lora


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_shadow_matches_autocast_lora_forward_and_gradients() -> None:
    torch.manual_seed(1241)
    left = torch.randn(2, 17, 64, device="cuda", dtype=torch.bfloat16).requires_grad_()
    right = left.detach().clone().requires_grad_()
    a_left = torch.randn(8, 64, device="cuda", dtype=torch.float32).requires_grad_()
    b_left = torch.randn(96, 8, device="cuda", dtype=torch.float32).requires_grad_()
    a_right = a_left.detach().clone().requires_grad_()
    b_right = b_left.detach().clone().requires_grad_()
    a_shadow = a_right.detach().bfloat16()
    b_shadow = b_right.detach().bfloat16()
    upstream = torch.randn(2, 17, 96, device="cuda", dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = F.linear(F.linear(left, a_left), b_left) * 2.0
        actual = shadow_lora(
            right, a_right, b_right, a_shadow, b_shadow, 2.0
        )
    expected.backward(upstream)
    actual.backward(upstream)

    assert torch.equal(actual, expected)
    assert torch.allclose(right.grad, left.grad, rtol=4e-3, atol=4e-3)
    assert torch.allclose(a_right.grad, a_left.grad, rtol=4e-3, atol=4e-3)
    assert torch.allclose(b_right.grad, b_left.grad, rtol=4e-3, atol=4e-3)


def test_explicit_invalidation_does_not_depend_on_parameter_version() -> None:
    child = torch.nn.Linear(2, 2)
    child._molt_shadow_original_forward = child.forward
    child._molt_shadow_version = (4, 9)
    model = torch.nn.Sequential(child)
    assert invalidate_bf16_lora_shadows(model) == 1
    assert child._molt_shadow_version is None
