from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from molt_stream.kernels.gated_activation import fused_silu_multiply


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_silu_multiply_matches_reference_on_cpu(dtype: torch.dtype) -> None:
    torch.manual_seed(3401)
    left_gate = torch.randn(3, 5, 17, dtype=dtype, requires_grad=True)
    left_up = torch.randn(3, 5, 17, dtype=dtype, requires_grad=True)
    right_gate = left_gate.detach().clone().requires_grad_()
    right_up = left_up.detach().clone().requires_grad_()
    gradient = torch.randn_like(left_gate)
    reference = F.silu(left_gate) * left_up
    candidate = fused_silu_multiply(right_gate, right_up)
    reference.backward(gradient)
    candidate.backward(gradient)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(candidate, reference, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(right_gate.grad, left_gate.grad, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(right_up.grad, left_up.grad, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_silu_multiply_matches_bf16_cuda_reference() -> None:
    torch.manual_seed(3402)
    left_gate = torch.randn(2, 31, 4864, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    left_up = torch.randn_like(left_gate, requires_grad=True)
    right_gate = left_gate.detach().clone().requires_grad_()
    right_up = left_up.detach().clone().requires_grad_()
    gradient = torch.randn_like(left_gate)
    reference = F.silu(left_gate) * left_up
    candidate = fused_silu_multiply(right_gate, right_up)
    reference.backward(gradient)
    candidate.backward(gradient)
    torch.testing.assert_close(candidate, reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(right_gate.grad, left_gate.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(right_up.grad, left_up.grad, rtol=2e-2, atol=2e-2)


def test_fused_silu_multiply_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="identical"):
        fused_silu_multiply(torch.randn(2, 3), torch.randn(2, 4))
