import torch

from molt_stream.kernels.frozen_rmsnorm import _FrozenRMSNormFunction


def test_frozen_rmsnorm_preserves_bf16_contract_and_reference_gradient():
    torch.manual_seed(23)
    weight = torch.randn(64, dtype=torch.float32)
    candidate_input = torch.randn(8, 64, dtype=torch.bfloat16, requires_grad=True)
    reference_input = candidate_input.detach().float().requires_grad_(True)
    gradient = torch.randn_like(candidate_input)

    candidate = _FrozenRMSNormFunction.apply(candidate_input, weight, 1e-6, 0.0)
    reference_inverse = torch.rsqrt(reference_input.square().mean(-1, keepdim=True) + 1e-6)
    reference = reference_input * reference_inverse * weight
    candidate.backward(gradient)
    reference.backward(gradient.float())

    assert candidate.dtype == torch.bfloat16
    torch.testing.assert_close(candidate.float(), reference, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(
        candidate_input.grad.float(), reference_input.grad, rtol=1.5e-2, atol=1.5e-2
    )


def test_gemma_weight_offset_preserves_reference_gradient():
    torch.manual_seed(24)
    weight = torch.randn(32, dtype=torch.float32)
    candidate_input = torch.randn(4, 32, dtype=torch.bfloat16, requires_grad=True)
    reference_input = candidate_input.detach().float().requires_grad_(True)
    gradient = torch.randn_like(candidate_input)

    candidate = _FrozenRMSNormFunction.apply(candidate_input, weight, 1e-6, 1.0)
    inverse = torch.rsqrt(reference_input.square().mean(-1, keepdim=True) + 1e-6)
    reference = reference_input * inverse * (1.0 + weight)
    candidate.backward(gradient)
    reference.backward(gradient.float())

    torch.testing.assert_close(candidate.float(), reference, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(
        candidate_input.grad.float(), reference_input.grad, rtol=1.5e-2, atol=1.5e-2
    )


def test_frozen_rmsnorm_converts_fp32_residual_to_bf16_without_losing_fp32_gradient():
    torch.manual_seed(25)
    weight = torch.randn(32, dtype=torch.float32)
    hidden = torch.randn(3, 32, dtype=torch.float32, requires_grad=True)
    output = _FrozenRMSNormFunction.apply(hidden, weight, 1e-6, 0.0)

    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert hidden.grad is not None
    assert hidden.grad.dtype == torch.float32
