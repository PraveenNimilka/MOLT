import torch

from molt_stream.kernels.frozen_rmsnorm import _FrozenRMSNormFunction


def test_frozen_rmsnorm_preserves_bf16_contract_and_reference_gradient():
    torch.manual_seed(23)
    weight = torch.randn(64, dtype=torch.float32)
    candidate_input = torch.randn(8, 64, dtype=torch.bfloat16, requires_grad=True)
    reference_input = candidate_input.detach().float().requires_grad_(True)
    gradient = torch.randn_like(candidate_input)

    candidate = _FrozenRMSNormFunction.apply(candidate_input, weight, 1e-6)
    reference_inverse = torch.rsqrt(reference_input.square().mean(-1, keepdim=True) + 1e-6)
    reference = reference_input * reference_inverse * weight
    candidate.backward(gradient)
    reference.backward(gradient.float())

    assert candidate.dtype == torch.bfloat16
    torch.testing.assert_close(candidate.float(), reference, rtol=8e-3, atol=8e-3)
    torch.testing.assert_close(
        candidate_input.grad.float(), reference_input.grad, rtol=1.5e-2, atol=1.5e-2
    )
