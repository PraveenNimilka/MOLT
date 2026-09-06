import pytest
import torch

from molt_stream.methods.activation_compression import CompressedSavedActivations


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compressed_saved_activations_bound_forward_and_gradient_error():
    torch.manual_seed(17)
    layer = torch.nn.Linear(128, 128, bias=False, device="cuda", dtype=torch.bfloat16)
    reference_input = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)

    reference = layer(reference_input).float().square().mean()
    reference.backward()
    reference_gradient = reference_input.grad.detach().float().clone()
    layer.zero_grad(set_to_none=True)

    with CompressedSavedActivations(minimum_bytes=0):
        candidate = layer(candidate_input).float().square().mean()
    candidate.backward()

    assert candidate == reference
    relative_gradient_error = (
        (candidate_input.grad.float() - reference_gradient).norm() / reference_gradient.norm()
    )
    assert float(relative_gradient_error) < 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compressed_saved_activations_support_fp32_saves():
    value = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    with CompressedSavedActivations(minimum_bytes=0):
        loss = value.sin().square().mean()
    loss.backward()
    assert torch.isfinite(value.grad).all()
