import pytest
import torch

from molt_stream.training.activation_offload import SavedActivationOffload


def test_activation_offload_rejects_negative_threshold():
    with pytest.raises(ValueError, match="non-negative"):
        SavedActivationOffload(minimum_bytes=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_activation_offload_preserves_forward_and_gradient():
    torch.manual_seed(7)
    layer = torch.nn.Linear(64, 64, bias=False, device="cuda")
    reference_input = torch.randn(32, 64, device="cuda", requires_grad=True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)

    reference = layer(reference_input).square().mean()
    reference.backward()
    reference_gradient = reference_input.grad.detach().clone()
    layer.zero_grad(set_to_none=True)

    with SavedActivationOffload(minimum_bytes=0):
        candidate = layer(candidate_input).square().mean()
    candidate.backward()

    torch.testing.assert_close(candidate, reference)
    torch.testing.assert_close(candidate_input.grad, reference_gradient)
