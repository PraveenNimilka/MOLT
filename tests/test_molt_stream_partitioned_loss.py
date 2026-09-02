import pytest
import torch

from molt_stream.kernels.partitioned_loss import exact_partitioned_linear_cross_entropy


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 32])
def test_partitioned_linear_ce_matches_loss_and_gradients(chunk_size):
    torch.manual_seed(17)
    hidden_reference = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
    weight_reference = torch.randn(11, 7, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, 11, (2, 4))

    logits = torch.nn.functional.linear(hidden_reference, weight_reference)
    reference = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    reference.backward()

    hidden = hidden_reference.detach().clone().requires_grad_(True)
    weight = weight_reference.detach().clone().requires_grad_(True)
    candidate = exact_partitioned_linear_cross_entropy(hidden, weight, targets, chunk_size)
    candidate.backward()

    torch.testing.assert_close(candidate, reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(hidden.grad, hidden_reference.grad, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(weight.grad, weight_reference.grad, rtol=1e-11, atol=1e-11)


def test_partitioned_linear_ce_validates_shapes():
    hidden = torch.randn(2, 3, 4)
    weight = torch.randn(5, 4)
    targets = torch.zeros(2, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="targets"):
        exact_partitioned_linear_cross_entropy(hidden, weight, targets, 2)


def test_partitioned_linear_ce_rejects_invalid_chunk_size():
    hidden = torch.randn(2, 3, 4)
    weight = torch.randn(5, 4)
    targets = torch.zeros(2, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="chunk_size"):
        exact_partitioned_linear_cross_entropy(hidden, weight, targets, 0)
