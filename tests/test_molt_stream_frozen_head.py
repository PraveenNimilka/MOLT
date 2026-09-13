import pytest
import torch

from molt_stream.kernels.partitioned_loss import exact_partitioned_linear_cross_entropy


@pytest.mark.parametrize("chunk", [1, 3, 16])
def test_precomputed_frozen_gradient_matches_scaled_reference(chunk):
    torch.manual_seed(43)
    h = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
    w = torch.randn(13, 7, dtype=torch.float64)
    y = torch.randint(13, (2, 4))
    reference = torch.nn.functional.cross_entropy((h @ w.T).flatten(0, 1), y.flatten())
    (reference * 0.37).backward()
    expected = h.grad.clone()
    h.grad = None
    actual = exact_partitioned_linear_cross_entropy(h, w, y, chunk, precompute_frozen_gradient=True)
    (actual * 0.37).backward()
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(h.grad, expected, atol=1e-12, rtol=1e-12)


def test_trainable_classifier_is_rejected():
    with pytest.raises(ValueError, match="frozen"):
        exact_partitioned_linear_cross_entropy(torch.ones(2, 3, requires_grad=True),
            torch.ones(5, 3, requires_grad=True), torch.zeros(2, dtype=torch.long), 1,
            precompute_frozen_gradient=True)


def test_no_grad_precomputed_path_preserves_reference_loss():
    hidden = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(13, 7, dtype=torch.float64)
    targets = torch.randint(13, (2, 4))
    reference = torch.nn.functional.cross_entropy((hidden @ weight.T).flatten(0, 1), targets.flatten())
    with torch.no_grad():
        actual = exact_partitioned_linear_cross_entropy(
            hidden,
            weight,
            targets,
            3,
            precompute_frozen_gradient=True,
            backend="auto",
        )
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
