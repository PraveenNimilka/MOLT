from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from molt_stream.kernels.frozen_linear_cross_entropy import frozen_linear_cross_entropy
from molt_stream.kernels.partitioned_loss import exact_partitioned_linear_cross_entropy


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_vocabulary_repeated_calls_are_reproducible() -> None:
    from molt_stream.kernels.frozen_linear_cross_entropy import (
        _analytical_frozen_head_forward,
        _triton_frozen_head_forward,
    )

    torch.manual_seed(1337)
    hidden = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(151936, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    targets = torch.randint(151936, (256,), device="cuda")
    reference_loss, reference_grad = _triton_frozen_head_forward(hidden, weight, targets, 96)
    expected = F.cross_entropy(F.linear(hidden, weight).float(), targets)
    torch.testing.assert_close(reference_loss, expected, rtol=1e-5, atol=1e-5)
    _, expected_grad = _analytical_frozen_head_forward(hidden, weight, targets, 96)
    torch.testing.assert_close(reference_grad, expected_grad, rtol=0, atol=2e-3)
    for _ in range(20):
        loss, grad = _triton_frozen_head_forward(hidden, weight, targets, 96)
        torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-6)
        assert torch.equal(grad, reference_grad)


def test_opcheck_cpu() -> None:
    h = torch.randn(8, 32, dtype=torch.float32, requires_grad=True)
    w = torch.randn(64, 32, dtype=torch.float32)
    t = torch.randint(64, (8,))
    torch.library.opcheck(frozen_linear_cross_entropy, (h, w, t, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_cuda() -> None:
    device = torch.device("cuda")
    h = torch.randn(8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
    t = torch.randint(128, (8,), device=device)
    torch.library.opcheck(frozen_linear_cross_entropy, (h, w, t, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_numerical_parity_against_unpartitioned_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    tokens, width, vocab = 128, 256, 4096
    chunk_size = 64

    hidden = torch.randn(tokens, width, device=device, dtype=dtype)
    weight = torch.randn(vocab, width, device=device, dtype=dtype)
    targets = torch.randint(vocab, (tokens,), device=device)

    # Reference unpartitioned in same dtype with FP32 CE reduction
    h_ref = hidden.detach().clone().requires_grad_(True)
    logits_ref = F.linear(h_ref, weight)
    loss_ref = F.cross_entropy(logits_ref.float(), targets, reduction="mean")
    loss_ref.backward()
    grad_ref = h_ref.grad.detach()

    # MOLT fused operator
    h_test = hidden.detach().clone().requires_grad_(True)
    loss_test = frozen_linear_cross_entropy(h_test, weight, targets, chunk_size)
    loss_test.backward()
    grad_test = h_test.grad.detach()

    # Parity checks
    loss_rel_diff = abs(loss_test.item() - loss_ref.item()) / loss_ref.item()
    assert loss_rel_diff <= 1e-5, f"Loss relative difference {loss_rel_diff} exceeds 1e-5"

    max_abs_grad_diff = (grad_test - grad_ref).abs().max().item()
    assert max_abs_grad_diff <= 2e-3, f"Max absolute gradient diff {max_abs_grad_diff} exceeds 2e-3"


def test_refusal_when_weight_requires_grad() -> None:
    h = torch.randn(4, 16, requires_grad=True)
    w = torch.randn(32, 16, requires_grad=True)
    t = torch.randint(32, (4,))
    with pytest.raises(ValueError, match="requires a frozen classifier"):
        frozen_linear_cross_entropy(h, w, t, 2)


def test_explicit_triton_backend_never_silently_falls_back_on_cpu() -> None:
    h = torch.randn(4, 16, requires_grad=True)
    w = torch.randn(32, 16)
    t = torch.randint(32, (4,))
    with pytest.raises(RuntimeError, match="unavailable"):
        exact_partitioned_linear_cross_entropy(
            h, w, t, 2, precompute_frozen_gradient=True, backend="triton"
        )


def test_validation_errors() -> None:
    w = torch.randn(32, 16)
    t = torch.randint(32, (4,))

    # 1D hidden
    with pytest.raises(ValueError, match="hidden must be"):
        frozen_linear_cross_entropy(torch.randn(16), w, t, 2)

    # Shape mismatch
    with pytest.raises(ValueError, match="targets must match"):
        frozen_linear_cross_entropy(torch.randn(4, 16), w, torch.randint(32, (5,)), 2)

    # Width mismatch
    with pytest.raises(ValueError, match="hidden width and classifier weight width must match"):
        frozen_linear_cross_entropy(torch.randn(4, 16), torch.randn(32, 20), t, 2)

    # Chunk size non-positive
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        frozen_linear_cross_entropy(torch.randn(4, 16), w, t, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_and_batched_inputs() -> None:
    device = torch.device("cuda")
    h = torch.randn(2, 8, 32, device=device, dtype=torch.bfloat16).transpose(0, 1).requires_grad_(True)
    assert not h.is_contiguous()
    w = torch.randn(64, 32, device=device, dtype=torch.bfloat16)
    t = torch.randint(64, (8, 2), device=device)

    loss = frozen_linear_cross_entropy(h, w, t, chunk_size=4)
    loss.backward()
    assert h.grad is not None
    assert h.grad.shape == h.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_adversarial_logits_stability() -> None:
    device = torch.device("cuda")
    # Extreme positive logits
    h = (torch.randn(4, 32, device=device, dtype=torch.float32) * 50.0).requires_grad_(True)
    w = torch.randn(64, 32, device=device, dtype=torch.float32) * 50.0
    t = torch.randint(64, (4,), device=device)

    loss = frozen_linear_cross_entropy(h, w, t, chunk_size=2)
    loss.backward()
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_odd_vocabulary_tail() -> None:
    device = torch.device("cuda")
    # Odd non-power-of-two vocabulary
    h = torch.randn(5, 47, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(317, 47, device=device, dtype=torch.bfloat16)
    t = torch.randint(317, (5,), device=device)

    loss = frozen_linear_cross_entropy(h, w, t, chunk_size=2)
    loss.backward()
    assert h.grad is not None
    assert not torch.isnan(loss)
    assert not torch.isnan(h.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_partitioned_loss_backend_dispatch() -> None:
    device = torch.device("cuda")
    h = torch.randn(16, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
    t = torch.randint(128, (16,), device=device)

    l_auto = exact_partitioned_linear_cross_entropy(h, w, t, 8, precompute_frozen_gradient=True, backend="auto")
    l_triton = exact_partitioned_linear_cross_entropy(h, w, t, 8, precompute_frozen_gradient=True, backend="triton")
    l_anal = exact_partitioned_linear_cross_entropy(h, w, t, 8, precompute_frozen_gradient=True, backend="analytical")

    assert abs(l_auto.item() - l_triton.item()) < 1e-5
    assert abs(l_triton.item() - l_anal.item()) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_accepts_fp32_hidden_with_bf16_frozen_cache() -> None:
    h = torch.randn(8, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    t = torch.randint(128, (8,), device="cuda")
    loss = exact_partitioned_linear_cross_entropy(
        h, w, t, 4, precompute_frozen_gradient=True, backend="triton"
    )
    loss.backward()
    assert h.grad is not None and h.grad.dtype == torch.float32
    assert torch.isfinite(loss) and torch.isfinite(h.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_loss_only_triton_path_is_bit_exact_to_training_loss() -> None:
    torch.manual_seed(91)
    hidden = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(151936, 64, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(151936, (16,), device="cuda")
    training_loss = exact_partitioned_linear_cross_entropy(
        hidden,
        weight,
        targets,
        8,
        precompute_frozen_gradient=True,
        backend="triton",
    )
    with torch.no_grad():
        evaluation_loss = exact_partitioned_linear_cross_entropy(
            hidden,
            weight,
            targets,
            8,
            precompute_frozen_gradient=True,
            backend="triton",
        )
    assert torch.equal(evaluation_loss, training_loss.detach())
