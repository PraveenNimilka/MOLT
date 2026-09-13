from __future__ import annotations

import pytest
import torch

from molt_stream.kernels.split_frozen_loss import (
    split_frozen_head_forward,
    split_frozen_head_supported,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(17, 64, 317), (256, 1536, 151936)])
def test_split_loss_matches_separated_reference_and_replays(shape):
    from molt_stream.kernels.frozen_linear_cross_entropy import (
        _separated_triton_frozen_head_forward,
        _triton_frozen_head_forward,
    )

    torch.manual_seed(1337)
    tokens, width, vocab = shape
    hidden = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(vocab, width, device="cuda", dtype=torch.bfloat16) * 0.05
    targets = torch.randint(vocab, (tokens,), device="cuda")
    expected = _separated_triton_frozen_head_forward(hidden, weight, targets, 96)
    actual = split_frozen_head_forward(hidden, weight, targets, 96)
    for left, right in zip(actual, expected):
        assert torch.equal(left, right)
    dispatched = _triton_frozen_head_forward(hidden, weight, targets, 96)
    for left, right in zip(dispatched, expected):
        assert torch.equal(left, right)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        split_frozen_head_forward(hidden, weight, targets, 96)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = split_frozen_head_forward(hidden, weight, targets, 96)
    for _ in range(3):
        hidden.add_(0.01)
        graph.replay()
        expected = _separated_triton_frozen_head_forward(hidden, weight, targets, 96)
        for left, right in zip(result, expected):
            assert torch.equal(left, right)


def test_split_loss_dispatch_rejects_unqualified_geometry():
    hidden = torch.empty(1, 8, 16)
    weight = torch.empty(32, 16)
    targets = torch.zeros(1, 8, dtype=torch.long)
    assert not split_frozen_head_supported(hidden, weight, targets, 96)
