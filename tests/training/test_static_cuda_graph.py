from __future__ import annotations

import pytest
import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.training.static_cuda_graph import (
    StaticCudaMicrobatch,
    require_static_training_model,
)


def test_static_cuda_graph_refuses_without_cuda_inputs() -> None:
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    with pytest.raises(CapabilityError, match="CUDA"):
        StaticCudaMicrobatch(
            lambda value: (value @ parameter).square().mean(),
            (torch.ones(2, 2),),
            (parameter,),
        )


def test_static_cuda_graph_rejects_nonzero_dropout() -> None:
    with pytest.raises(CapabilityError, match="requires zero dropout"):
        require_static_training_model(torch.nn.Sequential(torch.nn.Dropout(0.1)))
    require_static_training_model(torch.nn.Sequential(torch.nn.Dropout(0.0)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_static_cuda_graph_matches_two_eager_accumulated_microbatches() -> None:
    torch.manual_seed(418)
    eager = torch.nn.Linear(8, 4, bias=False, device="cuda", dtype=torch.float32)
    captured = torch.nn.Linear(8, 4, bias=False, device="cuda", dtype=torch.float32)
    captured.load_state_dict(eager.state_dict())
    batches = (
        torch.randn(3, 8, device="cuda"),
        torch.randn(3, 8, device="cuda"),
    )

    for batch in batches:
        eager(batch).square().mean().div(len(batches)).backward()

    runner = StaticCudaMicrobatch(
        lambda value: captured(value).square().mean().div(len(batches)),
        (batches[0],),
        tuple(captured.parameters()),
    )
    runner.zero_grad()
    observed_losses = []
    for batch in batches:
        observed_losses.append(runner.replay(batch).detach().clone())
    torch.cuda.synchronize()

    torch.testing.assert_close(captured.weight.grad, eager.weight.grad)
    expected_losses = [
        eager(batch).square().mean().div(len(batches)).detach() for batch in batches
    ]
    for actual, expected in zip(observed_losses, expected_losses):
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_static_cuda_graph_rejects_signature_changes() -> None:
    layer = torch.nn.Linear(4, 2, bias=False, device="cuda")
    runner = StaticCudaMicrobatch(
        lambda value: layer(value).sum(),
        (torch.ones(2, 4, device="cuda"),),
        tuple(layer.parameters()),
    )
    with pytest.raises(ValueError, match="signature changed"):
        runner.replay(torch.ones(3, 4, device="cuda"))
