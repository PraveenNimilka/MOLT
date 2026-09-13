from __future__ import annotations

import pytest
import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.measurement.update_regions import CapturedCudaRegionTimer
from molt_stream.training.static_cuda_graph import (
    StaticCudaMicrobatch,
    cuda_allocation_peak_ledger,
    require_static_training_model,
)


def test_cuda_allocation_peak_ledger_reconstructs_live_set() -> None:
    frame_a = [{"filename": "model.py", "line": 10, "name": "forward"}]
    frame_b = [{"filename": "loss.py", "line": 20, "name": "loss"}]
    snapshot = {
        "device_traces": [[
            {"action": "alloc", "addr": 100, "size": 40, "frames": frame_a},
            {"action": "alloc", "addr": 200, "size": 70, "frames": frame_b},
            {"action": "free_requested", "addr": 100, "size": 40},
            {"action": "alloc", "addr": 300, "size": 20, "frames": frame_a},
        ]]
    }

    ledger = cuda_allocation_peak_ledger(snapshot)

    assert ledger["incremental_peak_bytes"] == 110
    assert ledger["live_allocations_at_peak"] == 2
    assert ledger["groups"] == [
        {"origin": "loss.py:20:loss", "bytes": 70, "allocations": 1},
        {"origin": "model.py:10:forward", "bytes": 40, "allocations": 1},
    ]


def test_static_cuda_graph_exposes_default_pool_trim_option() -> None:
    import inspect

    parameters = inspect.signature(StaticCudaMicrobatch).parameters
    assert parameters["trim_unused_default_pool"].default is False


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_static_cuda_graph_reports_internal_replay_regions() -> None:
    layer = torch.nn.Linear(128, 64, bias=False, device="cuda")
    timer = CapturedCudaRegionTimer(
        ("graph_start", "transformer_end", "forward_end", "graph_end")
    )

    def loss_builder(value: torch.Tensor) -> torch.Tensor:
        hidden = layer(value)
        timer.record("transformer_end")
        return hidden.square().mean()

    value = torch.randn(128, 128, device="cuda")
    runner = StaticCudaMicrobatch(
        loss_builder,
        (value,),
        tuple(layer.parameters()),
        joint_forward_backward=True,
        region_timer=timer,
    )
    runner.zero_grad()
    runner.stage_inputs(value)
    runner.replay_staged()
    torch.cuda.synchronize()

    assert timer.elapsed_seconds("graph_start", "transformer_end") > 0
    assert timer.elapsed_seconds("transformer_end", "forward_end") > 0
    assert timer.elapsed_seconds("forward_end", "graph_end") > 0
