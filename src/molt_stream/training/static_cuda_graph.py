from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from molt_stream.core.errors import CapabilityError


def require_static_training_model(model: torch.nn.Module) -> None:
    """Reject stochastic module state that a replay would silently freeze."""

    active_dropout = [
        name or "<root>"
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0
    ]
    config = getattr(model, "config", None)
    configured_dropout = {
        name: float(getattr(config, name, 0.0) or 0.0)
        for name in ("attention_dropout", "hidden_dropout", "dropout")
        if float(getattr(config, name, 0.0) or 0.0) != 0.0
    }
    if active_dropout or configured_dropout:
        raise CapabilityError(
            "static training capture requires zero dropout; "
            f"modules={active_dropout}, config={configured_dropout}"
        )


@dataclass(frozen=True)
class TensorSignature:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


def summarize_cuda_memory_pools(
    snapshot: list[dict[str, object]] | None = None,
) -> dict[str, dict[str, int]]:
    """Aggregate live allocator segments by default versus graph-private pool."""

    segments = torch.cuda.memory_snapshot() if snapshot is None else snapshot
    result = {
        "default": {"reserved_bytes": 0, "allocated_bytes": 0, "active_bytes": 0},
        "graph_private": {
            "reserved_bytes": 0,
            "allocated_bytes": 0,
            "active_bytes": 0,
        },
    }
    for segment in segments:
        pool = tuple(segment.get("segment_pool_id", (0, 0)))
        category = "default" if pool == (0, 0) else "graph_private"
        result[category]["reserved_bytes"] += int(segment.get("total_size", 0))
        result[category]["allocated_bytes"] += int(
            segment.get("allocated_size", 0)
        )
        result[category]["active_bytes"] += int(segment.get("active_size", 0))
    return result


class StaticCudaMicrobatch:
    """Captured forward/backward for one fixed-shape deterministic microbatch.

    Optimizer mutation and thermal policy deliberately remain outside the graph.
    Gradient buffers are allocated before capture and never replaced, allowing
    repeated replays to preserve ordinary gradient-accumulation semantics.
    """

    def __init__(
        self,
        loss_builder: Callable[..., torch.Tensor],
        example_inputs: Sequence[torch.Tensor],
        parameters: Sequence[torch.nn.Parameter],
        *,
        warmup_steps: int = 3,
        joint_forward_backward: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise CapabilityError("static microbatch execution requires CUDA")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        if not example_inputs:
            raise ValueError("at least one example input is required")
        device = example_inputs[0].device
        if device.type != "cuda" or any(value.device != device for value in example_inputs):
            raise CapabilityError("all static inputs must use one CUDA device")
        unique_parameters = tuple(dict.fromkeys(parameters))
        if not unique_parameters or any(
            not parameter.requires_grad or parameter.device != device
            for parameter in unique_parameters
        ):
            raise CapabilityError(
                "captured parameters must be trainable tensors on the input CUDA device"
            )

        self._loss_builder = loss_builder
        self._inputs = tuple(value.detach().clone() for value in example_inputs)
        self._signatures = tuple(
            TensorSignature(tuple(value.shape), value.dtype, value.device)
            for value in self._inputs
        )
        self._parameters = unique_parameters
        self._device = device

        current = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current)
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup_steps):
                self._clear_gradients(set_to_none=True)
                loss = self._build_loss()
                loss.backward()
        current.wait_stream(warmup_stream)
        current.synchronize()

        # Keep these exact allocations alive: the captured AccumulateGrad nodes
        # write to their addresses on every replay.
        self._clear_gradients(set_to_none=False)
        self._joint_graph = None
        self._forward_graph = None
        self._backward_graph = None
        if joint_forward_backward:
            self._joint_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._joint_graph):
                self._loss = self._build_loss()
                self._loss.backward()
        else:
            self._forward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._forward_graph):
                self._loss = self._build_loss()
            self._backward_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                self._backward_graph, pool=self._forward_graph.pool()
            ):
                self._loss.backward(retain_graph=True)
        self.zero_grad()

    def _build_loss(self) -> torch.Tensor:
        loss = self._loss_builder(*self._inputs)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise CapabilityError("captured loss builder must return one scalar tensor")
        return loss

    def _clear_gradients(self, *, set_to_none: bool) -> None:
        for parameter in self._parameters:
            if parameter.grad is None:
                if not set_to_none:
                    parameter.grad = torch.zeros_like(parameter)
            elif set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    def zero_grad(self) -> None:
        """Zero captured buffers without invalidating their graph addresses."""

        self._clear_gradients(set_to_none=False)

    def replay_forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Stage inputs and replay only the captured forward/loss graph."""

        if self._forward_graph is None:
            raise RuntimeError("joint graph does not expose forward-only replay")

        if len(inputs) != len(self._inputs):
            raise ValueError("static input count changed")
        for source, destination, signature in zip(
            inputs, self._inputs, self._signatures
        ):
            actual = TensorSignature(tuple(source.shape), source.dtype, source.device)
            if actual != signature:
                raise ValueError(
                    "static input signature changed: "
                    f"expected {signature}, received {actual}"
                )
            destination.copy_(source)
        self._forward_graph.replay()
        return self._loss

    def replay_backward(self) -> None:
        """Replay backward after its corresponding forward replay."""

        if self._backward_graph is None:
            raise RuntimeError("joint graph already includes backward replay")
        self._backward_graph.replay()

    def replay(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self._joint_graph is not None:
            if len(inputs) != len(self._inputs):
                raise ValueError("static input count changed")
            for source, destination, signature in zip(
                inputs, self._inputs, self._signatures
            ):
                actual = TensorSignature(tuple(source.shape), source.dtype, source.device)
                if actual != signature:
                    raise ValueError(
                        "static input signature changed: "
                        f"expected {signature}, received {actual}"
                    )
                destination.copy_(source)
            self._joint_graph.replay()
            return self._loss
        loss = self.replay_forward(*inputs)
        self.replay_backward()
        return loss

    @property
    def signatures(self) -> tuple[TensorSignature, ...]:
        return self._signatures
