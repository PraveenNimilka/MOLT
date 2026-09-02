from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from molt_stream.core.errors import CapabilityError
from molt_stream.core.specs import StreamSpec
from molt_stream.streaming.nf4 import NF4DevicePayload, NF4Tensor


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


class _FrozenNF4LoRA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lora_a, lora_b, staged_weight, source, scaling):
        ctx.source = source
        ctx.scaling = scaling
        ctx.weight_dtype = staged_weight.dtype
        ctx.adapter_dtypes = (lora_a.dtype, lora_b.dtype)
        ctx.save_for_backward(x, lora_a, lora_b)
        active_a = lora_a.to(x.dtype)
        active_b = lora_b.to(x.dtype)
        return torch.nn.functional.linear(x, staged_weight) + scaling * torch.nn.functional.linear(
            torch.nn.functional.linear(x, active_a), active_b
        )

    @staticmethod
    def backward(ctx, grad_output):
        x, lora_a, lora_b = ctx.saved_tensors
        weight = ctx.source.dequantize(device=x.device, dtype=ctx.weight_dtype)
        x2 = x.reshape(-1, x.shape[-1])
        grad2 = grad_output.reshape(-1, grad_output.shape[-1])
        active_a = lora_a.to(x2.dtype)
        active_b = lora_b.to(grad2.dtype)
        xa = x2 @ active_a.t()
        grad_x = grad2 @ weight + ctx.scaling * (grad2 @ active_b) @ active_a
        grad_a = ctx.scaling * (grad2 @ active_b).t() @ x2
        grad_b = ctx.scaling * grad2.t() @ xa
        return (
            grad_x.reshape_as(x), grad_a.to(ctx.adapter_dtypes[0]),
            grad_b.to(ctx.adapter_dtypes[1]), None, None, None,
        )


class StreamedLoRALinear(nn.Module):
    """Frozen host NF4 base with GPU-resident trainable LoRA parameters."""

    def __init__(self, source: NF4Tensor, rank: int, alpha: float, device: torch.device):
        super().__init__()
        if len(source.shape) != 2:
            raise ValueError("streamed linear base must be a matrix")
        out_features, in_features = source.shape
        self.source = source
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, device=device))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x: torch.Tensor, staged_weight: torch.Tensor) -> torch.Tensor:
        return _FrozenNF4LoRA.apply(
            x, self.lora_a, self.lora_b, staged_weight, self.source, self.scaling
        )


@dataclass
class StreamedLayer:
    linear: StreamedLoRALinear

    @property
    def source(self) -> NF4Tensor:
        return self.linear.source

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.silu(self.linear(x, weight))


class StreamedLayerEngine(nn.Module):
    """Grouped double-buffered forward streaming with backward re-streaming.

    NF4 payloads are transferred in bundles so one stream dependency covers
    several decoder blocks. Frozen base weights remain NF4 in host memory;
    LoRA parameters remain resident on the compute device. Dequantization stays
    on the compute stream so it cannot race the single reusable copy stream.

    CUDA graph capture is deliberately rejected here. The custom backward
    re-streams a different host tensor for every layer, while graphed callables
    require fixed tensor addresses. Capturing that path would silently replay
    the captured layer's host address for other layers. A future implementation
    may graph compute against persistent device bundle slots, but those slots
    must be counted against the VRAM gate first.
    """

    def __init__(self, layers: Iterable[StreamedLayer], spec: StreamSpec):
        super().__init__()
        spec.validate()
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("at least one streamed layer is required")
        self.spec = spec
        requested = torch.device(spec.device)
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if requested.type == "cuda" and torch.cuda.is_available()
            else requested
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CapabilityError("CUDA streaming requested but CUDA is unavailable")
        self.adapters = nn.ModuleList(layer.linear for layer in self.layers)
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.last_forward: dict[str, float | int] = {}
        if spec.cuda_graphs:
            raise CapabilityError(
                "CUDA graphs are unsafe for the current host-restreaming backward: "
                "make_graphed_callables requires fixed tensor addresses"
            )

    def _stage(self, source: NF4Tensor) -> NF4DevicePayload:
        return source.stage(self.device, non_blocking=self.device.type == "cuda")

    def _bundles(self) -> list[Sequence[StreamedLayer]]:
        size = self.spec.bundle_size
        return [self.layers[index:index + size] for index in range(0, len(self.layers), size)]

    def _stage_bundle(self, bundle: Sequence[StreamedLayer]) -> list[NF4DevicePayload]:
        return [self._stage(layer.source) for layer in bundle]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != self.device.type or (
            self.device.type == "cuda" and x.device.index != self.device.index
        ):
            raise ValueError(f"input must be on {self.device}")
        started = time.perf_counter()
        transfer_seconds = 0.0
        if self.copy_stream is None or not self.spec.double_buffer:
            for layer in self.layers:
                transfer_started = time.perf_counter()
                payload = self._stage(layer.source)
                transfer_seconds += time.perf_counter() - transfer_started
                x = layer(x, payload.dequantize(_dtype(self.spec.compute_dtype)))
        else:
            bundles = self._bundles()
            with torch.cuda.stream(self.copy_stream):
                staged_next = self._stage_bundle(bundles[0])
            stream_waits = 0
            for bundle_index, bundle in enumerate(bundles):
                torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
                stream_waits += 1
                payloads = staged_next
                if bundle_index + 1 < len(bundles):
                    with torch.cuda.stream(self.copy_stream):
                        staged_next = self._stage_bundle(bundles[bundle_index + 1])
                for layer, payload in zip(bundle, payloads, strict=True):
                    staged = payload.dequantize(_dtype(self.spec.compute_dtype))
                    x = layer(x, staged)
                    payload.packed.record_stream(torch.cuda.current_stream(self.device))
                    payload.scales.record_stream(torch.cuda.current_stream(self.device))
        self.last_forward = {
            "layers": len(self.layers),
            "bundle_size": self.spec.bundle_size,
            "bundles": (len(self._bundles()) if self.spec.double_buffer else len(self.layers)),
            "stream_waits": (stream_waits if self.copy_stream is not None and self.spec.double_buffer else 0),
            "seconds_enqueued": time.perf_counter() - started,
            "synchronous_transfer_seconds": transfer_seconds,
        }
        return x
