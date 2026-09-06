from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _OffloadedTensor:
    value: torch.Tensor
    device: torch.device


class SavedActivationOffload(AbstractContextManager[None]):
    """Move large, non-leaf autograd saves to pinned CPU memory.

    Parameters and other leaf tensors stay resident.  This is important for
    quantized linear layers: offloading their packed weights would add PCIe
    traffic without reducing activation residency.
    """

    def __init__(self, *, minimum_bytes: int = 1 << 20) -> None:
        if minimum_bytes < 0:
            raise ValueError("minimum_bytes must be non-negative")
        self.minimum_bytes = minimum_bytes
        self._hooks: AbstractContextManager[None] | None = None

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor | _OffloadedTensor:
        size_bytes = tensor.numel() * tensor.element_size()
        if tensor.device.type != "cuda" or tensor.is_leaf or size_bytes < self.minimum_bytes:
            return tensor
        host = torch.empty_like(tensor, device="cpu", pin_memory=True)
        host.copy_(tensor.detach(), non_blocking=True)
        return _OffloadedTensor(host, tensor.device)

    @staticmethod
    def _unpack(value: torch.Tensor | _OffloadedTensor) -> torch.Tensor:
        if isinstance(value, _OffloadedTensor):
            return value.value.to(value.device, non_blocking=True)
        return value

    def __enter__(self) -> None:
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None:
        if self._hooks is None:
            raise RuntimeError("activation offload context was not entered")
        return self._hooks.__exit__(exc_type, exc_value, traceback)


def activation_offload_context(enabled: bool) -> AbstractContextManager[None]:
    return SavedActivationOffload() if enabled else nullcontext()
