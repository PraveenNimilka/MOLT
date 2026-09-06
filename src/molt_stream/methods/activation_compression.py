from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import prod
import os
from pathlib import Path

import torch

from molt_stream.core.errors import CapabilityError


def _configure_compiler_cache() -> None:
    root = Path(".c").resolve()
    inductor = root / "i"
    triton_cache = root / "t"
    inductor.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_HOME", str(root))
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(inductor))


_configure_compiler_cache()

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _pack_symmetric_int4_kernel(source, packed, scales, size: tl.constexpr):
        block = tl.program_id(0)
        base = block * 64
        offsets = tl.arange(0, 64)
        mask = base + offsets < size
        values = tl.load(source + base + offsets, mask=mask, other=0.0).to(tl.float32)
        absolute_max = tl.max(tl.abs(values), axis=0)
        scale = tl.maximum(absolute_max / 7.0, 1.0e-12)
        tl.store(scales + block, scale)

        pairs = tl.arange(0, 32)
        even_offsets = base + pairs * 2
        odd_offsets = even_offsets + 1
        even = tl.load(source + even_offsets, mask=even_offsets < size, other=0.0).to(tl.float32)
        odd = tl.load(source + odd_offsets, mask=odd_offsets < size, other=0.0).to(tl.float32)
        even_q = tl.maximum(-7.0, tl.minimum(7.0, tl.floor(even / scale + 0.5))).to(tl.int32) + 8
        odd_q = tl.maximum(-7.0, tl.minimum(7.0, tl.floor(odd / scale + 0.5))).to(tl.int32) + 8
        byte = (even_q << 4) | odd_q
        tl.store(packed + block * 32 + pairs, byte.to(tl.uint8))

    @triton.jit
    def _unpack_symmetric_int4_kernel(packed, scales, output, size: tl.constexpr):
        block = tl.program_id(0)
        pairs = tl.arange(0, 32)
        byte = tl.load(packed + block * 32 + pairs).to(tl.int32)
        scale = tl.load(scales + block).to(tl.float32)
        even = ((byte >> 4) - 8).to(tl.float32) * scale
        odd = ((byte & 15) - 8).to(tl.float32) * scale
        even_offsets = block * 64 + pairs * 2
        odd_offsets = even_offsets + 1
        tl.store(output + even_offsets, even, mask=even_offsets < size)
        tl.store(output + odd_offsets, odd, mask=odd_offsets < size)


@dataclass(frozen=True)
class _CompressedActivation:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype


def _compress(tensor: torch.Tensor) -> _CompressedActivation:
    if not _TRITON_AVAILABLE or tensor.device.type != "cuda":
        raise CapabilityError("4-bit saved-activation compression requires Triton CUDA")
    source = tensor.contiguous().view(-1)
    blocks = (source.numel() + 63) // 64
    packed = torch.empty(blocks * 32, dtype=torch.uint8, device=source.device)
    scales = torch.empty(blocks, dtype=torch.float32, device=source.device)
    _pack_symmetric_int4_kernel[(blocks,)](source, packed, scales, source.numel())
    return _CompressedActivation(packed, scales, tuple(tensor.shape), tensor.dtype)


def _decompress(value: _CompressedActivation) -> torch.Tensor:
    size = prod(value.shape)
    output = torch.empty(size, dtype=value.dtype, device=value.packed.device)
    _unpack_symmetric_int4_kernel[(value.scales.numel(),)](
        value.packed, value.scales, output, size
    )
    return output.view(value.shape)


class CompressedSavedActivations(AbstractContextManager[None]):
    """Experimental lossy 4-bit storage for large saved BF16 activations."""

    def __init__(self, *, minimum_bytes: int = 1 << 20) -> None:
        if minimum_bytes < 0:
            raise ValueError("minimum_bytes must be non-negative")
        if not _TRITON_AVAILABLE:
            raise CapabilityError("4-bit saved-activation compression requires Triton")
        self.minimum_bytes = minimum_bytes
        self._hooks: AbstractContextManager[None] | None = None
        self.compressed_tensor_count = 0
        self.original_bytes = 0
        self.stored_bytes = 0
        self.size_histogram: dict[int, int] = {}

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor | _CompressedActivation:
        size_bytes = tensor.numel() * tensor.element_size()
        if (
            tensor.device.type != "cuda"
            or tensor.dtype not in {torch.bfloat16, torch.float16, torch.float32}
            or tensor.is_leaf
            or size_bytes < self.minimum_bytes
        ):
            return tensor
        compressed = _compress(tensor)
        self.compressed_tensor_count += 1
        self.original_bytes += size_bytes
        stored_bytes = (
            compressed.packed.numel() * compressed.packed.element_size()
            + compressed.scales.numel() * compressed.scales.element_size()
        )
        self.stored_bytes += stored_bytes
        self.size_histogram[size_bytes] = self.size_histogram.get(size_bytes, 0) + 1
        return compressed

    @staticmethod
    def _unpack(value: torch.Tensor | _CompressedActivation) -> torch.Tensor:
        return _decompress(value) if isinstance(value, _CompressedActivation) else value

    def __enter__(self) -> None:
        self._hooks = torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool | None:
        if self._hooks is None:
            raise RuntimeError("activation compression context was not entered")
        return self._hooks.__exit__(exc_type, exc_value, traceback)
