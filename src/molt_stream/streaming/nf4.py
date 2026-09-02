from __future__ import annotations

from dataclasses import dataclass

import torch


# QLoRA's asymmetric NormalFloat-4 lookup values. This implementation is a
# clean-room research primitive, not a bitsandbytes binary-compatible format.
NF4_CODEBOOK = torch.tensor(
    [
        -1.0, -0.6961928, -0.52507305, -0.3949175,
        -0.28444138, -0.18477343, -0.09105004, 0.0,
        0.0795803, 0.1609302, 0.2461123, 0.33791524,
        0.44070983, 0.562617, 0.72295684, 1.0,
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class NF4DevicePayload:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    block_size: int
    elements: int

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        low = self.packed & 0x0F
        high = self.packed >> 4
        indices = torch.stack((low, high), dim=-1).reshape(-1).long()
        values = NF4_CODEBOOK.to(self.packed.device)[indices]
        values = values.reshape(-1, self.block_size) * self.scales[:, None]
        return values.reshape(-1)[: self.elements].reshape(self.shape).to(dtype)


@dataclass(frozen=True)
class NF4Tensor:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    block_size: int
    elements: int

    @classmethod
    def quantize(
        cls, value: torch.Tensor, *, block_size: int = 64, pin_memory: bool = False
    ) -> "NF4Tensor":
        if value.numel() == 0 or block_size < 16 or block_size % 2:
            raise ValueError("invalid tensor or NF4 block size")
        flat = value.detach().float().cpu().reshape(-1)
        elements = flat.numel()
        padding = (-elements) % block_size
        if padding:
            flat = torch.nn.functional.pad(flat, (0, padding))
        blocks = flat.reshape(-1, block_size)
        scales = blocks.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
        normalized = blocks / scales[:, None]
        codebook = NF4_CODEBOOK
        indices = (normalized[..., None] - codebook).abs().argmin(dim=-1).to(torch.uint8)
        packed = indices[:, 0::2] | (indices[:, 1::2] << 4)
        packed = packed.reshape(-1).contiguous()
        scales = scales.contiguous()
        if pin_memory and torch.cuda.is_available():
            packed = packed.pin_memory()
            scales = scales.pin_memory()
        return cls(packed, scales, tuple(value.shape), block_size, elements)

    @property
    def storage_bytes(self) -> int:
        return self.packed.numel() * self.packed.element_size() + self.scales.numel() * 4

    def dequantize(
        self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        target = torch.device(device)
        return self.stage(target, non_blocking=non_blocking).dequantize(dtype)

    def stage(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> NF4DevicePayload:
        target = torch.device(device)
        return NF4DevicePayload(
            self.packed.to(target, non_blocking=non_blocking),
            self.scales.to(target, non_blocking=non_blocking),
            self.shape,
            self.block_size,
            self.elements,
        )
