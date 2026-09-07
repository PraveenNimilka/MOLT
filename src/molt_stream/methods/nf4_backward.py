from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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

    _NF4_BACKWARD_CONFIGS = [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
    ]

    @triton.autotune(
        configs=_NF4_BACKWARD_CONFIGS,
        key=["M", "N", "K", "R", "FUSE_LORA"],
    )
    @triton.jit
    def _nf4_backward_input_kernel(
        grad_ptr,
        packed_ptr,
        scales_ptr,
        code_ptr,
        lora_grad_ptr,
        lora_a_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        stride_om: tl.constexpr,
        stride_ok: tl.constexpr,
        stride_lgm: tl.constexpr,
        stride_lgr: tl.constexpr,
        stride_ar: tl.constexpr,
        stride_ak: tl.constexpr,
        R: tl.constexpr,
        LORA_SCALE: tl.constexpr,
        FUSE_LORA: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            grad = tl.load(
                grad_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
                other=0.0,
            ).to(tl.bfloat16)
            flat = offs_n[:, None] * K + offs_k[None, :]
            packed = tl.load(
                packed_ptr + flat // 2,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0,
            ).to(tl.int32)
            # bitsandbytes stores the first flat value in the high nibble.
            indices = tl.where((flat & 1) == 0, packed >> 4, packed & 15)
            values = tl.load(code_ptr + indices).to(tl.float32)
            scales = tl.load(
                scales_ptr + flat // BLOCK_SIZE,
                mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            weight = (values * scales).to(tl.bfloat16)
            accumulator += tl.dot(grad, weight)

        if FUSE_LORA:
            # LoRA ranks are padded to a Tensor Core compatible inner tile.
            # Masking retains the exact rank-r equation without allocating a
            # padded adapter tensor.
            offs_r = tl.arange(0, 16)
            lora_grad = tl.load(
                lora_grad_ptr
                + offs_m[:, None] * stride_lgm
                + offs_r[None, :] * stride_lgr,
                mask=(offs_m[:, None] < M) & (offs_r[None, :] < R),
                other=0.0,
            ).to(tl.bfloat16)
            lora_a = tl.load(
                lora_a_ptr
                + offs_r[:, None] * stride_ar
                + offs_k[None, :] * stride_ak,
                mask=(offs_r[:, None] < R) & (offs_k[None, :] < K),
                other=0.0,
            ).to(tl.bfloat16)
            # Match PEFT's established BF16 boundary: the frozen-base input
            # gradient is cast before the separately accumulated LoRA term is
            # added.  Combining both in FP32 changes BF16 rounding and fails
            # differential gradient parity even though the real-valued
            # equation is equivalent.
            base_gradient = accumulator.to(tl.bfloat16)
            lora_gradient = (
                tl.dot(lora_grad, lora_a) * LORA_SCALE
            ).to(tl.bfloat16)
            accumulator = (base_gradient + lora_gradient).to(tl.bfloat16)

        tl.store(
            output_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
            accumulator,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        )


def resolved_nf4_scales(quant_state: Any) -> torch.Tensor:
    """Resolve double-quantized block scales once without expanding weights."""
    if getattr(quant_state, "quant_type", None) != "nf4":
        raise CapabilityError("packed backward requires NF4 quantization")
    if int(getattr(quant_state, "blocksize", 0)) != 64:
        raise CapabilityError("packed backward currently requires NF4 block size 64")
    cached = getattr(quant_state, "_molt_resolved_scales", None)
    if isinstance(cached, torch.Tensor) and cached.device == quant_state.absmax.device:
        return cached
    if quant_state.nested:
        import bitsandbytes.functional as bnb_functional

        scales = bnb_functional.dequantize_blockwise(
            quant_state.absmax, quant_state.state2
        ).float()
        scales.add_(float(quant_state.offset))
    else:
        scales = quant_state.absmax.float()
    quant_state._molt_resolved_scales = scales
    return scales


def packed_nf4_backward_input(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_state: Any,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate `grad_output @ dequant(weight)` without dense weight storage."""
    if not _TRITON_AVAILABLE or grad_output.device.type != "cuda":
        raise CapabilityError("packed NF4 backward requires Triton CUDA")
    if grad_output.dtype != torch.bfloat16:
        raise CapabilityError("packed NF4 backward currently requires BF16 gradients")
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    shape = tuple(int(value) for value in quant_state.shape)
    if len(shape) != 2 or grad_output.shape[-1] != shape[0]:
        raise ValueError("gradient width must match the NF4 output dimension")
    n, k = shape
    flat = grad_output.reshape(-1, n)
    resolved = resolved_nf4_scales(quant_state) if scales is None else scales
    output = torch.empty((flat.shape[0], k), device=flat.device, dtype=flat.dtype)
    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(flat.shape[0], meta["BLOCK_M"]),
            triton.cdiv(k, meta["BLOCK_K"]),
        )

    _nf4_backward_input_kernel[grid](
        flat,
        packed_weight,
        resolved,
        quant_state.code,
        flat,
        flat,
        output,
        flat.shape[0],
        n,
        k,
        flat.stride(0),
        flat.stride(1),
        output.stride(0),
        output.stride(1),
        0,
        0,
        0,
        0,
        0.0,
        False,
        1,
        int(quant_state.blocksize),
    )
    return output.reshape(*grad_output.shape[:-1], k)


def packed_nf4_lora_backward_input(
    grad_output: torch.Tensor,
    packed_weight: torch.Tensor,
    quant_state: Any,
    lora_grad: torch.Tensor,
    lora_a: torch.Tensor,
    scaling: float,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse the base NF4 and rank-r LoRA contributions to ``grad_input``."""
    if not _TRITON_AVAILABLE or grad_output.device.type != "cuda":
        raise CapabilityError("packed NF4-LoRA backward requires Triton CUDA")
    if grad_output.dtype != torch.bfloat16 or lora_grad.dtype != torch.bfloat16:
        raise CapabilityError("packed NF4-LoRA backward requires BF16 gradients")
    if lora_a.dtype not in {torch.bfloat16, torch.float32}:
        raise CapabilityError("LoRA A must use BF16 compute or an FP32 master")
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not lora_grad.is_contiguous():
        lora_grad = lora_grad.contiguous()
    shape = tuple(int(value) for value in quant_state.shape)
    if len(shape) != 2 or grad_output.shape[-1] != shape[0]:
        raise ValueError("gradient width must match the NF4 output dimension")
    n, k = shape
    flat = grad_output.reshape(-1, n)
    flat_lora = lora_grad.reshape(-1, lora_grad.shape[-1])
    if flat_lora.shape[0] != flat.shape[0]:
        raise ValueError("LoRA gradient rows must match the output gradient")
    if lora_a.ndim != 2 or flat_lora.shape[1] != lora_a.shape[0] or lora_a.shape[1] != k:
        raise ValueError("LoRA gradient and A matrix dimensions are incompatible")
    if lora_a.shape[0] > 16:
        raise CapabilityError("fused packed NF4-LoRA backward currently supports rank <= 16")
    resolved = resolved_nf4_scales(quant_state) if scales is None else scales
    output = torch.empty((flat.shape[0], k), device=flat.device, dtype=flat.dtype)
    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (
            triton.cdiv(flat.shape[0], meta["BLOCK_M"]),
            triton.cdiv(k, meta["BLOCK_K"]),
        )

    _nf4_backward_input_kernel[grid](
        flat,
        packed_weight,
        resolved,
        quant_state.code,
        flat_lora,
        lora_a,
        output,
        flat.shape[0],
        n,
        k,
        flat.stride(0),
        flat.stride(1),
        output.stride(0),
        output.stride(1),
        flat_lora.stride(0),
        flat_lora.stride(1),
        lora_a.stride(0),
        lora_a.stride(1),
        lora_a.shape[0],
        float(scaling),
        True,
        int(quant_state.blocksize),
    )
    return output.reshape(*grad_output.shape[:-1], k)
