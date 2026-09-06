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

    @triton.jit
    def _nf4_backward_input_kernel(
        grad_ptr,
        packed_ptr,
        scales_ptr,
        code_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_gm: tl.constexpr,
        stride_gn: tl.constexpr,
        stride_om: tl.constexpr,
        stride_ok: tl.constexpr,
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
    # Reuse each decoded NF4 tile across a larger accumulator. The initial
    # 16x32x32 screen redundantly decoded weights across too many programs.
    block_m, block_n, block_k = 512, 32, 16
    _nf4_backward_input_kernel[(triton.cdiv(flat.shape[0], block_m), triton.cdiv(k, block_k))](
        flat,
        packed_weight,
        resolved,
        quant_state.code,
        output,
        flat.shape[0],
        n,
        k,
        flat.stride(0),
        flat.stride(1),
        output.stride(0),
        output.stride(1),
        int(quant_state.blocksize),
        block_m,
        block_n,
        block_k,
        num_warps=4,
    )
    return output.reshape(*grad_output.shape[:-1], k)
