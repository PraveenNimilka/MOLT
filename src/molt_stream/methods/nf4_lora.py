from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F

from molt_stream.core.errors import CapabilityError


class _ScheduledNF4LoRA(torch.autograd.Function):
    """Experimental first-order NF4 + LoRA manually scheduled backward.

    The NF4 forward remains the installed bitsandbytes GEMM. The custom backward
    computes the frozen-base input gradient and both adapter gradients without
    retaining PEFT's separate LoRA autograd graph. This is an experimental
    scheduling optimization, not a claim that the NF4 GEMM itself is novel.
    Full-model qualification is projection-specific: equivalent real-valued
    derivatives alone do not establish identical BF16 branch accumulation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        quant_state: Any,
        scaling: float,
        decoded_backward_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        import bitsandbytes as bnb

        if x.device.type != "cuda" or packed_weight.device.type != "cuda":
            raise CapabilityError("scheduled NF4-LoRA requires CUDA tensors")
        if packed_weight.requires_grad:
            raise CapabilityError("scheduled NF4-LoRA requires frozen NF4 weights")
        if lora_a.dtype != torch.float32 or lora_b.dtype != torch.float32:
            raise CapabilityError("scheduled NF4-LoRA requires FP32 adapter masters")

        base = bnb.matmul_4bit(
            x,
            packed_weight,
            quant_state=quant_state,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.dtype == torch.bfloat16):
            low_rank = F.linear(x, lora_a)
            result = base + F.linear(low_rank, lora_b) * float(scaling)

        ctx.quant_state = quant_state
        ctx.scaling = float(scaling)
        ctx.input_shape = x.shape
        if decoded_backward_weight is not None and (
            decoded_backward_weight.shape != (lora_b.shape[0], x.shape[-1])
            or decoded_backward_weight.device != x.device
            or decoded_backward_weight.dtype != x.dtype
        ):
            raise CapabilityError(
                "cached NF4 backward weight must match the decoded weight geometry, "
                "device, and compute dtype"
            )
        ctx.has_cached_backward_weight = decoded_backward_weight is not None
        saved = (x, packed_weight, lora_a, lora_b)
        if decoded_backward_weight is not None:
            saved += (decoded_backward_weight,)
        ctx.save_for_backward(*saved)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, None, None, None]:
        x, packed_weight, lora_a, lora_b, *cached = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
        scale = ctx.scaling

        import bitsandbytes.functional as bnb_functional

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=flat_x.dtype == torch.bfloat16
        ):
            low_rank = flat_x @ lora_a.t()
            # Match autograd's scale-before-matmul boundary. Scaling the
            # resulting gradients afterward is not equivalent in BF16.
            scaled_grad = flat_grad * scale
            grad_through_b = scaled_grad @ lora_b
            # The packed contraction failed production-shape differential
            # tests. Use the same dequantize + matmul as bitsandbytes until
            # that kernel independently qualifies. This retains a dense
            # temporary and makes no packed-memory saving claim.
            decoded = (
                cached[0]
                if ctx.has_cached_backward_weight
                else bnb_functional.dequantize_4bit(packed_weight, ctx.quant_state)
            )
            grad_base = flat_grad @ decoded.to(flat_grad.dtype)
            grad_x = grad_base + grad_through_b @ lora_a
            grad_b = scaled_grad.t() @ low_rank
            grad_a = grad_through_b.t() @ flat_x

        return (
            grad_x.reshape(ctx.input_shape),
            None,
            grad_a.to(lora_a.dtype),
            grad_b.to(lora_b.dtype),
            None,
            None,
            None,
        )


def scheduled_nf4_lora(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    quant_state: Any,
    scaling: float,
    decoded_backward_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return _ScheduledNF4LoRA.apply(
        x,
        packed_weight,
        lora_a,
        lora_b,
        quant_state,
        scaling,
        decoded_backward_weight,
    )


def _scheduled_forward(module: torch.nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """PEFT Linear4bit adapter entry point with strict safe fallback."""
    if (
        args or kwargs or module.disable_adapters or module.merged
        or x.dtype != torch.bfloat16
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        return module._molt_original_forward(x, *args, **kwargs)
    active = list(module.active_adapters)
    if len(active) != 1:
        return module._molt_original_forward(x, *args, **kwargs)
    adapter = active[0]
    if (
        adapter not in module.lora_A
        or adapter in module.lora_variant
        or module.use_dora.get(adapter, False)
        or not isinstance(module.lora_dropout[adapter], torch.nn.Identity)
        or module.base_layer.bias is not None
    ):
        return module._molt_original_forward(x, *args, **kwargs)
    return scheduled_nf4_lora(
        x,
        module.base_layer.weight,
        module.lora_A[adapter].weight,
        module.lora_B[adapter].weight,
        module.base_layer.weight.quant_state,
        float(module.scaling[adapter]),
        getattr(module, "_molt_decoded_backward_weight", None),
    )


def enable_scheduled_nf4_lora(
    model: torch.nn.Module,
    *,
    module_suffixes: tuple[str, ...] | None = None,
    require_narrow_output: bool = False,
    cache_backward_weights: bool = False,
) -> int:
    """Enable the experimental path on selected compatible PEFT modules.

    ``module_suffixes`` keeps dispatch tied to projection geometries that have
    passed a randomized component benchmark. A missing suffix list retains the
    original all-compatible-module behavior for explicit research use only;
    that broad mode has not passed full-model trajectory qualification.
    ``cache_backward_weights`` trades dense BF16 residency for eliminating
    repeated dequantization and is never selected implicitly.
    """
    try:
        from peft.tuners.lora.bnb import Linear4bit
    except ImportError as exc:  # pragma: no cover - guarded by QLoRA requirements
        raise CapabilityError("PEFT bitsandbytes Linear4bit support is unavailable") from exc

    patched = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, Linear4bit) or hasattr(module, "_molt_original_forward"):
            continue
        if module_suffixes is not None and not module_name.endswith(module_suffixes):
            continue
        if require_narrow_output and not (
            isinstance(getattr(module, "in_features", None), int)
            and isinstance(getattr(module, "out_features", None), int)
            and module.out_features < module.in_features
        ):
            continue
        module._molt_original_forward = module.forward
        if cache_backward_weights:
            import bitsandbytes.functional as bnb_functional

            with torch.no_grad():
                module._molt_decoded_backward_weight = (
                    bnb_functional.dequantize_4bit(
                        module.base_layer.weight.data,
                        module.base_layer.weight.quant_state,
                    ).to(dtype=torch.bfloat16)
                )
        module.forward = MethodType(_scheduled_forward, module)
        patched += 1
    if patched == 0:
        requested = "all modules" if module_suffixes is None else ", ".join(module_suffixes)
        raise CapabilityError(
            f"No compatible PEFT NF4-LoRA modules were found for {requested}"
            + (" with grouped-query geometry" if require_narrow_output else "")
        )
    return patched
