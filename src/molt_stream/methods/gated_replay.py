from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from torch.nn import functional as F

from molt_stream.core.errors import CapabilityError


@dataclass
class _RMSNormReplayRecipe:
    source: torch.Tensor
    weight: torch.Tensor
    eps: float
    weight_offset: float
    packed_count: int = 0
    target: torch.Tensor | None = None

    def materialize(self) -> torch.Tensor:
        values = self.source.float()
        inverse_rms = torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (
            values * inverse_rms * (self.weight.float() + self.weight_offset)
        ).to(torch.bfloat16)


def _replayable_norm_forward(
    module: torch.nn.Module, hidden: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    result = module._molt_original_forward(hidden, *args, **kwargs)
    target_mlp = module._molt_target_mlp
    if torch.is_grad_enabled() and result.dtype == torch.bfloat16:
        target_mlp._molt_norm_replay_recipe = _RMSNormReplayRecipe(
            source=hidden,
            weight=module.weight,
            eps=float(module.variance_epsilon),
            weight_offset=float(module.weight_offset),
            target=result,
        )
    else:
        target_mlp._molt_norm_replay_recipe = None
    return result


class _ResidualRecipeGatedNF4LoRA(torch.autograd.Function):
    """Gated NF4+LoRA MLP that retains the residual instead of normalized input."""

    @staticmethod
    def forward(
        ctx: Any,
        source: torch.Tensor,
        normalized: torch.Tensor,
        gate_packed_weight: torch.Tensor,
        gate_lora_a: torch.Tensor,
        gate_lora_b: torch.Tensor,
        up_packed_weight: torch.Tensor,
        up_lora_a: torch.Tensor,
        up_lora_b: torch.Tensor,
        down_packed_weight: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
        gate_quant_state: Any,
        up_quant_state: Any,
        down_quant_state: Any,
        gate_scaling: float,
        up_scaling: float,
        down_scaling: float,
        norm_weight: torch.Tensor,
        norm_eps: float,
        norm_weight_offset: float,
    ) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gate = _nf4_lora_forward(
                normalized,
                gate_packed_weight,
                gate_lora_a,
                gate_lora_b,
                gate_quant_state,
                gate_scaling,
            )
            up = _nf4_lora_forward(
                normalized,
                up_packed_weight,
                up_lora_a,
                up_lora_b,
                up_quant_state,
                up_scaling,
            )
            gated = F.silu(gate) * up
            result = _nf4_lora_forward(
                gated,
                down_packed_weight,
                down_lora_a,
                down_lora_b,
                down_quant_state,
                down_scaling,
            )
        ctx.gate_quant_state = gate_quant_state
        ctx.up_quant_state = up_quant_state
        ctx.down_quant_state = down_quant_state
        ctx.gate_scaling = float(gate_scaling)
        ctx.up_scaling = float(up_scaling)
        ctx.down_scaling = float(down_scaling)
        ctx.norm_eps = float(norm_eps)
        ctx.norm_weight_offset = float(norm_weight_offset)
        ctx.save_for_backward(
            source,
            gate,
            up,
            gate_packed_weight,
            gate_lora_a,
            gate_lora_b,
            up_packed_weight,
            up_lora_a,
            up_lora_b,
            down_packed_weight,
            down_lora_a,
            down_lora_b,
            norm_weight,
        )
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        (
            source,
            gate,
            up,
            gate_packed_weight,
            gate_lora_a,
            gate_lora_b,
            up_packed_weight,
            up_lora_a,
            up_lora_b,
            down_packed_weight,
            down_lora_a,
            down_lora_b,
            norm_weight,
        ) = ctx.saved_tensors
        normalized = _RMSNormReplayRecipe(
            source,
            norm_weight,
            ctx.norm_eps,
            ctx.norm_weight_offset,
        ).materialize()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            activated_gate = F.silu(gate)
            gated = activated_gate * up
        grad_gated, grad_down_a, grad_down_b = _nf4_lora_backward(
            grad_output,
            gated,
            down_packed_weight,
            down_lora_a,
            down_lora_b,
            ctx.down_quant_state,
            ctx.down_scaling,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            grad_up = grad_gated * activated_gate
            grad_gate = torch.ops.aten.silu_backward(grad_gated * up, gate)
        gate_grad_x, grad_gate_a, grad_gate_b = _nf4_lora_backward(
            grad_gate,
            normalized,
            gate_packed_weight,
            gate_lora_a,
            gate_lora_b,
            ctx.gate_quant_state,
            ctx.gate_scaling,
        )
        up_grad_x, grad_up_a, grad_up_b = _nf4_lora_backward(
            grad_up,
            normalized,
            up_packed_weight,
            up_lora_a,
            up_lora_b,
            ctx.up_quant_state,
            ctx.up_scaling,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            grad_normalized = gate_grad_x + up_grad_x
        return (
            None,
            grad_normalized,
            None,
            grad_gate_a,
            grad_gate_b,
            None,
            grad_up_a,
            grad_up_b,
            None,
            grad_down_a,
            grad_down_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _nf4_lora_forward(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    quant_state: Any,
    scaling: float,
) -> torch.Tensor:
    import bitsandbytes as bnb

    base = bnb.matmul_4bit(x, packed_weight, quant_state=quant_state)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        low_rank = F.linear(x, lora_a)
        return base + F.linear(low_rank, lora_b) * float(scaling)


def _nf4_lora_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    quant_state: Any,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import bitsandbytes.functional as bnb_functional

    flat_x = x.reshape(-1, x.shape[-1])
    flat_grad = grad_output.reshape(-1, grad_output.shape[-1])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        low_rank = flat_x @ lora_a.t()
        scaled_grad = flat_grad * float(scaling)
        grad_through_b = scaled_grad @ lora_b
        decoded = bnb_functional.dequantize_4bit(packed_weight, quant_state)
        grad_base = flat_grad @ decoded.to(flat_grad.dtype)
        grad_x = grad_base + grad_through_b @ lora_a
        grad_b = scaled_grad.t() @ low_rank
        grad_a = grad_through_b.t() @ flat_x
    return (
        grad_x.reshape(x.shape),
        grad_a.to(lora_a.dtype),
        grad_b.to(lora_b.dtype),
    )


class _AsymmetricGatedNF4LoRA(torch.autograd.Function):
    """Exact gated-MLP schedule that rematerializes only the up branch.

    Forward receives the ordinary gate/up projection results so their existing
    autograd nodes remain authoritative for those projection gradients. It
    retains the gate result but not the equally large up result, SiLU result,
    or gated product. Backward rematerializes up once, reconstructs the gated
    product, and uses the qualified scheduled down-projection equations.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        gate: torch.Tensor,
        up: torch.Tensor,
        up_packed_weight: torch.Tensor,
        up_lora_a: torch.Tensor,
        up_lora_b: torch.Tensor,
        down_packed_weight: torch.Tensor,
        down_lora_a: torch.Tensor,
        down_lora_b: torch.Tensor,
        up_quant_state: Any,
        down_quant_state: Any,
        up_scaling: float,
        down_scaling: float,
        rematerialize_up: bool,
        offload_up: bool,
        host_up: torch.Tensor,
        offload_gate: bool,
        host_gate: torch.Tensor,
    ) -> torch.Tensor:
        if x.device.type != "cuda" or x.dtype != torch.bfloat16:
            raise CapabilityError("asymmetric gated replay requires CUDA BF16 tensors")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gated = F.silu(gate) * up
            result = _nf4_lora_forward(
                gated,
                down_packed_weight,
                down_lora_a,
                down_lora_b,
                down_quant_state,
                down_scaling,
            )
        ctx.up_quant_state = up_quant_state
        ctx.down_quant_state = down_quant_state
        ctx.up_scaling = float(up_scaling)
        ctx.down_scaling = float(down_scaling)
        ctx.rematerialize_up = bool(rematerialize_up)
        ctx.offload_up = bool(offload_up)
        ctx.offload_gate = bool(offload_gate)
        if ctx.rematerialize_up and ctx.offload_up:
            raise CapabilityError("up projection cannot be replayed and offloaded")
        if ctx.offload_up:
            if host_up.device.type != "cpu" or not host_up.is_pinned():
                raise CapabilityError("gated activation offload requires pinned host storage")
            if host_up.shape != up.shape or host_up.dtype != up.dtype:
                raise CapabilityError("gated activation host storage must match the up result")
        if ctx.offload_gate:
            if host_gate.device.type != "cpu" or not host_gate.is_pinned():
                raise CapabilityError("gated activation offload requires pinned gate storage")
            if host_gate.shape != gate.shape or host_gate.dtype != gate.dtype:
                raise CapabilityError("gated activation host storage must match the gate result")
        if ctx.offload_up or ctx.offload_gate:
            if ctx.offload_up:
                host_up.copy_(up.detach(), non_blocking=True)
            if ctx.offload_gate:
                host_gate.copy_(gate.detach(), non_blocking=True)
        saved = (
            x,
            host_gate if ctx.offload_gate else gate,
            up_packed_weight,
            up_lora_a,
            up_lora_b,
            down_packed_weight,
            down_lora_a,
            down_lora_b,
        )
        if ctx.offload_up:
            saved += (host_up,)
        elif not ctx.rematerialize_up:
            saved += (up,)
        ctx.save_for_backward(*saved)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        (
            x,
            stored_gate,
            up_packed_weight,
            up_lora_a,
            up_lora_b,
            down_packed_weight,
            down_lora_a,
            down_lora_b,
            *retained,
        ) = ctx.saved_tensors
        gate = (
            stored_gate.to(grad_output.device, non_blocking=True)
            if ctx.offload_gate
            else stored_gate
        )
        if ctx.rematerialize_up:
            up = _nf4_lora_forward(
                x,
                up_packed_weight,
                up_lora_a,
                up_lora_b,
                ctx.up_quant_state,
                ctx.up_scaling,
            )
        elif ctx.offload_up:
            up = retained[0].to(grad_output.device, non_blocking=True)
        else:
            up = retained[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            activated_gate = F.silu(gate)
            gated = activated_gate * up
        grad_gated, grad_down_a, grad_down_b = _nf4_lora_backward(
            grad_output,
            gated,
            down_packed_weight,
            down_lora_a,
            down_lora_b,
            ctx.down_quant_state,
            ctx.down_scaling,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            grad_up = grad_gated * activated_gate
            grad_gate = torch.ops.aten.silu_backward(grad_gated * up, gate)
        return (
            None,
            grad_gate,
            grad_up,
            None,
            None,
            None,
            None,
            grad_down_a,
            grad_down_b,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def asymmetric_gated_nf4_lora(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    up_packed_weight: torch.Tensor,
    up_lora_a: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_packed_weight: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    up_quant_state: Any,
    down_quant_state: Any,
    up_scaling: float,
    down_scaling: float,
    rematerialize_up: bool = True,
    offload_up: bool = False,
    host_up: torch.Tensor | None = None,
    offload_gate: bool = False,
    host_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    if offload_up and host_up is None:
        raise CapabilityError("gated activation offload requires a host buffer")
    if host_up is None:
        host_up = torch.empty(0)
    if offload_gate and host_gate is None:
        raise CapabilityError("gated activation offload requires a gate host buffer")
    if host_gate is None:
        host_gate = torch.empty(0)
    return _AsymmetricGatedNF4LoRA.apply(
        x,
        gate,
        up,
        up_packed_weight,
        up_lora_a,
        up_lora_b,
        down_packed_weight,
        down_lora_a,
        down_lora_b,
        up_quant_state,
        down_quant_state,
        up_scaling,
        down_scaling,
        rematerialize_up,
        offload_up,
        host_up,
        offload_gate,
        host_gate,
    )


def _adapter_parts(module: torch.nn.Module) -> tuple[Any, ...] | None:
    active = list(module.active_adapters)
    if len(active) != 1:
        return None
    adapter = active[0]
    if (
        adapter not in module.lora_A
        or adapter in module.lora_variant
        or module.use_dora.get(adapter, False)
        or not isinstance(module.lora_dropout[adapter], torch.nn.Identity)
        or module.base_layer.bias is not None
    ):
        return None
    return (
        module.base_layer.weight,
        module.lora_A[adapter].weight,
        module.lora_B[adapter].weight,
        module.base_layer.weight.quant_state,
        float(module.scaling[adapter]),
    )


def _asymmetric_mlp_forward(
    module: torch.nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any
) -> torch.Tensor:
    if (
        args
        or kwargs
        or not torch.is_grad_enabled()
        or x.dtype != torch.bfloat16
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
    ):
        return module._molt_original_forward(x, *args, **kwargs)
    up_parts = _adapter_parts(module.up_proj)
    down_parts = _adapter_parts(module.down_proj)
    if up_parts is None or down_parts is None:
        return module._molt_original_forward(x, *args, **kwargs)
    gate = module.gate_proj(x)
    up = module.up_proj(x)
    if module._molt_offload_up:
        host_up = getattr(module, "_molt_host_up", None)
        if host_up is None or host_up.shape != up.shape or host_up.dtype != up.dtype:
            if torch.cuda.is_current_stream_capturing():
                raise CapabilityError("gated activation host storage was not warmed up")
            host_up = torch.empty_like(up, device="cpu", pin_memory=True)
            module._molt_host_up = host_up
    else:
        host_up = None
    if module._molt_offload_gate:
        host_gate = getattr(module, "_molt_host_gate", None)
        if host_gate is None or host_gate.shape != gate.shape or host_gate.dtype != gate.dtype:
            if torch.cuda.is_current_stream_capturing():
                raise CapabilityError("gated activation gate storage was not warmed up")
            host_gate = torch.empty_like(gate, device="cpu", pin_memory=True)
            module._molt_host_gate = host_gate
    else:
        host_gate = None
    return asymmetric_gated_nf4_lora(
        x,
        gate,
        up,
        up_parts[0],
        up_parts[1],
        up_parts[2],
        down_parts[0],
        down_parts[1],
        down_parts[2],
        up_parts[3],
        down_parts[3],
        up_parts[4],
        down_parts[4],
        module._molt_rematerialize_up,
        module._molt_offload_up,
        host_up,
        module._molt_offload_gate,
        host_gate,
    )


def enable_asymmetric_gated_replay(
    model: torch.nn.Module,
    *,
    rematerialize_up: bool = True,
    rematerialize_blocks: int | None = None,
    offload_blocks: int = 0,
    offload_gate_blocks: int = 0,
) -> int:
    """Patch structurally compatible SiLU-gated PEFT NF4 MLP blocks.

    Dispatch is model-class agnostic but intentionally capability gated. This
    covers the common gate/up/down SwiGLU structure without claiming support
    for arbitrary activations, biases, adapter variants, or dense projections.
    """

    try:
        from peft.tuners.lora.bnb import Linear4bit
    except ImportError as exc:  # pragma: no cover
        raise CapabilityError("PEFT bitsandbytes Linear4bit support is unavailable") from exc

    if rematerialize_blocks is not None and rematerialize_blocks < 0:
        raise ValueError("rematerialize_blocks cannot be negative")
    if offload_blocks < 0:
        raise ValueError("offload_blocks cannot be negative")
    if offload_gate_blocks < 0:
        raise ValueError("offload_gate_blocks cannot be negative")
    if rematerialize_up and (offload_blocks or offload_gate_blocks):
        raise ValueError("up projection replay and offload are mutually exclusive")
    compatible: list[torch.nn.Module] = []
    for module in model.modules():
        if hasattr(module, "_molt_original_forward"):
            continue
        if not all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            continue
        if not all(isinstance(getattr(module, name), Linear4bit) for name in ("gate_proj", "up_proj", "down_proj")):
            continue
        if type(module.act_fn).__name__ not in {"SiLU", "SiLUActivation"}:
            continue
        if _adapter_parts(module.up_proj) is None or _adapter_parts(module.down_proj) is None:
            continue
        compatible.append(module)
    if not compatible:
        raise CapabilityError("No compatible SiLU-gated PEFT NF4 MLP blocks were found")
    if rematerialize_blocks is not None and rematerialize_blocks > len(compatible):
        raise CapabilityError(
            f"requested {rematerialize_blocks} replay blocks but only "
            f"{len(compatible)} compatible blocks were found"
        )
    if offload_blocks > len(compatible):
        raise CapabilityError(
            f"requested {offload_blocks} offload blocks but only "
            f"{len(compatible)} compatible blocks were found"
        )
    if offload_gate_blocks > len(compatible):
        raise CapabilityError(
            f"requested {offload_gate_blocks} gate offload blocks but only "
            f"{len(compatible)} compatible blocks were found"
        )
    selected = len(compatible) if rematerialize_blocks is None else rematerialize_blocks
    patched = 0
    for index, module in enumerate(compatible):
        module._molt_original_forward = module.forward
        module._molt_rematerialize_up = bool(rematerialize_up and index < selected)
        module._molt_offload_up = bool(index < offload_blocks)
        module._molt_offload_gate = bool(index < offload_gate_blocks)
        module.forward = MethodType(_asymmetric_mlp_forward, module)
        patched += 1
    return patched
