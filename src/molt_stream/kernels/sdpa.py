from __future__ import annotations

import torch
from types import SimpleNamespace

from molt_stream.core.errors import CapabilityError


_MOLT_SDPA_NAME = "molt_bf16_sdpa"


def _bf16_sdpa_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs,
):
    """Narrow the Q/K/V boundary before delegating attention semantics.

    RoPE tables are commonly FP32 and therefore promote Q/K even under
    autocast. Fused CUDA SDPA implementations require a uniform low-precision
    Q/K/V contract. The cast happens after RoPE, so positional arithmetic is
    still evaluated in the model's chosen precision.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)
    delegated_module = module
    selected_kernel = getattr(
        getattr(module, "config", None), "_molt_sdpa_kernel", None
    )
    if selected_kernel == "efficient" and query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1] != 0 or key.shape[1] != value.shape[1]:
            raise CapabilityError("SDPA query/key/value head geometry is incompatible")
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        # Transformers otherwise requests native GQA, which Windows' efficient
        # SDPA backend does not implement. The repeated tensors have ordinary
        # multi-head geometry, so expose that exact contract to the delegate.
        delegated_module = SimpleNamespace(
            num_key_value_groups=1,
            is_causal=getattr(module, "is_causal", True),
        )

    return sdpa_attention_forward(
        delegated_module,
        query,
        key,
        value,
        attention_mask,
        **kwargs,
    )


def enable_bf16_fused_sdpa_boundary(model: torch.nn.Module, kernel: str) -> None:
    """Register MOLT's BF16 attention boundary for a supported decoder."""
    from transformers import AttentionInterface
    from molt_stream.kernels.execution_plan import resolve_decoder_architecture

    resolve_decoder_architecture(model)
    config = getattr(model, "config", None)
    if config is None or getattr(config, "_attn_implementation", None) != "sdpa":
        raise CapabilityError("fused SDPA boundary requires a model loaded with SDPA")
    if kernel not in {"cudnn", "efficient"}:
        raise CapabilityError("BF16 fused SDPA boundary requires cudnn or efficient")
    AttentionInterface.register(_MOLT_SDPA_NAME, _bf16_sdpa_forward)
    config._molt_sdpa_kernel = kernel
    config._attn_implementation = _MOLT_SDPA_NAME
