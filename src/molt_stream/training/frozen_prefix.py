"""Experimental, bounded exact-window replay of an immutable Qwen2 prefix.

Frozen activation caching is established prior art (AutoFreeze/PipeTransformer).
This narrow implementation is for controlled repeated-epoch experiments only.
It is not KV caching, approximate training, or a general-model execution backend.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from types import TracebackType
from typing import Any

import torch

from molt_stream.core.errors import CapabilityError


class FrozenPrefixReplay:
    """Temporarily bypass frozen decoder blocks on exact input-window hits.

    Entries retain the original dtype on CPU and are bounded by tensor bytes.
    Keys/Python bookkeeping add overhead beyond that bound. No cache is saved in
    checkpoints: reconstructing it changes runtime cost, not optimizer state.
    Concurrent forwards, custom positions/masks, stochastic prefixes and weight
    mutation are unsupported. Direct `.data` mutation must never be used.
    """

    def __init__(self, model: torch.nn.Module, *, boundary: int, max_bytes: int):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        if getattr(base.config, "model_type", None) != "qwen2":
            raise CapabilityError("Frozen-prefix replay currently requires Qwen2")
        if type(boundary) is not int or not 0 < boundary < len(base.model.layers):
            raise ValueError("boundary must leave both frozen and trainable decoder blocks")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.body = base.model
        self.prefix = list(self.body.layers[:boundary])
        self.modules = [self.body.embed_tokens, *self.prefix]
        if any(p.requires_grad for module in self.modules for p in module.parameters()):
            raise CapabilityError("Embedding and prefix parameters must be frozen")
        if (getattr(base.config, "attention_dropout", 0) != 0
                or any(isinstance(m, torch.nn.Dropout) and m.p != 0
                       for module in self.modules for m in module.modules())):
            raise CapabilityError("Prefix dropout must be zero")
        rope = getattr(base.config, "rope_scaling", None)
        if rope is not None and rope.get("rope_type", rope.get("type")) != "default":
            raise CapabilityError("Dynamic/scaled RoPE is not verified for prefix replay")
        self.max_bytes = max_bytes
        self.resident_bytes = 0
        self.hits = self.misses = 0
        self._entries: OrderedDict[tuple[object, ...], torch.Tensor] = OrderedDict()
        self._originals: list[tuple[torch.nn.Module, bool, Callable[..., Any]]] = []
        self._fingerprint = self._versions()
        self._config = repr(base.config)
        self._base = base
        self._active = False
        self._key: tuple[object, ...] | None = None
        self._hit: torch.Tensor | None = None

    def _versions(self) -> tuple[object, ...]:
        return tuple((id(t), t._version, t.requires_grad, t.dtype, t.device, tuple(t.shape))
                     for module in self.modules
                     for t in (*module.parameters(), *module.buffers()))

    def _install(self, module: torch.nn.Module, replacement: Callable[..., Any]) -> None:
        self._originals.append((module, "forward" in module.__dict__, module.forward))
        module.forward = replacement

    def __enter__(self) -> FrozenPrefixReplay:
        if self._originals:
            raise RuntimeError("Replay context is already installed")
        original_body = self.body.forward

        def forward_body(*args: Any, **kwargs: Any) -> Any:
            if (args or set(kwargs) - {"input_ids", "use_cache", "return_dict"}
                    or kwargs.get("use_cache") is not False
                    or kwargs.get("return_dict") is not True):
                raise CapabilityError("Replay arguments require explicit input_ids, use_cache=False, return_dict=True")
            if self._active:
                raise CapabilityError("Concurrent or recursive replay is unsupported")
            if self._versions() != self._fingerprint or repr(self._base.config) != self._config:
                raise CapabilityError("Frozen prefix or model configuration changed; discard the replay cache")
            x = kwargs.get("input_ids")
            if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.dtype != torch.long:
                raise CapabilityError("Replay requires a rank-two int64 token tensor")
            device_type = x.device.type
            key = (tuple(x.shape), str(x.device), self.body.training,
                   torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type),
                   torch.get_float32_matmul_precision(),
                   torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                   torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
                   torch.backends.cuda.math_sdp_enabled(),
                   x.detach().cpu().contiguous().numpy().tobytes())
            self._active = True
            self._key = key
            self._hit = self._entries.get(key)
            if self._hit is None:
                self.misses += 1
            else:
                self.hits += 1
                self._entries.move_to_end(key)
            try:
                return original_body(**kwargs)
            finally:
                self._key = self._hit = None
                self._active = False

        self._install(self.body, forward_body)
        for index, layer in enumerate(self.prefix):
            original = layer.forward

            def forward_layer(*args: Any, _original=original, _last=index == len(self.prefix) - 1,
                              **kwargs: Any) -> torch.Tensor:
                if not self._active:
                    raise CapabilityError("Prefix replay outside its decoder call is unsupported")
                hidden = args[0] if args else kwargs["hidden_states"]
                if self._hit is not None:
                    return self._hit.to(hidden.device) if _last else hidden
                output = _original(*args, **kwargs)
                if not isinstance(output, torch.Tensor):
                    raise CapabilityError("Replay requires tensor-returning Qwen2 decoder blocks")
                if _last:
                    size = output.numel() * output.element_size()
                    if size <= self.max_bytes:
                        while self.resident_bytes + size > self.max_bytes:
                            _, evicted = self._entries.popitem(last=False)
                            self.resident_bytes -= evicted.numel() * evicted.element_size()
                        if self._key is None:
                            raise CapabilityError("Replay cache key was not initialized")
                        self._entries[self._key] = output.detach().to(device="cpu", copy=True)
                        self.resident_bytes += size
                return output

            self._install(layer, forward_layer)
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, traceback: TracebackType | None) -> None:
        for module, had_override, original in reversed(self._originals):
            if had_override:
                module.forward = original
            else:
                del module.forward
        self._originals.clear()
        self._entries.clear()
        self.resident_bytes = 0
