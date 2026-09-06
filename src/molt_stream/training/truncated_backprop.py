"""Experimental upper-decoder adapter placement and exact graph truncation."""
from __future__ import annotations

import torch

from molt_stream.core.errors import CapabilityError


QWEN2_LINEAR_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def upper_layer_indices(layer_count: int, train_last_layers: int) -> list[int]:
    if type(layer_count) is not int or type(train_last_layers) is not int:
        raise ValueError("layer counts must be integers")
    if not 1 <= train_last_layers <= layer_count:
        raise ValueError("train_last_layers must be within the decoder depth")
    return list(range(layer_count - train_last_layers, layer_count))


def truncate_before_decoder_layer(model: torch.nn.Module, layer_index: int) -> None:
    """Detach hidden states at a frozen/trainable decoder boundary.

    The hook is retained by the module. It changes credit assignment by design:
    only adapters at and above the boundary receive gradients.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(getattr(base, "config", None), "model_type", None) != "qwen2":
        raise CapabilityError("Truncated adapter backprop currently requires Qwen2")
    layers = base.model.layers
    if not 0 <= layer_index < len(layers):
        raise ValueError("layer_index is outside the decoder")

    def detach_hidden(_module: torch.nn.Module, args: tuple[object, ...]) -> tuple[object, ...]:
        if not args or not isinstance(args[0], torch.Tensor):
            raise CapabilityError("Qwen2 decoder hidden state was not positional")
        return (args[0].detach(), *args[1:])

    layers[layer_index].register_forward_pre_hook(detach_hidden)


def activate_upper_layer_training(model: torch.nn.Module, train_last_layers: int) -> int:
    """Freeze lower LoRA tensors and truncate their now-unused backward graph."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layer_count = int(base.config.num_hidden_layers)
    boundary = layer_count - train_last_layers
    upper_layer_indices(layer_count, train_last_layers)
    frozen = 0
    for layer in base.model.layers[:boundary]:
        for name, parameter in layer.named_parameters():
            if "lora_" in name and parameter.requires_grad:
                parameter.requires_grad_(False)
                frozen += parameter.numel()
    truncate_before_decoder_layer(model, boundary)
    return frozen
