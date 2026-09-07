from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.training.decoder_runtime import (
    prepare_static_decoder_inputs_from_hidden,
    run_static_decoder_layer,
)


def _stage_activation(value: torch.Tensor, storage: str) -> torch.Tensor:
    detached = value.detach()
    if storage == "cuda":
        return detached.clone()
    if detached.device.type == "cpu":
        return detached.clone()
    return detached.to("cpu", non_blocking=False)


def _restore_activation(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=False).requires_grad_(True)


def _parameter_layout(
    decoder: torch.nn.Module,
) -> tuple[tuple[torch.nn.Parameter, ...], tuple[tuple[int, ...], ...]]:
    trainable = tuple(parameter for parameter in decoder.parameters() if parameter.requires_grad)
    if not trainable:
        raise CapabilityError("layer-major execution requires trainable decoder parameters")
    indices = {id(parameter): index for index, parameter in enumerate(trainable)}
    per_layer: list[tuple[int, ...]] = []
    observed: set[int] = set()
    for layer in decoder.layers[: int(decoder.config.num_hidden_layers)]:
        layer_indices = tuple(
            indices[id(parameter)]
            for parameter in layer.parameters()
            if parameter.requires_grad
        )
        observed.update(layer_indices)
        per_layer.append(layer_indices)
    if observed != set(range(len(trainable))):
        raise CapabilityError(
            "layer-major execution requires every trainable parameter to belong "
            "to exactly one decoder layer"
        )
    return trainable, tuple(per_layer)


def _require_deterministic_decoder(decoder: torch.nn.Module) -> None:
    stochastic = {
        name: float(getattr(decoder.config, name, 0.0) or 0.0)
        for name in ("attention_dropout", "hidden_dropout", "dropout")
    }
    enabled = {name: value for name, value in stochastic.items() if value != 0.0}
    if enabled:
        raise CapabilityError(
            "layer-major execution currently requires zero decoder dropout; "
            f"found {enabled}"
        )
    if any(
        bool(getattr(layer, "gradient_checkpointing", False))
        for layer in decoder.layers[: int(decoder.config.num_hidden_layers)]
    ):
        raise CapabilityError(
            "layer-major execution owns recomputation and cannot be nested inside "
            "Transformers gradient checkpointing"
        )


class _LayerMajorDecoder(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        decoder: torch.nn.Module,
        input_ids: torch.Tensor,
        per_layer: tuple[tuple[int, ...], ...],
        cache_nf4: bool,
        activation_storage: str,
        *parameters: torch.nn.Parameter,
    ) -> torch.Tensor:
        device_type = input_ids.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(device_type)
        ctx.cache_nf4 = cache_nf4
        microbatches = int(input_ids.shape[0])
        hidden = [decoder.embed_tokens(input_ids[index]) for index in range(microbatches)]
        static = prepare_static_decoder_inputs_from_hidden(decoder, hidden[0])
        staged_inputs: list[tuple[torch.Tensor, ...]] = []
        for layer_index in range(int(decoder.config.num_hidden_layers)):
            staged_inputs.append(
                tuple(_stage_activation(value, activation_storage) for value in hidden)
            )
            if cache_nf4:
                from molt_stream.methods.nf4_layer_cache import dense_nf4_layer_cache

                with dense_nf4_layer_cache(decoder.layers[layer_index]):
                    hidden = [
                        run_static_decoder_layer(decoder, layer_index, value, static)
                        for value in hidden
                    ]
            else:
                hidden = [
                    run_static_decoder_layer(decoder, layer_index, value, static)
                    for value in hidden
                ]
        final_inputs = tuple(
            _stage_activation(value, activation_storage) for value in hidden
        )
        result = torch.stack([decoder.norm(value) for value in hidden])
        ctx.decoder = decoder
        ctx.static = static
        ctx.staged_inputs = tuple(staged_inputs)
        ctx.final_inputs = final_inputs
        ctx.per_layer = per_layer
        ctx.parameters = parameters
        ctx.device = input_ids.device
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor):
        decoder = ctx.decoder
        parameter_gradients: list[torch.Tensor | None] = [
            None for _ in ctx.parameters
        ]
        hidden_gradients: list[torch.Tensor] = []
        with torch.enable_grad(), torch.autocast(
            ctx.device.type,
            dtype=ctx.autocast_dtype,
            enabled=ctx.autocast_enabled,
        ):
            for microbatch, staged in enumerate(ctx.final_inputs):
                hidden = _restore_activation(staged, ctx.device)
                normalized = decoder.norm(hidden)
                hidden_gradients.append(
                    torch.autograd.grad(
                        normalized,
                        hidden,
                        grad_outputs=grad_output[microbatch],
                    )[0]
                )

            for layer_index in range(int(decoder.config.num_hidden_layers) - 1, -1, -1):
                layer_parameter_indices = ctx.per_layer[layer_index]
                layer_parameters = tuple(
                    ctx.parameters[index] for index in layer_parameter_indices
                )
                next_hidden_gradients: list[torch.Tensor] = []
                cache_context = nullcontext()
                if ctx.cache_nf4:
                    from molt_stream.methods.nf4_layer_cache import (
                        dense_nf4_layer_cache,
                    )

                    cache_context = dense_nf4_layer_cache(
                        decoder.layers[layer_index]
                    )
                with cache_context:
                    layer_inputs = tuple(
                        _restore_activation(staged, ctx.device)
                        for staged in ctx.staged_inputs[layer_index]
                    )
                    layer_outputs = tuple(
                        run_static_decoder_layer(
                            decoder, layer_index, hidden, ctx.static
                        )
                        for hidden in layer_inputs
                    )
                    gradients = torch.autograd.grad(
                        layer_outputs,
                        (*layer_inputs, *layer_parameters),
                        grad_outputs=tuple(hidden_gradients),
                        allow_unused=True,
                    )
                    next_hidden_gradients.extend(
                        gradients[: len(layer_inputs)]
                    )
                    for parameter_index, gradient in zip(
                        layer_parameter_indices,
                        gradients[len(layer_inputs) :],
                    ):
                        if gradient is not None:
                            parameter_gradients[parameter_index] = gradient
                hidden_gradients = next_hidden_gradients
        return (None, None, None, None, None, *parameter_gradients)


def layer_major_decoder_hidden(
    decoder: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    dense_nf4_cache: bool = False,
    activation_storage: str = "cpu",
) -> torch.Tensor:
    """Experimental exact all-layer traversal over ``[G, B, S]`` token IDs."""

    if input_ids.ndim != 3 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be an int64 tensor shaped [G, B, S]")
    if activation_storage not in {"cpu", "cuda"}:
        raise ValueError("activation_storage must be cpu or cuda")
    if activation_storage == "cuda" and input_ids.device.type != "cuda":
        raise ValueError("CUDA activation storage requires CUDA input IDs")
    _require_deterministic_decoder(decoder)
    parameters, per_layer = _parameter_layout(decoder)
    return _LayerMajorDecoder.apply(
        decoder,
        input_ids,
        per_layer,
        dense_nf4_cache,
        activation_storage,
        *parameters,
    )
