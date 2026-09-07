from __future__ import annotations

from dataclasses import dataclass

import torch

from molt_stream.kernels.execution_plan import resolve_decoder_architecture


@dataclass(frozen=True)
class StaticDecoderInputs:
    """Values shared by every layer of one cache-free decoder pass."""

    position_ids: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    attention_masks: tuple[torch.Tensor | None, ...]


def prepare_static_decoder_inputs(
    decoder: torch.nn.Module,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, StaticDecoderInputs]:
    """Reproduce Transformers' cache-free decoder preamble explicitly."""

    hidden = decoder.embed_tokens(input_ids)
    return hidden, prepare_static_decoder_inputs_from_hidden(decoder, hidden)


def prepare_static_decoder_inputs_from_hidden(
    decoder: torch.nn.Module,
    hidden: torch.Tensor,
) -> StaticDecoderInputs:
    """Build cache-free positional and masking state for embedded inputs."""

    architecture = resolve_decoder_architecture(decoder)
    position_ids = torch.arange(
        hidden.shape[1], device=hidden.device, dtype=torch.long
    ).unsqueeze(0)
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    mask_kwargs = {
        "config": decoder.config,
        "inputs_embeds": hidden,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    masks: dict[str, torch.Tensor | None] = {
        "full_attention": create_causal_mask(**mask_kwargs)
    }
    layer_types = tuple(
        getattr(
            decoder.config,
            "layer_types",
            ["full_attention"] * int(decoder.config.num_hidden_layers),
        )
    )
    if "sliding_attention" in layer_types:
        masks["sliding_attention"] = create_sliding_window_causal_mask(
            **mask_kwargs
        )
    if architecture.family == "llama":
        position_embeddings = decoder.rotary_emb(hidden, position_ids=position_ids)
    else:
        position_embeddings = decoder.rotary_emb(hidden, position_ids)
    return StaticDecoderInputs(
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        attention_masks=tuple(masks[layer_type] for layer_type in layer_types),
    )


def run_static_decoder_layer(
    decoder: torch.nn.Module,
    layer_index: int,
    hidden: torch.Tensor,
    static: StaticDecoderInputs,
) -> torch.Tensor:
    """Execute one decoder layer with the same arguments as HF's model loop."""

    if layer_index < 0 or layer_index >= int(decoder.config.num_hidden_layers):
        raise IndexError("decoder layer index is out of range")
    return decoder.layers[layer_index](
        hidden,
        attention_mask=static.attention_masks[layer_index],
        position_embeddings=static.position_embeddings,
        position_ids=static.position_ids,
        past_key_values=None,
        use_cache=False,
    )


def static_decoder_forward(
    decoder: torch.nn.Module, input_ids: torch.Tensor
) -> torch.Tensor:
    """Reference explicit traversal used to qualify layer-major scheduling."""

    hidden, static = prepare_static_decoder_inputs(decoder, input_ids)
    for layer_index in range(int(decoder.config.num_hidden_layers)):
        hidden = run_static_decoder_layer(decoder, layer_index, hidden, static)
    return decoder.norm(hidden)
