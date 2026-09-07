from __future__ import annotations

import pytest
import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.training.layer_major import layer_major_decoder_hidden

pytest.importorskip("transformers")


def _model(family: str = "qwen", dropout: float = 0.0):
    from transformers import Gemma2Config, Gemma2Model
    from transformers import LlamaConfig, LlamaModel
    from transformers import Qwen2Config, Qwen2Model

    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "attention_dropout": dropout,
    }
    if family == "qwen":
        model = Qwen2Model(Qwen2Config(**values))
    elif family == "llama":
        model = LlamaModel(LlamaConfig(**values))
    else:
        model = Gemma2Model(
            Gemma2Config(**values, head_dim=8, sliding_window=32)
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.layers[0].self_attn.q_proj.weight.requires_grad_(True)
    model.layers[1].mlp.down_proj.weight.requires_grad_(True)
    model.train()
    return model


@pytest.mark.parametrize("family", ["qwen", "llama", "gemma"])
def test_layer_major_matches_microbatch_major_outputs_and_gradients(
    family: str,
) -> None:
    model = _model(family)
    tokens = torch.tensor(
        [
            [[2, 3, 4, 5]],
            [[6, 7, 8, 9]],
            [[10, 11, 12, 13]],
        ]
    )
    reference_outputs = []
    for microbatch in tokens:
        output = model(input_ids=microbatch, use_cache=False).last_hidden_state
        reference_outputs.append(output.detach())
        output.square().mean().div(tokens.shape[0]).backward()
    expected = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    model.zero_grad(set_to_none=True)

    candidate = layer_major_decoder_hidden(model, tokens)
    torch.testing.assert_close(
        candidate, torch.stack(reference_outputs), rtol=0.0, atol=0.0
    )
    candidate.square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                parameter.grad, expected[name], rtol=1e-6, atol=1e-7
            )


def test_layer_major_rejects_trainable_parameters_outside_layers() -> None:
    model = _model()
    model.norm.weight.requires_grad_(True)
    with pytest.raises(CapabilityError, match="every trainable parameter"):
        layer_major_decoder_hidden(model, torch.tensor([[[2, 3, 4, 5]]]))


def test_layer_major_rejects_dropout_and_nested_checkpointing() -> None:
    with pytest.raises(CapabilityError, match="zero decoder dropout"):
        layer_major_decoder_hidden(
            _model(dropout=0.1), torch.tensor([[[2, 3, 4, 5]]])
        )

    model = _model()
    model.layers[0].gradient_checkpointing = True
    with pytest.raises(CapabilityError, match="owns recomputation"):
        layer_major_decoder_hidden(model, torch.tensor([[[2, 3, 4, 5]]]))


def test_layer_major_rejects_invalid_activation_storage() -> None:
    with pytest.raises(ValueError, match="must be cpu or cuda"):
        layer_major_decoder_hidden(
            _model(),
            torch.tensor([[[2, 3, 4, 5]]]),
            activation_storage="disk",
        )
