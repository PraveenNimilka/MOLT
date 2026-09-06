import pytest
import torch

from molt_stream.training.truncated_backprop import (
    activate_upper_layer_training,
    truncate_before_decoder_layer,
    upper_layer_indices,
)


def test_upper_layer_indices_are_exact():
    assert upper_layer_indices(28, 4) == [24, 25, 26, 27]
    with pytest.raises(ValueError):
        upper_layer_indices(4, 5)


def test_qwen_boundary_blocks_lower_gradients_but_preserves_upper_gradients():
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2))
    truncate_before_decoder_layer(model, 2)
    model(torch.arange(8).unsqueeze(0), labels=torch.arange(8).unsqueeze(0)).loss.backward()
    assert all(parameter.grad is None for parameter in model.model.layers[0].parameters())
    assert all(parameter.grad is None for parameter in model.model.layers[1].parameters())
    assert any(parameter.grad is not None for parameter in model.model.layers[2].parameters())


def test_activation_freezes_only_lower_lora_parameters():
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(2, 1, bias=False)
            self.base = torch.nn.Linear(2, 2, bias=False)
        def forward(self, value):
            return self.base(value) + self.lora_A(value).expand(-1, 2)
    class FakeQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"model_type": "qwen2", "num_hidden_layers": 3})()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(), Layer(), Layer()])
    model = FakeQwen()
    frozen = activate_upper_layer_training(model, 1)
    assert frozen == 4
    assert not model.model.layers[0].lora_A.weight.requires_grad
    assert not model.model.layers[1].lora_A.weight.requires_grad
    assert model.model.layers[2].lora_A.weight.requires_grad
    assert all(layer.base.weight.requires_grad for layer in model.model.layers)
