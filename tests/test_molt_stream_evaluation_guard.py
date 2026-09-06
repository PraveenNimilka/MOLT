import pytest
import torch

from molt_stream.training.evaluation_guard import EvaluationThermalStop, guarded_decoder_evaluation


def test_guard_preserves_output_and_removes_hooks_on_stop():
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
    x = torch.arange(8).unsqueeze(0)
    with torch.no_grad():
        expected = model(x).logits
        with guarded_decoder_evaluation(model, lambda: True):
            torch.testing.assert_close(model(x).logits, expected)
        with pytest.raises(EvaluationThermalStop):
            with guarded_decoder_evaluation(model, lambda: False):
                model(x)
        assert all(not layer._forward_pre_hooks for layer in model.model.layers)
        assert not model.model.norm._forward_pre_hooks
        torch.testing.assert_close(model(x).logits, expected)
