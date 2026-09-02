import torch
import pytest

from molt_stream.streaming.engine import StreamedLayer, StreamedLayerEngine, StreamedLoRALinear
from molt_stream.streaming.nf4 import NF4Tensor
from molt_stream.core.specs import DataSpec, ModelSpec, StreamSpec, TrainingMode, TrainingSpec


def test_nf4_has_expected_compact_storage_and_bounded_error():
    torch.manual_seed(1)
    value = torch.randn(64, 64)
    packed = NF4Tensor.quantize(value, block_size=64)
    restored = packed.dequantize()
    assert packed.storage_bytes == value.numel() // 2 + value.numel() // 64 * 4
    assert torch.mean((restored - value) ** 2).sqrt() < 0.15


def test_streamed_lora_backward_updates_only_adapters_on_cpu():
    torch.manual_seed(2)
    source = NF4Tensor.quantize(torch.randn(16, 16), block_size=16)
    linear = StreamedLoRALinear(source, rank=4, alpha=8, device=torch.device("cpu"))
    engine = StreamedLayerEngine([StreamedLayer(linear)], StreamSpec(device="cpu", compute_dtype="float32", quant_block_size=16))
    value = torch.randn(3, 16, requires_grad=True)
    engine(value).square().mean().backward()
    assert value.grad is not None
    assert linear.lora_a.grad is not None
    assert linear.lora_b.grad is not None
    assert source.packed.grad is None


def test_stream_spec_rejects_invalid_bundle_size():
    with pytest.raises(ValueError, match="bundle_size"):
        StreamSpec(device="cpu", bundle_size=0).validate()


def test_training_spec_rejects_implicit_compiler_fallback(tmp_path):
    data = tmp_path / "tokens.bin"
    data.write_bytes(bytes(range(64)))
    spec = TrainingSpec(
        mode=TrainingMode.PRETRAIN,
        data=DataSpec(str(data), context_length=8, storage_dtype="uint8"),
        model=ModelSpec(vocab_size=256, context_length=8, layers=1, width=8, heads=1, hidden_width=16),
        stream=StreamSpec(device="cpu"),
        execution_backend="unknown",
    )
    with pytest.raises(ValueError, match="execution_backend"):
        spec.validate()


def test_cpu_stream_report_exposes_bundle_policy():
    source = NF4Tensor.quantize(torch.randn(16, 16), block_size=16)
    linear = StreamedLoRALinear(source, rank=4, alpha=8, device=torch.device("cpu"))
    engine = StreamedLayerEngine(
        [StreamedLayer(linear)],
        StreamSpec(device="cpu", compute_dtype="float32", bundle_size=4),
    )
    engine(torch.randn(2, 16))
    assert engine.last_forward["bundle_size"] == 4
    assert engine.last_forward["stream_waits"] == 0


def test_streamed_linear_matches_dequantized_reference():
    torch.manual_seed(3)
    source = NF4Tensor.quantize(torch.randn(8, 8), block_size=16)
    linear = StreamedLoRALinear(source, rank=2, alpha=4, device=torch.device("cpu"))
    value = torch.randn(2, 8)
    staged = source.dequantize()
    expected = torch.nn.functional.linear(value, staged) + linear.scaling * torch.nn.functional.linear(
        torch.nn.functional.linear(value, linear.lora_a), linear.lora_b
    )
    assert torch.equal(linear(value, staged), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_stream_matches_synchronous_output_and_adapter_gradients():
    torch.manual_seed(17)
    sources = [NF4Tensor.quantize(torch.randn(32, 32), block_size=16, pin_memory=True) for _ in range(4)]

    def build(double_buffer: bool) -> StreamedLayerEngine:
        modules = [StreamedLoRALinear(source, rank=4, alpha=8, device=torch.device("cuda")) for source in sources]
        return StreamedLayerEngine(
            [StreamedLayer(module) for module in modules],
            StreamSpec(device="cuda", compute_dtype="float32", quant_block_size=16, double_buffer=double_buffer, bundle_size=4),
        )

    synchronous = build(False)
    grouped = build(True)
    grouped.load_state_dict(synchronous.state_dict())
    left = torch.randn(2, 5, 32, device="cuda", requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    output_left = synchronous(left)
    output_right = grouped(right)
    assert torch.allclose(output_left, output_right, rtol=0, atol=0)
    output_left.square().mean().backward()
    output_right.square().mean().backward()
    assert torch.allclose(left.grad, right.grad, rtol=1e-5, atol=1e-6)
    for expected, actual in zip(synchronous.parameters(), grouped.parameters(), strict=True):
        assert torch.allclose(expected.grad, actual.grad, rtol=1e-5, atol=1e-6)
    assert grouped.last_forward["stream_waits"] == 1
