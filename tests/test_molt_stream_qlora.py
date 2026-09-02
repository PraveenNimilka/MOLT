import pytest
import torch
from torch import nn

from molt_stream.core.errors import CapabilityError
from molt_stream.core.specs import DataSpec, StreamSpec, TrainingMode, TrainingSpec
from molt_stream.cli import build_parser
from molt_stream.training.qlora import (
    _local_checkpoint_parameter_count,
    _resolve_lora_target_modules,
    _shifted_causal_loss,
    train_qlora,
)


class _NextTokenModel(nn.Module):
    def forward(self, *, input_ids, use_cache):
        assert use_cache is False
        targets = (input_ids + 1) % 8
        logits = torch.full((*input_ids.shape, 8), -20.0)
        logits.scatter_(-1, targets.unsqueeze(-1), 20.0)
        return type("Output", (), {"logits": logits})()


def test_train_cli_accepts_an_explicit_reproduction_seed():
    args = build_parser().parse_args([
        "train", "--config", "experiment.json", "--seed", "2027"
    ])
    assert args.seed == 2027


def test_qlora_loss_uses_mmap_targets_without_a_second_shift():
    inputs = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])

    assert _shifted_causal_loss(_NextTokenModel(), inputs, targets).item() < 1e-6


def test_local_checkpoint_parameter_count_uses_logical_safetensor_shapes(tmp_path):
    from safetensors.torch import save_file

    save_file({"first": torch.zeros(3, 5), "second": torch.zeros(7)}, tmp_path / "model.safetensors")

    assert _local_checkpoint_parameter_count(str(tmp_path)) == 22


def test_lora_target_modules_support_default_and_explicit_projection_sets():
    assert _resolve_lora_target_modules("all-linear") == "all-linear"
    assert _resolve_lora_target_modules("q_proj, k_proj,v_proj,o_proj") == [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ]
    StreamSpec(lora_target_modules="q_proj,k_proj,v_proj,o_proj").validate()


def test_lora_target_modules_reject_empty_or_ambiguous_sets():
    with pytest.raises(ValueError, match="at least one"):
        StreamSpec(lora_target_modules=" , ").validate()
    with pytest.raises(ValueError, match="cannot be combined"):
        StreamSpec(lora_target_modules="all-linear,q_proj").validate()


def test_lora_plus_ratio_is_opt_in_and_validated():
    StreamSpec(lora_plus_lr_ratio=None).validate()
    StreamSpec(lora_plus_lr_ratio=16.0).validate()
    with pytest.raises(ValueError, match=">= 1"):
        StreamSpec(lora_plus_lr_ratio=0.5).validate()


def test_qlora_refuses_missing_optional_runtime(tmp_path, monkeypatch):
    import importlib.util

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        "molt_stream.training.qlora.importlib.util.find_spec",
        lambda name: None if name == "peft" else find_spec(name),
    )
    tokens = tmp_path / "tokens.bin"
    tokens.write_bytes(bytes(range(64)))
    spec = TrainingSpec(
        mode=TrainingMode.QLORA,
        data=DataSpec(str(tokens), context_length=8, storage_dtype="uint8"),
        base_model="unavailable/model",
    )
    with pytest.raises(CapabilityError, match="missing optional packages: peft"):
        train_qlora(spec)
