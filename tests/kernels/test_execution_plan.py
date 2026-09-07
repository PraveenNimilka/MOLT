from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from molt_stream.kernels.execution_plan import (
    build_qlora_execution_plan,
    compatible_projection_names,
    projection_geometries,
    resolve_decoder_architecture,
)
from molt_stream.core.specs import DataSpec, TrainingMode, TrainingSpec


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 16, bias=False)
        self.k_proj = torch.nn.Linear(16, 8, bias=False)
        self.v_proj = torch.nn.Linear(16, 8, bias=False)
        self.o_proj = torch.nn.Linear(16, 16, bias=False)
        self.gate_proj = torch.nn.Linear(16, 32, bias=False)
        self.up_proj = torch.nn.Linear(16, 32, bias=False)
        self.down_proj = torch.nn.Linear(32, 16, bias=False)


class _Model(torch.nn.Module):
    def __init__(self, model_type: str) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.layers = torch.nn.ModuleList([_Block(), _Block()])


@pytest.mark.parametrize(
    ("model_type", "family"),
    [("qwen2", "qwen"), ("llama", "llama"), ("gemma2", "gemma")],
)
def test_resolves_supported_decoder_families(model_type: str, family: str) -> None:
    assert resolve_decoder_architecture(SimpleNamespace(model_type=model_type)).family == family


def test_unknown_architecture_fails_closed() -> None:
    with pytest.raises(Exception, match="do not support"):
        resolve_decoder_architecture(SimpleNamespace(model_type="bert"))


def test_projection_inventory_groups_shapes_and_counts() -> None:
    geometries = projection_geometries(_Model("qwen2"))
    assert any(
        item.role == "q_proj"
        and item.input_features == 16
        and item.output_features == 16
        and item.count == 2
        for item in geometries
    )
    assert sum(item.count for item in geometries) == 14


def test_projection_selection_returns_exact_names() -> None:
    model = _Model("gemma")
    assert compatible_projection_names(model, ("q_proj",)) == (
        "layers.0.q_proj",
        "layers.1.q_proj",
    )
    with pytest.raises(Exception, match="Unsupported gemma projection"):
        compatible_projection_names(model, ("made_up_proj",))


def test_execution_plan_fingerprint_is_stable_and_sensitive_to_kernel_choice() -> None:
    model = _Model("llama")
    automatic = build_qlora_execution_plan(
        model,
        attention_backend="sdpa",
        sdpa_kernel="auto",
        checkpoint_stride=1,
    )
    repeated = build_qlora_execution_plan(
        model,
        attention_backend="sdpa",
        sdpa_kernel="auto",
        checkpoint_stride=1,
    )
    efficient = build_qlora_execution_plan(
        model,
        attention_backend="sdpa",
        sdpa_kernel="efficient",
        checkpoint_stride=1,
    )

    assert automatic.fingerprint == repeated.fingerprint
    assert automatic.fingerprint != efficient.fingerprint
    assert automatic.to_dict()["architecture_family"] == "llama"
    assert automatic.to_dict()["accumulation_order"] == "microbatch-major"
    assert automatic.to_dict()["static_cuda_graph"] is False
    assert automatic.to_dict()["joint_cuda_graph"] is False


def test_execution_plan_fingerprint_covers_accumulation_topology() -> None:
    model = _Model("gemma2")
    microbatch_major = build_qlora_execution_plan(
        model,
        attention_backend="sdpa",
        sdpa_kernel="auto",
        checkpoint_stride=1,
        accumulation_steps=4,
    )
    layer_major = build_qlora_execution_plan(
        model,
        attention_backend="sdpa",
        sdpa_kernel="auto",
        checkpoint_stride=1,
        accumulation_steps=4,
        accumulation_order="layer-major",
    )
    assert microbatch_major.fingerprint != layer_major.fingerprint
    captured = build_qlora_execution_plan(
        model,
        attention_backend="sdpa",
        sdpa_kernel="auto",
        checkpoint_stride=1,
        accumulation_steps=4,
        static_cuda_graph=True,
    )
    assert microbatch_major.fingerprint != captured.fingerprint
    assert len(captured.compatible_fingerprints) == 2
    joint = build_qlora_execution_plan(
        model,
        attention_backend="sdpa",
        sdpa_kernel="auto",
        checkpoint_stride=1,
        accumulation_steps=4,
        static_cuda_graph=True,
        joint_cuda_graph=True,
    )
    assert joint.fingerprint != captured.fingerprint
    assert joint.compatible_fingerprints == frozenset({joint.fingerprint})


def test_non_graph_plan_accepts_verified_schema_one_and_two_fingerprints() -> None:
    plan = build_qlora_execution_plan(
        _Model("qwen2"),
        attention_backend="sdpa",
        sdpa_kernel="auto",
        checkpoint_stride=1,
        accumulation_steps=4,
    )

    def fingerprint(value):
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    schema_three = plan.to_dict()
    schema_three["schema_version"] = 3
    schema_three.pop("joint_cuda_graph")
    schema_two = dict(schema_three)
    schema_two["schema_version"] = 2
    schema_two.pop("static_cuda_graph")
    schema_one = dict(schema_two)
    schema_one["schema_version"] = 1
    schema_one.pop("accumulation_steps")
    schema_one.pop("accumulation_order")
    assert plan.compatible_fingerprints == frozenset(
        {
            plan.fingerprint,
            fingerprint(schema_three),
            fingerprint(schema_two),
            fingerprint(schema_one),
        }
    )


def test_native_nf4_roles_are_strictly_validated(tmp_path) -> None:
    data = tmp_path / "train.bin"
    data.write_bytes(b"\0" * 16)
    valid = TrainingSpec(
        mode=TrainingMode.QLORA,
        data=DataSpec(str(data), context_length=4),
        base_model="model",
        qlora_autocast=True,
        qlora_native_nf4_roles="k_proj,v_proj",
    )
    valid.validate()
    with pytest.raises(ValueError, match="unknown native NF4"):
        TrainingSpec(
            mode=TrainingMode.QLORA,
            data=DataSpec(str(data), context_length=4),
            base_model="model",
            qlora_autocast=True,
            qlora_native_nf4_roles="k_proj,unknown",
        ).validate()
