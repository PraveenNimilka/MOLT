from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import torch

from molt_stream.core.errors import CapabilityError


@dataclass(frozen=True)
class DecoderArchitecture:
    """Static decoder contract used by MOLT-native kernel dispatch.

    The contract deliberately describes semantic projection roles instead of
    Python class names.  This keeps kernel selection independent of a specific
    Transformers release while making unsupported architectures fail closed.
    """

    family: str
    model_types: tuple[str, ...]
    attention_projections: tuple[str, ...]
    mlp_projections: tuple[str, ...]
    norm_suffixes: tuple[str, ...]
    gated_activation: bool
    rmsnorm_weight_offset: float

    @property
    def linear_projections(self) -> tuple[str, ...]:
        return self.attention_projections + self.mlp_projections


_ARCHITECTURES = (
    DecoderArchitecture(
        family="qwen",
        model_types=("qwen2", "qwen2_moe", "qwen3", "qwen3_moe"),
        attention_projections=("q_proj", "k_proj", "v_proj", "o_proj"),
        mlp_projections=("gate_proj", "up_proj", "down_proj"),
        norm_suffixes=("input_layernorm", "post_attention_layernorm"),
        gated_activation=True,
        rmsnorm_weight_offset=0.0,
    ),
    DecoderArchitecture(
        family="llama",
        model_types=("llama", "mistral", "mixtral"),
        attention_projections=("q_proj", "k_proj", "v_proj", "o_proj"),
        mlp_projections=("gate_proj", "up_proj", "down_proj"),
        norm_suffixes=("input_layernorm", "post_attention_layernorm"),
        gated_activation=True,
        rmsnorm_weight_offset=0.0,
    ),
    DecoderArchitecture(
        family="gemma",
        model_types=("gemma", "gemma2", "gemma3_text"),
        attention_projections=("q_proj", "k_proj", "v_proj", "o_proj"),
        mlp_projections=("gate_proj", "up_proj", "down_proj"),
        norm_suffixes=("input_layernorm", "post_attention_layernorm"),
        gated_activation=True,
        rmsnorm_weight_offset=1.0,
    ),
)


def resolve_decoder_architecture(model_or_config: object) -> DecoderArchitecture:
    """Return a supported decoder contract or reject native dispatch."""

    config = getattr(model_or_config, "config", model_or_config)
    model_type = str(getattr(config, "model_type", "")).lower()
    for architecture in _ARCHITECTURES:
        if model_type in architecture.model_types:
            return architecture
    supported = ", ".join(
        model_type for architecture in _ARCHITECTURES for model_type in architecture.model_types
    )
    raise CapabilityError(
        f"MOLT-native kernels do not support model_type={model_type or '<missing>'}; "
        f"supported model types: {supported}"
    )


@dataclass(frozen=True)
class ProjectionGeometry:
    role: str
    input_features: int
    output_features: int
    count: int


@dataclass(frozen=True)
class QLoRAExecutionPlan:
    """Immutable, serializable kernel topology for one loaded decoder."""

    schema_version: int
    architecture_family: str
    model_type: str
    attention_backend: str
    sdpa_kernel: str
    checkpoint_stride: int
    accumulation_steps: int
    accumulation_order: str
    static_cuda_graph: bool
    joint_cuda_graph: bool
    native_nf4_roles: tuple[str, ...]
    projections: tuple[ProjectionGeometry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "architecture_family": self.architecture_family,
            "model_type": self.model_type,
            "attention_backend": self.attention_backend,
            "sdpa_kernel": self.sdpa_kernel,
            "checkpoint_stride": self.checkpoint_stride,
            "accumulation_steps": self.accumulation_steps,
            "accumulation_order": self.accumulation_order,
            "static_cuda_graph": self.static_cuda_graph,
            "joint_cuda_graph": self.joint_cuda_graph,
            "native_nf4_roles": list(self.native_nf4_roles),
            "projections": [
                {
                    "role": item.role,
                    "input_features": item.input_features,
                    "output_features": item.output_features,
                    "count": item.count,
                }
                for item in self.projections
            ],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @property
    def compatible_fingerprints(self) -> frozenset[str]:
        """Current and safe legacy identities accepted during resume.

        Schema 1/2 predate static CUDA graphs, so they are accepted only when
        the current plan is also non-graph. Resume still loads the immutable
        saved training specification; this compatibility set only prevents a
        serialization-schema addition from invalidating existing runs.
        """

        values = {self.fingerprint}
        schema_three = self.to_dict()
        schema_three["schema_version"] = 3
        schema_three.pop("joint_cuda_graph")
        if self.static_cuda_graph:
            if not self.joint_cuda_graph:
                values.add(_fingerprint(schema_three))
            return frozenset(values)
        values.add(_fingerprint(schema_three))
        schema_two = dict(schema_three)
        schema_two["schema_version"] = 2
        schema_two.pop("static_cuda_graph")
        values.add(_fingerprint(schema_two))
        schema_one = dict(schema_two)
        schema_one["schema_version"] = 1
        schema_one.pop("accumulation_steps")
        schema_one.pop("accumulation_order")
        values.add(_fingerprint(schema_one))
        return frozenset(values)


def _fingerprint(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def projection_geometries(model: torch.nn.Module) -> tuple[ProjectionGeometry, ...]:
    """Inventory compatible decoder projection shapes without modifying a model."""

    architecture = resolve_decoder_architecture(model)
    roles = set(architecture.linear_projections)
    counts: dict[tuple[str, int, int], int] = {}
    for name, module in model.named_modules():
        role = name.rsplit(".", 1)[-1]
        if role not in roles:
            continue
        input_features = getattr(module, "in_features", None)
        output_features = getattr(module, "out_features", None)
        if not isinstance(input_features, int) or not isinstance(output_features, int):
            continue
        key = (role, input_features, output_features)
        counts[key] = counts.get(key, 0) + 1
    return tuple(
        ProjectionGeometry(role, input_features, output_features, count)
        for (role, input_features, output_features), count in sorted(counts.items())
    )


def compatible_projection_names(
    model: torch.nn.Module, enabled_roles: Iterable[str]
) -> tuple[str, ...]:
    """Resolve exact module names, refusing unknown semantic roles."""

    architecture = resolve_decoder_architecture(model)
    requested = tuple(dict.fromkeys(str(role) for role in enabled_roles))
    unknown = sorted(set(requested) - set(architecture.linear_projections))
    if unknown:
        raise CapabilityError(
            f"Unsupported {architecture.family} projection roles: {', '.join(unknown)}"
        )
    requested_set = set(requested)
    return tuple(
        name
        for name, _module in model.named_modules()
        if name.rsplit(".", 1)[-1] in requested_set
    )


def build_qlora_execution_plan(
    model: torch.nn.Module,
    *,
    attention_backend: str,
    sdpa_kernel: str,
    checkpoint_stride: int,
    accumulation_steps: int = 1,
    accumulation_order: str = "microbatch-major",
    static_cuda_graph: bool = False,
    joint_cuda_graph: bool = False,
    native_nf4_roles: Iterable[str] = (),
) -> QLoRAExecutionPlan:
    """Resolve the runtime topology once, before the first optimizer update."""

    architecture = resolve_decoder_architecture(model)
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    if accumulation_order not in {"microbatch-major", "layer-major"}:
        raise ValueError("unsupported accumulation execution order")
    if joint_cuda_graph and not static_cuda_graph:
        raise ValueError("joint CUDA graph requires static CUDA graph execution")
    roles = tuple(dict.fromkeys(str(role) for role in native_nf4_roles if role))
    unknown = sorted(set(roles) - set(architecture.linear_projections))
    if unknown:
        raise CapabilityError(
            f"Unsupported {architecture.family} native NF4 roles: {', '.join(unknown)}"
        )
    return QLoRAExecutionPlan(
        schema_version=4,
        architecture_family=architecture.family,
        model_type=str(getattr(model.config, "model_type", "")),
        attention_backend=attention_backend,
        sdpa_kernel=sdpa_kernel,
        checkpoint_stride=checkpoint_stride,
        accumulation_steps=accumulation_steps,
        accumulation_order=accumulation_order,
        static_cuda_graph=static_cuda_graph,
        joint_cuda_graph=joint_cuda_graph,
        native_nf4_roles=roles,
        projections=projection_geometries(model),
    )
