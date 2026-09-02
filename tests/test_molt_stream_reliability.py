from __future__ import annotations

from pathlib import Path

import pytest
import torch

from molt_stream.core.specs import TrainingSpec
from molt_stream.experiments.store import AtomicCheckpointStore
from molt_stream.training.engine import train


def _config(token_file: Path, artifacts: Path) -> dict[str, object]:
    return {
        "mode": "pretrain",
        "data": {
            "path": str(token_file),
            "context_length": 4,
            "storage_dtype": "int32",
        },
        "model": {
            "vocab_size": 32,
            "context_length": 4,
            "layers": 1,
            "width": 8,
            "heads": 1,
            "hidden_width": 16,
        },
        "stream": {"device": "cpu", "compute_dtype": "float32"},
        "max_steps": 1,
        "artifacts_dir": str(artifacts),
    }


def test_training_spec_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens.bin"
    torch.arange(32, dtype=torch.int32).numpy().tofile(tokens)
    value = _config(tokens, tmp_path / "runs")
    value["bach_size"] = 8

    with pytest.raises(ValueError, match="unknown training configuration field.*bach_size"):
        TrainingSpec.from_dict(value)


def test_activation_checkpointing_is_explicit_and_defaults_safe(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens.bin"
    torch.arange(32, dtype=torch.int32).numpy().tofile(tokens)
    value = _config(tokens, tmp_path / "runs")

    assert TrainingSpec.from_dict(value).activation_checkpointing is True
    value["activation_checkpointing"] = False
    assert TrainingSpec.from_dict(value).activation_checkpointing is False


def test_evaluation_interval_is_explicit_and_validated(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens.bin"
    torch.arange(32, dtype=torch.int32).numpy().tofile(tokens)
    value = _config(tokens, tmp_path / "runs")
    assert TrainingSpec.from_dict(value).evaluation_interval is None
    value["evaluation_interval"] = 5
    spec = TrainingSpec.from_dict(value)
    spec.validate()
    assert spec.evaluation_interval == 5
    value["evaluation_interval"] = 0
    with pytest.raises(ValueError, match="evaluation_interval"):
        TrainingSpec.from_dict(value).validate()


def test_config_paths_expand_environment_variables_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = tmp_path / "tokens.bin"
    torch.arange(32, dtype=torch.int32).numpy().tofile(tokens)
    monkeypatch.setenv("MOLT_TEST_DATA_ROOT", str(tmp_path))
    value = _config(tokens, tmp_path / "runs")
    value["data"]["path"] = "${MOLT_TEST_DATA_ROOT}/tokens.bin"

    assert Path(TrainingSpec.from_dict(value).data.path) == tokens

    value["data"]["path"] = "${MOLT_MISSING_DATA_ROOT}/tokens.bin"
    with pytest.raises(ValueError, match="MOLT_MISSING_DATA_ROOT"):
        TrainingSpec.from_dict(value)


def test_completed_run_cannot_be_resumed(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens.bin"
    torch.arange(32, dtype=torch.int32).numpy().tofile(tokens)
    spec = TrainingSpec.from_dict(_config(tokens, tmp_path / "runs"))
    spec.validate()
    run = train(spec)

    assert AtomicCheckpointStore(run).load()["termination_reason"] == "completed"
    with pytest.raises(ValueError, match="already complete"):
        train(spec, resume=run)
