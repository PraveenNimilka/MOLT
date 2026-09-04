"""Self-contained test inputs; never require local prepared datasets."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def smoke_config(tmp_path: Path) -> Path:
    """Copy the shipped smoke spec with temporary, deterministic token files."""
    template = Path(__file__).resolve().parents[1] / "configs/molt-stream-smoke.json"
    spec = json.loads(template.read_text(encoding="utf-8"))
    for key, name in (("path", "train.bin"), ("validation_path", "validation.bin")):
        tokens = tmp_path / name
        tokens.write_bytes(bytes(range(256)) * 8)
        spec["data"][key] = str(tokens)
    spec["artifacts_dir"] = str(tmp_path / "runs")
    config = tmp_path / "smoke.json"
    config.write_text(json.dumps(spec), encoding="utf-8")
    return config
