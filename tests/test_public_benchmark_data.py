from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs" / "benchmarks" / "0.11.0a3-diagnostic.json"
SVG_PATH = ROOT / "docs" / "assets" / "benchmark-diagnostic-0.11.0a3.svg"
RENDERER_PATH = ROOT / "tools" / "render_benchmark_summary.py"


def test_public_benchmark_record_and_figure_are_consistent() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    assert data["status"] == "diagnostic_only"
    assert data["pairs_per_family"] == 6
    assert [result["family"] for result in data["results"]] == [
        "Qwen",
        "Llama-family",
        "Gemma",
    ]

    for result in data["results"]:
        assert result["official_gate_passed"] is False
        assert len(result["contract_sha256"]) == 64
        assert len(result["local_results_sha256"]) == 64
        for key in ("time_reduction_percent", "energy_reduction_percent"):
            metric = result[key]
            assert metric["ci95_low"] < metric["mean"] < metric["ci95_high"]

    spec = importlib.util.spec_from_file_location("benchmark_renderer", RENDERER_PATH)
    assert spec is not None and spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    assert SVG_PATH.read_text(encoding="utf-8") == renderer.render(data)
