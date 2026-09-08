from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_beginner_dataset_is_pinned_and_well_formed() -> None:
    path = ROOT / "examples" / "beginner" / "molt-demo.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "eba10123241b2bf33d674bd5fdaa4d81b46e7adb773004c7375aa4ab2e7d6ef4"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 30
    assert all(set(row) == {"prompt", "completion"} for row in rows)


def test_beginner_configurator_emits_registered_geometry(
    tmp_path: Path, monkeypatch,
) -> None:
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"stream": {}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["configure.py", str(path)])
    module = runpy.run_path(
        str(ROOT / "examples" / "beginner" / "configure.py"),
        run_name="launch_material_test",
    )
    assert module["main"]() == 0
    value = json.loads(path.read_text(encoding="utf-8"))
    assert (value["batch_size"], value["gradient_accumulation"]) == (1, 1)
    assert (value["max_steps"], value["seed"]) == (20, 1337)
    assert value["stream"]["cuda_graphs"] is True
    assert value["qlora_joint_cuda_graph"] is True


def test_demo_record_matches_public_files() -> None:
    demo = ROOT / "docs" / "demo"
    record = json.loads((demo / "launch-demo.json").read_text(encoding="utf-8"))
    video = demo / "molt-launch-demo-75s.mp4"
    assert record["status"] == "completed"
    assert record["recording"]["duration_seconds"] == 75.0
    assert hashlib.sha256(video.read_bytes()).hexdigest() == record["recording"]["sha256"]
    assert record["adapter"]["reload_response"] == "THERMAL-READY"


def test_launch_pages_use_consistent_license_and_version() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    announcement = (ROOT / "RELEASE_ANNOUNCEMENT.md").read_text(encoding="utf-8")
    quickstart = (ROOT / "docs" / "BEGINNER_QUICKSTART.md").read_text(encoding="utf-8")
    for text in (readme, announcement, quickstart):
        assert "source-available" in text.lower()
        assert "PolyForm Shield 1.0.0" in text
    assert "0.11.0a7" in readme
    assert "moltengine==0.11.0a7" in quickstart


def test_public_reproduction_templates_cover_all_families() -> None:
    templates = ROOT / "benchmarks" / "templates"
    values = {
        path.stem.split("-")[0]: json.loads(path.read_text(encoding="utf-8"))
        for path in templates.glob("*-diagnostic.json")
    }
    assert set(values) == {"qwen", "llama", "gemma"}
    assert {value["data"]["context_length"] for value in values.values()} == {
        512, 1024
    }
    for value in values.values():
        assert value["stream"]["cuda_graphs"] is True
        assert value["stream"]["lora_target_modules"] == "all-linear"
        assert value["batch_size"] == value["gradient_accumulation"] == 1
