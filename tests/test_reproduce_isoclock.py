from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location(
    "molt_reproduce_isoclock", BENCHMARKS / "reproduce_isoclock.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preflight_only_never_applies_clock(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(MODULE, "temporary_graphics_clock", lambda *_: (_ for _ in ()).throw(
        AssertionError("preflight changed clocks")
    ))
    result = MODULE.main(["--preflight-only", "--config", str(tmp_path / "missing.json")])
    assert result == 2
    assert "Blocked:" in capsys.readouterr().err


def test_preflight_accepts_matched_geometry(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        '{"mode":"qlora","batch_size":1,"gradient_accumulation":4,'
        '"data":{"context_length":512},"stream":{"lora_rank":8,'
        '"lora_target_modules":"all-linear"}}',
        encoding="utf-8",
    )
    paths = []
    for name in ("model", "train.bin", "validation.bin", "unsloth"):
        path = tmp_path / name
        path.mkdir() if "." not in name else path.write_bytes(b"\0")
        paths.append(path)
    monkeypatch.setattr(MODULE.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(MODULE.torch.cuda, "mem_get_info", lambda: (7 * MODULE.GIB, 8 * MODULE.GIB))
    monkeypatch.setattr(MODULE.torch.cuda, "get_device_name", lambda _index: "test GPU")
    args = MODULE._parser().parse_args([
        "--preflight-only", "--config", str(config), "--model", str(paths[0]),
        "--train-data", str(paths[1]), "--validation-data", str(paths[2]),
        "--unsloth-site", str(paths[3]),
    ])
    measured, errors = MODULE.preflight(args)
    assert errors == []
    assert measured["telemetry_interval_ms"] == 100


def test_comparison_table_uses_measured_values():
    base = {
        "state": "completed", "end_to_end_tokens_per_second": 1000.0,
        "compute_tokens_per_second": 1200.0, "seconds": 10.0,
        "board_energy_joules": 400.0, "joules_per_token": 0.04,
        "thermal_pause_seconds": 0.0, "peak_temperature_c": 67.0,
        "peak_allocated_gib": 2.9, "final_nll": 1.5,
    }
    table = MODULE._table({"molt": base, "unsloth": {**base, "state": "thermal_abort"}})
    assert "End-to-end tok/s" in table
    assert "thermal_abort" in table
    assert "1,000.00" in table
