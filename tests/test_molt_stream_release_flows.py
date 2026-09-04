import importlib
import json
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from molt_stream.cli import main
from molt_stream.core.discovery import find_runs, init_workspace
from molt_stream.core.specs import load_spec
from molt_stream.experiments.store import AtomicCheckpointStore


@pytest.mark.parametrize("command,module,function,extra,result", [
    ("benchmark", "throughput", "training_throughput_benchmark", [],
     {"sustained_tokens_per_second": 1, "total_seconds_including_setup_and_warmup": 1}),
    ("fusion-benchmark", "throughput", "fusion_memory_benchmark", [], {}),
    ("curriculum-benchmark", "curriculum", "run_curriculum_experiment", ["--seeds", "1"], {}),
    ("loss-partition-benchmark", "loss_partition", "benchmark_exact_loss_partitioning", [], {}),
])
def test_research_dispatch_uses_real_signature(monkeypatch, smoke_config, command, module, function, extra, result):
    module = importlib.import_module("molt_stream.training." + module)
    call = create_autospec(getattr(module, function), return_value=result)
    monkeypatch.setattr(module, function, call)
    assert main(["--json", command, "--config", str(smoke_config), *extra]) == 0
    call.assert_called_once()


def test_stream_alias_preserves_signature(monkeypatch):
    from molt_stream.training import stream_benchmark
    call = create_autospec(stream_benchmark.stream_tune, return_value={})
    monkeypatch.setattr(stream_benchmark, "stream_tune", call)
    assert main(["--json", "research", "stream-tune", "--synchronous"]) == 0
    assert call.call_args.kwargs["double_buffer"] is False


def test_power_query_does_not_apply(monkeypatch):
    from molt_stream.measurement import telemetry
    call = create_autospec(telemetry.manage_power_limit, return_value={})
    monkeypatch.setattr(telemetry, "manage_power_limit", call)
    assert main(["--json", "power-limit"]) == 0
    assert call.call_args.kwargs["apply"] is False


def test_config_overrides_are_used(smoke_config, capsys):
    assert main(["--json", "train", "--config", str(smoke_config), "--dry-run",
                 "--batch-size", "7", "--context-length", "16", "--learning-rate", "0.002"]) == 0
    spec = json.loads(capsys.readouterr().out)["spec"]
    assert spec["batch_size"] == 7
    assert spec["data"]["context_length"] == spec["model"]["context_length"] == 16
    assert spec["learning_rate"] == 0.002


def test_workspace_template_and_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = init_workspace()
    spec = load_spec(paths["configs"] / "default_training.json")
    assert Path(spec.data.path).is_absolute()
    assert (paths["root"] / "molt-workspace.json").is_file()


def test_discovery_really_verifies_checksum(tmp_path):
    run = tmp_path / "run"
    store = AtomicCheckpointStore(run)
    store.save({"step": 1})
    (run / "spec.resolved.json").write_text("{}")
    assert find_runs([tmp_path])[0]["integrity"] == "Verified (SHA-256)"
    (run / "checkpoint.pt").write_bytes(b"corrupt")
    assert find_runs([tmp_path])[0]["integrity"] == "Invalid checkpoint"


def test_config_loader_is_shared():
    from molt_stream.training import engine
    assert engine.load_spec is load_spec


def test_qlora_context_independent_of_scratch_defaults(smoke_config):
    from dataclasses import replace
    from molt_stream.core.specs import TrainingMode
    spec = load_spec(smoke_config)
    spec = replace(spec, mode=TrainingMode.QLORA, base_model="local/model",
                   model=replace(spec.model, context_length=128))
    spec.validate()


def test_compare_paired_maps_pairs_and_writes_output(tmp_path, monkeypatch):
    from molt_stream.measurement import comparison
    compare = create_autospec(comparison.compare_runs, return_value={"seed": 1})
    aggregate = create_autospec(comparison.aggregate_comparisons, return_value={"ok": True})
    monkeypatch.setattr(comparison, "compare_runs", compare)
    monkeypatch.setattr(comparison, "aggregate_comparisons", aggregate)
    out = tmp_path / "pairs.json"
    assert main(["--json", "compare-paired", "--pair", "a", "b", "--pair", "c", "d",
                 "--output", str(out)]) == 0
    assert compare.call_count == 2
    assert aggregate.call_args.args[0] == [{"seed": 1}, {"seed": 1}]
    assert json.loads(out.read_text())["ok"]


def test_compare_export(tmp_path, monkeypatch):
    monkeypatch.setattr("molt_stream.measurement.comparison.compare_runs", lambda *a, **kw: {"ok": True})
    out = tmp_path / "comparison.json"
    assert main(["--json", "compare", "a", "b", "--output", str(out)]) == 0
    assert out.is_file()


def test_qlora_benchmark_uses_spec(monkeypatch, smoke_config):
    from molt_stream.training import qlora
    call = create_autospec(qlora.benchmark_qlora_adapter, return_value={})
    monkeypatch.setattr(qlora, "benchmark_qlora_adapter", call)
    config = json.loads(smoke_config.read_text())
    config.update(mode="qlora", base_model="local/model")
    (smoke_config.parent / "spec.resolved.json").write_text(json.dumps(config))
    assert main(["--json", "qlora-benchmark", "--run", str(smoke_config.parent)]) == 0
    assert call.call_args.args[0].batch_size == load_spec(smoke_config).batch_size
