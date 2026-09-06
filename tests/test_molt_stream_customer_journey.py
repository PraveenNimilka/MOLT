import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from molt_stream.core.specs import DataSpec, ModelSpec, StreamSpec, TrainingMode, TrainingSpec
from molt_stream.experiments.store import AtomicCheckpointStore
from molt_stream.training import engine


def tiny_spec(root):
    tokens = root / "tokens.bin"
    tokens.write_bytes(bytes(range(32)) * 32)
    return TrainingSpec(mode=TrainingMode.PRETRAIN,
        data=DataSpec(str(tokens), context_length=8, validation_path=str(tokens), storage_dtype="uint8"),
        model=ModelSpec(vocab_size=32, context_length=8, layers=1, width=16, heads=2, hidden_width=32),
        stream=StreamSpec(device="cpu", compute_dtype="float32"), batch_size=2, max_steps=2,
        artifacts_dir=str(root / "runs"))


def test_simulated_boundary_resume_matches_uninterrupted(tmp_path, monkeypatch):
    spec = tiny_spec(tmp_path)
    monkeypatch.setattr(engine, "latest_temperature_c", lambda _: 85.0)
    run = engine.train(spec)
    assert AtomicCheckpointStore(run).load()["step"] == 1
    monkeypatch.setattr(engine, "latest_temperature_c", lambda _: None)
    engine.train(spec, resume=run)
    reference = engine.train(spec)
    resumed = AtomicCheckpointStore(run).load()
    expected = AtomicCheckpointStore(reference).load()
    for name, weight in expected["model"].items():
        assert torch.equal(weight, resumed["model"][name])
    assert torch.equal(expected["torch_rng"], resumed["torch_rng"])


def test_fresh_workspace_subprocess_journey(tmp_path):
    spec = tiny_spec(tmp_path)
    (tmp_path / "tiny.json").write_text(json.dumps(spec.to_dict()))
    source = str(Path(__file__).resolve().parents[1] / "src")
    def cli(*args, expected=0):
        process = subprocess.run([sys.executable, "-m", "molt_stream.cli", "--json", *args],
            cwd=tmp_path, env={**os.environ, "PYTHONPATH": source}, capture_output=True,
            text=True, encoding="utf-8", timeout=60)
        assert process.returncode == expected, process.stdout + process.stderr
        return json.loads(process.stdout) if expected == 0 else process.stderr
    cli("doctor")
    cli("config", "--init")
    cli("prepare", "--config", "tiny.json")
    cli("train", "--config", "tiny.json", "--dry-run")
    run = cli("train", "--config", "tiny.json", "-y")["run"]
    cli("evaluate", "--run", run)
    cli("report", "--run", run)
    cli("generate", "--run", run, "--prompt-ids", "1,2", "--max-new-tokens", "2")
    cli("export", "--run", run, "--output-dir", str(tmp_path / "bundle"))
    AtomicCheckpointStore(tmp_path / "bundle").load()
    assert "already complete" in cli("resume", "--run", run, expected=1)


@pytest.mark.parametrize("command", ["doctor", "runs", "fit-test", "export", "info", "inspect", "config",
    "prepare", "train", "resume", "benchmark", "evaluate", "report", "compare", "compare-paired",
    "frontier", "generate", "stream-tune", "fusion-benchmark", "curriculum-benchmark",
    "loss-partition-benchmark", "power-limit", "optimize-gpu", "qlora-benchmark", "research"])
def test_every_public_command_has_subprocess_help(tmp_path, command):
    source = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-m", "molt_stream.cli", command, "--help"], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": source}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_text_preparation_is_bounded_and_non_destructive(tmp_path):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, GPT2Config
    from molt_stream.data.text import prepare_text
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    local = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(local)
    GPT2Config(vocab_size=3, n_layer=1, n_head=1, n_embd=8).save_pretrained(local)
    text = tmp_path / "text.txt"
    text.write_text("hello world " * 100, encoding="utf-8")
    target = tmp_path / "prepared"
    result = prepare_text(str(text), str(local), str(target), 0.1, base_model=str(local))
    assert result["train_tokens"] == 180 and result["validation_tokens"] == 20
    with pytest.raises(FileExistsError):
        prepare_text(str(text), str(local), str(target))
    assert (target / "dataset.json").is_file()
    from molt_stream.core.specs import load_spec
    load_spec(result["training_config"]).validate()
