import json

import pytest
import torch


def _tiny_tokenizer(path):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, PreTrainedTokenizerFast
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(path)
    GPT2Config(vocab_size=3, n_layer=1, n_head=1, n_embd=8).save_pretrained(path)


def test_jsonl_preparation_uses_document_boundary(tmp_path):
    from molt_stream.data.records import prepare_records
    model = tmp_path / "model"
    _tiny_tokenizer(model)
    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps({"text": "hello world"}) for _ in range(4)), encoding="utf-8")
    target = tmp_path / "prepared"
    result = prepare_records(str(source), str(model), str(target), 0.25, str(model))
    assert result["train_records"] == 3
    assert result["validation_records"] == 1
    assert result["train_tokens"] == 6
    assert result["validation_tokens"] == 2
    assert result["record_schema"] == ["text"]
    assert (target / "training.json").is_file()


def test_jsonl_invalid_schema_is_actionable(tmp_path):
    from molt_stream.data.records import prepare_records
    model = tmp_path / "model"
    _tiny_tokenizer(model)
    source = tmp_path / "records.jsonl"
    source.write_text('{"unexpected": "value"}\n{"unexpected": "value"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot infer schema"):
        prepare_records(str(source), str(model), str(tmp_path / "prepared"), 0.5)


def test_qlora_export_writes_standard_safetensors_adapter(tmp_path):
    pytest.importorskip("safetensors.torch")
    from safetensors.torch import load_file
    from molt_stream.experiments.export import export_run
    from molt_stream.experiments.store import AtomicCheckpointStore
    run = tmp_path / "run"
    adapters = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
    }
    AtomicCheckpointStore(run).save({"adapters": adapters, "step": 1})
    (run / "spec.resolved.json").write_text(json.dumps({
        "mode": "qlora", "base_model": str(tmp_path / "base"),
        "stream": {"lora_rank": 2, "lora_alpha": 4.0},
    }), encoding="utf-8")
    target = tmp_path / "adapter"
    result = export_run(str(run), str(target))
    assert result["format"] == "Hugging Face PEFT safetensors adapter"
    assert torch.equal(load_file(target / "adapter_model.safetensors")[next(iter(adapters))],
                       next(iter(adapters.values())))
    config = json.loads((target / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["target_modules"] == ["q_proj"]
    manifest = json.loads((target / "export.json").read_text(encoding="utf-8"))
    assert "adapter_model.safetensors" in manifest["files"]
    assert manifest["base_weights_included"] is False
