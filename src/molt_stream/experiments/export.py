"""Atomic, verified exports for MOLT checkpoints and interoperable LoRA adapters."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import torch

from molt_stream.core.integrity import sha256
from molt_stream.experiments.store import AtomicCheckpointStore


def _read_spec(source: Path) -> dict[str, Any]:
    value = json.loads((source / "spec.resolved.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Resolved run specification must be a JSON object")
    return value


def _write_manifest(package: Path, *, format_name: str, base_model: str | None,
                    base_weights_included: bool, notes: str) -> None:
    files = {
        str(path.relative_to(package)).replace("\\", "/"): {
            "sha256": sha256(path), "bytes": path.stat().st_size,
        }
        for path in sorted(package.rglob("*")) if path.is_file() and path.name != "export.json"
    }
    (package / "export.json").write_text(json.dumps({
        "schema_version": 2, "format": format_name, "base_model": base_model,
        "base_weights_included": base_weights_included, "datasets_included": False,
        "notes": notes, "files": files,
    }, indent=2), encoding="utf-8")


def _copy_molt_bundle(source: Path, checkpoint: Path, package: Path,
                      spec: dict[str, Any]) -> str:
    shutil.copy2(checkpoint, package / "checkpoint.pt")
    shutil.copy2(checkpoint.with_suffix(".complete.json"), package / "checkpoint.complete.json")
    shutil.copy2(source / "spec.resolved.json", package / "spec.resolved.json")
    if (source / "metrics.summary.json").exists():
        shutil.copy2(source / "metrics.summary.json", package / "metrics.summary.json")
    _write_manifest(
        package, format_name="MOLT checkpoint bundle", base_model=spec.get("base_model"),
        base_weights_included=False,
        notes="Resume artifact; local paths may need updating. This is not a standalone model.",
    )
    return "MOLT checkpoint bundle"


def _adapter_target_modules(adapters: dict[str, torch.Tensor]) -> list[str]:
    result = {
        parts[-3] for name in adapters for parts in [name.split(".")]
        if len(parts) >= 3 and parts[-2] in {"lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"}
    }
    if not result:
        raise ValueError("Checkpoint adapter names are not recognized as PEFT LoRA parameters")
    return sorted(result)


def _write_hf_adapter(source: Path, package: Path, spec: dict[str, Any]) -> None:
    state = AtomicCheckpointStore(source).load(map_location="cpu")
    raw = state.get("adapters")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("The verified checkpoint contains no QLoRA adapter state")
    adapters: dict[str, torch.Tensor] = {}
    for name, tensor in raw.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Adapter state must map parameter names to tensors")
        if tensor.layout != torch.strided:
            raise ValueError(f"Adapter tensor '{name}' is not a dense strided tensor")
        adapters[name] = tensor.detach().cpu().contiguous()
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required; re-run install.ps1 with QLoRA support") from exc
    save_file(adapters, package / "adapter_model.safetensors", metadata={"format": "pt"})
    stream = spec.get("stream") if isinstance(spec.get("stream"), dict) else {}
    config = {
        "base_model_name_or_path": spec.get("base_model"), "bias": "none",
        "inference_mode": True, "lora_alpha": stream.get("lora_alpha", 16.0),
        "lora_dropout": 0.0, "peft_type": "LORA", "r": stream.get("lora_rank", 8),
        "target_modules": _adapter_target_modules(adapters), "task_type": "CAUSAL_LM",
    }
    (package / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (package / "README.md").write_text(
        "# MOLT LoRA adapter\n\nThis package contains adapter weights only. "
        "Load it with the base model named in `adapter_config.json`.\n", encoding="utf-8",
    )


def _convert_gguf(package: Path, spec: dict[str, Any], llama_cpp: str | None) -> None:
    if not llama_cpp:
        raise ValueError("GGUF export requires --llama-cpp PATH to an official llama.cpp checkout")
    converter = Path(llama_cpp).resolve() / "convert_lora_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"llama.cpp converter not found: {converter}")
    base_model = spec.get("base_model")
    if not isinstance(base_model, str) or not Path(base_model).exists():
        raise ValueError("GGUF adapter conversion requires the run's accessible local base model")
    output = package / "adapter.gguf"
    process = subprocess.run([
        sys.executable, str(converter), str(package), "--base", base_model,
        "--outfile", str(output), "--outtype", "f16",
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if process.returncode or not output.is_file():
        detail = (process.stderr or process.stdout).strip()[-2000:]
        raise RuntimeError(f"llama.cpp GGUF conversion failed: {detail}")


def export_run(run: str, output_dir: str, *, format: str = "auto",
               llama_cpp: str | None = None) -> dict[str, str]:
    source, target = Path(run).resolve(), Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Export destination already exists: {target}")
    checkpoint = AtomicCheckpointStore(source).resolve()
    spec = _read_spec(source)
    selected = "hf" if format == "auto" and str(spec.get("mode", "")).lower() == "qlora" else format
    if selected == "auto":
        selected = "molt"
    if selected not in {"molt", "hf", "gguf"}:
        raise ValueError("format must be one of: auto, molt, hf, gguf")
    if selected in {"hf", "gguf"} and str(spec.get("mode", "")).lower() != "qlora":
        raise ValueError("Hugging Face and GGUF adapter export require a QLoRA run")

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="molt-export-", dir=target.parent) as temporary:
        package = Path(temporary) / "run"
        package.mkdir()
        if selected == "molt":
            format_name = _copy_molt_bundle(source, checkpoint, package, spec)
        else:
            _write_hf_adapter(source, package, spec)
            if selected == "gguf":
                _convert_gguf(package, spec, llama_cpp)
                format_name = "GGUF LoRA adapter package"
                notes = "Adapter only; load with a compatible GGUF base model. Conversion performed by llama.cpp."
            else:
                format_name = "Hugging Face PEFT safetensors adapter"
                notes = "Adapter only; load with the base model named in adapter_config.json."
            _write_manifest(package, format_name=format_name, base_model=spec.get("base_model"),
                            base_weights_included=False, notes=notes)
        os.rename(package, target)
    return {"directory": str(target), "format": format_name}
