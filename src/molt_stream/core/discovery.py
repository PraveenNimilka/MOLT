"""MOLT Workspace & Hardware Discovery.

Provides automatic zero-configuration discovery of:
- NVIDIA GPU capabilities, VRAM, and power limits via NVML.
- Workspace directory structure (molt-workspace/).
- Local model weights (Hugging Face / safetensors).
- Local binary datasets (*.bin).
- Historical training runs and checkpoint integrity.
"""
from __future__ import annotations

from molt_stream.core.workspace import workspace_paths

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def get_hardware_info() -> dict[str, Any]:
    """Inspect system hardware and return structured diagnostic details."""
    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version.split()[0],
        "cpu": platform.processor() or "Unknown CPU",
        "cpu_cores_physical": None,
        "cpu_cores_logical": os.cpu_count(),
        "total_ram_gb": None,
        "available_ram_gb": None,
        "gpu_detected": False,
        "gpu_name": None,
        "gpu_vram_total_gb": None,
        "gpu_vram_free_gb": None,
        "gpu_power_limit_w": None,
        "gpu_temperature_c": None,
        "cpu_temperature_c": None,
        "cpu_temperature_source": None,
        "gpu_driver": None,
        "cuda_version": None,
        "pytorch_version": None,
        "suggested_profile": "BALANCED",
        "suggested_memory_budget_gb": 6.5,
        "suggested_context_length": 1024,
    }

    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_ram_gb"] = round(vm.total / (1024**3), 2)
        info["available_ram_gb"] = round(vm.available / (1024**3), 2)
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
    except Exception:
        pass

    try:
        from molt_stream.core.cpu_temperature import CpuTemperatureReader

        cpu_temperature = CpuTemperatureReader(cache_seconds=0.1)
        info["cpu_temperature_c"] = cpu_temperature.read()
        info["cpu_temperature_source"] = cpu_temperature.source
    except Exception:
        pass

    try:
        import torch
        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_detected"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_total_gb"] = round(props.total_memory / (1024**3), 2)
    except Exception:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            if not info["gpu_name"]:
                info["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
                info["gpu_detected"] = True
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["gpu_vram_total_gb"] = round(mem.total / (1024**3), 2)
            info["gpu_vram_free_gb"] = round(mem.free / (1024**3), 2)
            info["gpu_temperature_c"] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            info["gpu_driver"] = pynvml.nvmlSystemGetDriverVersion()
            try:
                info["gpu_power_limit_w"] = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0, 1)
            except Exception:
                pass
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    # Determine suggestions based on detected hardware
    vram = info.get("gpu_vram_total_gb")
    if vram:
        if vram <= 8.5:
            info["suggested_profile"] = "BALANCED"
            info["suggested_memory_budget_gb"] = 6.8
            info["suggested_context_length"] = 1024
        elif vram <= 12.5:
            info["suggested_profile"] = "BALANCED"
            info["suggested_memory_budget_gb"] = 10.0
            info["suggested_context_length"] = 2048
        else:
            info["suggested_profile"] = "SPEED"
            info["suggested_memory_budget_gb"] = round(vram * 0.85, 1)
            info["suggested_context_length"] = 2048

    return info


def find_models(search_roots: list[Path] | None = None) -> list[dict[str, Any]]:
    """Scan candidate directories for loadable models."""
    if search_roots is None:
        search_roots = workspace_paths("models") or [
            Path("models"),
            Path("molt-workspace/models"),
            Path("../AI MODELS"),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]

    models: list[dict[str, Any]] = []
    seen: set[str] = set()

    for root in search_roots:
        if not root.exists():
            continue
        # Direct check if root itself is a model
        if (root / "config.json").exists():
            resolved = str(root.resolve())
            if resolved not in seen:
                seen.add(resolved)
                models.append({
                    "name": root.name,
                    "path": str(root),
                    "type": "HuggingFace/Safetensors",
                    "has_weights": any(root.glob("*.safetensors")) or (root / "pytorch_model.bin").exists(),
                })
            continue

        # Scan immediate children
        try:
            for child in root.iterdir():
                if child.is_dir() and (child / "config.json").exists():
                    resolved = str(child.resolve())
                    if resolved not in seen:
                        seen.add(resolved)
                        models.append({
                            "name": child.name,
                            "path": str(child),
                            "type": "HuggingFace/Safetensors",
                            "has_weights": any(child.glob("*.safetensors")) or (child / "pytorch_model.bin").exists(),
                        })
        except Exception:
            pass

    return models


def find_datasets(search_roots: list[Path] | None = None) -> list[dict[str, Any]]:
    """Scan candidate directories for binary token datasets."""
    if search_roots is None:
        search_roots = workspace_paths("datasets") or [
            Path("datasets"),
            Path("molt-workspace/datasets"),
            Path("data/prepared"),
            Path("data"),
        ]

    datasets: list[dict[str, Any]] = []
    seen: set[str] = set()

    for root in search_roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*.bin"):
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    size = path.stat().st_size
                    # 4 bytes per int32 token
                    est_tokens = size // 4 if size % 4 == 0 else size
                    datasets.append({
                        "name": path.name,
                        "path": str(path),
                        "bytes": size,
                        "estimated_tokens": est_tokens,
                        "parent": path.parent.name,
                    })
        except Exception:
            pass

    return datasets


def find_runs(search_roots: list[Path] | None = None) -> list[dict[str, Any]]:
    """Scan candidate directories for prior training runs and checkpoints."""
    if search_roots is None:
        search_roots = workspace_paths("runs") or [
            Path("runs"),
            Path("molt-workspace/runs"),
            Path("artifacts/molt-stream/runs"),
            Path("artifacts/molt-stream"),
        ]

    runs: list[dict[str, Any]] = []
    seen: set[str] = set()

    for root in search_roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("spec.resolved.json"):
                run_dir = path.parent
                resolved = str(run_dir.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)

                spec_data = {}
                try:
                    spec_data = json.loads(path.read_text("utf-8"))
                except Exception:
                    pass

                summary_data = {}
                summary_file = run_dir / "metrics.summary.json"
                if summary_file.exists():
                    try:
                        summary_data = json.loads(summary_file.read_text("utf-8"))
                    except Exception:
                        pass

                has_checkpoint = (run_dir / "checkpoint.pt").exists()
                integrity = "Unknown"
                if has_checkpoint:
                    complete_json = run_dir / "checkpoint.complete.json"
                    if complete_json.exists():
                        try:
                            meta = json.loads(complete_json.read_text("utf-8"))
                            if meta.get("sha256"):
                                from molt_stream.core.integrity import valid_checkpoint
                                integrity = ("Verified (SHA-256)" if valid_checkpoint(
                                    run_dir / "checkpoint.pt", complete_json
                                ) else "Invalid checkpoint")
                        except Exception:
                            integrity = "Unverified"
                    else:
                        integrity = "Raw (.pt)"

                runs.append({
                    "run_id": run_dir.name,
                    "path": str(run_dir),
                    "mode": spec_data.get("mode", summary_data.get("mode", "unknown")),
                    "step": summary_data.get("step", 0),
                    "tokens": summary_data.get("tokens", 0),
                    "state": summary_data.get("state", "in_progress" if has_checkpoint else "unstarted"),
                    "checkpoint": has_checkpoint,
                    "integrity": integrity,
                    "mtime": run_dir.stat().st_mtime,
                })
        except Exception:
            pass

    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def init_workspace(root: Path = Path("molt-workspace")) -> dict[str, Path]:
    """Initialize a standard user workspace with models, datasets, runs, and configs."""
    dirs = {
        "root": root,
        "models": root / "models",
        "datasets": root / "datasets",
        "runs": root / "runs",
        "configs": root / "configs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# MOLT Workspace\n\n"
            "Place your model weights in `models/` and token datasets in `datasets/`.\n"
            "MOLT will automatically discover them when you run `molt` or `molt train`.\n",
            encoding="utf-8",
        )

    sample_config = dirs["configs"] / "default_training.json"
    if not sample_config.exists():
        sample = {
            "mode": "qlora",
            "base_model": str((root / "models/qwen2-0.5b").resolve()),
            "data": {
                "path": str((root / "datasets/train.bin").resolve()),
                "validation_path": str((root / "datasets/val.bin").resolve()),
                "context_length": 1024,
                "storage_dtype": "int32",
                "sequential": True,
                "packing": "contiguous"
            },
            "batch_size": 1,
            "gradient_accumulation": 1,
            "max_steps": 1000,
            "learning_rate": 0.00015,
            "seed": 2026,
            "artifacts_dir": str(dirs["runs"].resolve())
        }
        sample_config.write_text(json.dumps(sample, indent=2), encoding="utf-8")

    manifest = root / "molt-workspace.json"
    if not manifest.exists():
        with manifest.open("x", encoding="utf-8") as handle:
            json.dump({"schema_version": 1, "paths": {key: str(path.resolve())
                      for key, path in dirs.items()}}, handle, indent=2)
    return dirs
