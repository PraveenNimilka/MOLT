from __future__ import annotations

import json
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from molt_stream.core.specs import GateSpec, StreamSpec
from molt_stream.measurement.telemetry import NVMLTelemetry
from molt_stream.streaming.engine import StreamedLayer, StreamedLayerEngine, StreamedLoRALinear
from molt_stream.streaming.nf4 import NF4Tensor


def stream_tune(
    *,
    width: int = 1024,
    layers: int = 8,
    sequence: int = 128,
    batch: int = 1,
    steps: int = 10,
    rank: int = 8,
    double_buffer: bool = True,
    bundle_size: int = 4,
    seed: int = 1337,
    output_dir: str | Path = "artifacts/molt-stream/benchmarks",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("stream-tune requires CUDA")
    if min(width, layers, sequence, batch, steps, rank, bundle_size) <= 0 or rank >= width:
        raise ValueError("invalid benchmark dimensions")
    torch.manual_seed(seed)
    device = torch.device("cuda")
    spec = StreamSpec(
        device="cuda", compute_dtype="bfloat16", lora_rank=rank,
        double_buffer=double_buffer, bundle_size=bundle_size,
    )
    sources = [
        NF4Tensor.quantize(torch.randn(width, width) / width**0.5, pin_memory=True)
        for _ in range(layers)
    ]
    modules = [StreamedLoRALinear(source, rank, 2 * rank, device) for source in sources]
    engine = StreamedLayerEngine(
        [StreamedLayer(module) for module in modules], spec
    )
    optimizer = torch.optim.AdamW(engine.parameters(), lr=2e-4)
    value = torch.randn(batch, sequence, width, device=device, dtype=torch.bfloat16)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        engine(value).float().square().mean().backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    telemetry = NVMLTelemetry(0.05)
    telemetry.start()
    completed = 0
    started = time.perf_counter()
    thermal_abort = False
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = engine(value).float().square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        completed += 1
        temperatures = [p.gpu_temperature_c for p in telemetry.points if p.gpu_temperature_c is not None]
        if temperatures and max(temperatures) > GateSpec().maximum_temperature_c:
            thermal_abort = True
            break
    seconds = time.perf_counter() - started
    measured = telemetry.stop()
    vectors = completed * batch * sequence
    energy = measured["gpu_board_energy_joules"]
    report = {
        "schema_version": 1,
        "benchmark": "molt-streamed-nf4-lora-mechanism",
        "workload": {"width": width, "layers": layers, "sequence": sequence, "batch": batch, "steps": steps, "rank": rank, "double_buffer": double_buffer, "bundle_size": bundle_size, "seed": seed},
        "completed_steps": completed,
        "thermal_abort": thermal_abort,
        "seconds": seconds,
        "activation_vectors_per_second": vectors / seconds if seconds else None,
        "gpu_board_energy_joules": energy,
        "joules_per_activation_vector": float(energy) / vectors if energy is not None and vectors else None,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        "quantized_host_base_bytes": sum(source.storage_bytes for source in sources),
        "telemetry": {key: value for key, value in measured.items() if key != "points"},
        "streaming": engine.last_forward,
        "gate_scope": (
            "This synthetic residual-linear mechanism benchmark is not language-model tokens/s "
            "and cannot pass the 8B quality or throughput gates."
        ),
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"stream-tune-{time.strftime('%Y%m%d-%H%M%S')}.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    report["artifact_path"] = str(path)
    return report


def inspect_capabilities() -> dict[str, Any]:
    import importlib.util

    value: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "system_ram_bytes": psutil.virtual_memory().total,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "triton": importlib.util.find_spec("triton") is not None,
        "liger_kernel": importlib.util.find_spec("liger_kernel") is not None,
        "bitsandbytes": importlib.util.find_spec("bitsandbytes") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "peft": importlib.util.find_spec("peft") is not None,
        "msvc_cl": shutil.which("cl"),
        "nvcc": shutil.which("nvcc"),
        "ninja": shutil.which("ninja"),
    }
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        CUDA_HOME = None
    value["cuda_toolkit_home"] = CUDA_HOME
    if torch.cuda.is_available():
        value.update(
            gpu=torch.cuda.get_device_name(0),
            gpu_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
            cuda_runtime=torch.version.cuda,
        )
    value["qlora_8b_ready"] = all(
        value[name] for name in ("bitsandbytes", "transformers", "peft")
    )
    value["torch_compile_cuda_ready"] = bool(value["cuda_available"] and value["triton"])
    value["native_cpp_cuda_extension_ready"] = bool(
        value["msvc_cl"] and value["nvcc"] and value["ninja"] and CUDA_HOME
    )
    return value
