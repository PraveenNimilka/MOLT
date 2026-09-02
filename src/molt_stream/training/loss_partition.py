from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from molt_stream.core.errors import CapabilityError
from molt_stream.core.specs import TrainingSpec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.kernels.partitioned_loss import exact_partitioned_linear_cross_entropy
from molt_stream.measurement.telemetry import NVMLTelemetry
from molt_stream.training.model import SmallCausalLM


def select_partition_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a kill-test survivor without optimizing on validation quality."""
    eligible = [
        item for item in records
        if item["gradient_close"]
        and item["loss_close"]
        and (
            item["step_speedup_percent"] >= 5.0
            or (
                item["peak_memory_reduction_percent"] >= 15.0
                and item["step_speedup_percent"] >= -10.0
            )
        )
    ]
    if not eligible:
        return {
            "survived": False,
            "selected_chunk_size": None,
            "reason": "no exact candidate passed the preregistered time/memory Pareto kill gate",
        }
    selected = max(
        eligible,
        key=lambda item: (
            item["step_speedup_percent"], item["peak_memory_reduction_percent"]
        ),
    )
    return {
        "survived": True,
        "selected_chunk_size": selected["chunk_size"],
        "reason": "best exact candidate passing the preregistered Pareto kill gate",
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_updates(
    spec: TrainingSpec,
    *,
    state: dict[str, torch.Tensor],
    batch: tuple[torch.Tensor, torch.Tensor],
    chunk_size: int | None,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    device = torch.device(spec.stream.device)
    model = SmallCausalLM(spec.model).to(device)
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, fused=True)
    x, y = batch

    def update() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if chunk_size is None:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.flatten(0, 1), y.flatten()
                )
            else:
                hidden = model.hidden_states(x)
                loss = exact_partitioned_linear_cross_entropy(
                    hidden, model.lm_head.weight, y, chunk_size
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return float(loss.detach())

    for _ in range(warmup_steps):
        update()
    torch.cuda.synchronize()
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, fused=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    telemetry = NVMLTelemetry(0.02, enable_gpu=True)
    telemetry.start()
    started = time.perf_counter()
    losses = [update() for _ in range(measured_steps)]
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    measured = telemetry.stop()
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return {
        "chunk_size": chunk_size,
        "loss": losses[-1],
        "mean_loss": statistics.fmean(losses),
        "seconds": seconds,
        "seconds_per_update": seconds / measured_steps,
        "tokens_per_second": y.numel() * measured_steps / seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "energy_joules": measured["gpu_board_energy_joules"],
        "peak_temperature_c": measured["peak_gpu_temperature_c"],
        "thermal_throttle_observed": measured["thermal_throttle_observed"],
        "gradients": gradients,
    }


def benchmark_exact_loss_partitioning(
    spec: TrainingSpec,
    *,
    chunk_sizes: tuple[int, ...] = (1024, 2048, 4096, 8192),
    warmup_steps: int = 2,
    measured_steps: int = 5,
) -> dict[str, Any]:
    """Run the exact-loss numerical and time/memory cheap-kill frontier."""
    spec.validate()
    if spec.stream.device != "cuda" or not torch.cuda.is_available():
        raise CapabilityError("exact-loss partition benchmark requires CUDA")
    if min((*chunk_sizes, warmup_steps, measured_steps)) <= 0:
        raise ValueError("chunk sizes and step counts must be positive")
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    device = torch.device("cuda")
    template = SmallCausalLM(spec.model).to(device)
    initial_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
    del template
    batcher = MMapTokenBatcher(spec.data, seed=spec.seed, device=device)
    batch = batcher.batch(spec.batch_size)
    baseline = _run_updates(
        spec,
        state=initial_state,
        batch=batch,
        chunk_size=None,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    baseline_gradients = baseline.pop("gradients")
    records: list[dict[str, Any]] = []
    for chunk_size in chunk_sizes:
        candidate = _run_updates(
            spec,
            state=initial_state,
            batch=batch,
            chunk_size=chunk_size,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )
        gradients = candidate.pop("gradients")
        maximum_gradient_error = max(
            float((gradients[name] - reference).abs().max())
            for name, reference in baseline_gradients.items()
        )
        reference_gradient_scale = max(
            float(reference.abs().max()) for reference in baseline_gradients.values()
        )
        relative_gradient_error = maximum_gradient_error / max(reference_gradient_scale, 1e-12)
        candidate["loss_close"] = abs(candidate["loss"] - baseline["loss"]) <= max(
            1e-4, abs(baseline["loss"]) * 1e-4
        )
        candidate["maximum_gradient_absolute_error"] = maximum_gradient_error
        candidate["maximum_gradient_relative_error"] = relative_gradient_error
        candidate["gradient_close"] = relative_gradient_error <= 0.01
        candidate["step_speedup_percent"] = 100.0 * (
            1.0 - candidate["seconds_per_update"] / baseline["seconds_per_update"]
        )
        candidate["peak_memory_reduction_percent"] = 100.0 * (
            1.0 - candidate["peak_allocated_bytes"] / baseline["peak_allocated_bytes"]
        )
        if baseline["energy_joules"] and candidate["energy_joules"] is not None:
            candidate["energy_reduction_percent"] = 100.0 * (
                1.0 - candidate["energy_joules"] / baseline["energy_joules"]
            )
        else:
            candidate["energy_reduction_percent"] = None
        records.append(candidate)
    decision = select_partition_candidate(records)
    report = {
        "schema_version": 1,
        "experiment_id": "EXP-1030",
        "method": "exact-partitioned-linear-cross-entropy",
        "novelty_claim": "none; chunked/fused linear CE and Cut Cross-Entropy are prior art",
        "workload": spec.to_dict(),
        "execution_backend_used": "eager",
        "energy_measurement_limit": (
            "Short sequential screening intervals are not matched-temperature energy evidence; "
            "energy is recorded but excluded from candidate selection."
        ),
        "baseline": baseline,
        "candidates": records,
        "decision": decision,
        "gate": (
            "loss and gradients equivalent; then >=5% step speedup, or >=15% peak-memory "
            "reduction with no more than 10% step slowdown"
        ),
    }
    output = Path(spec.artifacts_dir).parent / "experiments" / f"exp-1030-{time.strftime('%Y%m%d-%H%M%S')}.json"
    _atomic_json(output, report)
    report["artifact_path"] = str(output)
    return report
