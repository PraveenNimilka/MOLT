"""Diagnostic profiler; instrumented timings are not acceptance benchmarks."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from molt_stream.core.specs import load_spec, TrainingMode
from molt_stream.core.contracts import ProgressEvent
from molt_stream.training.engine import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = load_spec(args.config)
    if spec.mode != TrainingMode.QLORA:
        parser.error("QLoRA configuration required")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    spec = replace(spec, max_steps=3, evaluation_interval=3, artifacts_dir=str(root / "runs"))
    previous_step = 0
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
        record_shapes=True, profile_memory=True,
        on_trace_ready=lambda p: p.export_chrome_trace(str(root / "trace.json")),
    ) as profiler:
        def progress(event: ProgressEvent) -> None:
            nonlocal previous_step
            if event.kind == "step" and event.step > previous_step:
                previous_step = event.step
                profiler.step()
        run = train(spec, progress=progress)
    rows = [{"operation": entry.key, "calls": entry.count,
             "device_type": str(entry.device_type), "input_shapes": entry.input_shapes,
             "self_cuda_us": entry.self_device_time_total,
             "self_cpu_us": entry.self_cpu_time_total}
            for entry in profiler.key_averages(group_by_input_shape=True)]
    rows.sort(key=lambda item: item["self_cuda_us"], reverse=True)
    (root / "operators.json").write_text(json.dumps({
        "run": str(run), "instrumented": True, "operators": rows,
        "note": "Instrumented timings are not throughput benchmarks. CPU ATen attribution and native CUDA kernel rows overlap; never sum them together. Use shapes and the trace to attribute vocabulary-head work.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(rows[:15], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
