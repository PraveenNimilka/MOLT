"""Paired repeated-window cache falsification, including cold-fill and thermal cost.

Each arm must run in a separate process. This deliberately small repeated-data
workload tests runtime equivalence, not generalization or single-pass streaming.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import traceback

import torch

from molt_stream.core.specs import load_spec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.measurement.telemetry import NVMLTelemetry, integrate_board_energy
from molt_stream.measurement.thermal import passive_cooldown, wait_for_thermal_headroom
from molt_stream.measurement.update_accounting import UpdateAccounting
from molt_stream.training.evaluation_guard import guarded_decoder_evaluation
from molt_stream.training.frozen_prefix import FrozenPrefixReplay
from molt_stream.training.qlora import _build_qlora_model, _build_qlora_optimizer, _shifted_causal_loss
from molt_stream.training.update_transaction import UpdateTransaction


class ThermalStop(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--cache-mib", type=int, default=128)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--forward-guard", action="store_true",
                        help="Check temperature between training decoder blocks as well")
    parser.add_argument("--start-temperature-c", type=float, default=50.0)
    args = parser.parse_args()
    spec = load_spec(args.config)
    if args.seed is not None:
        spec = replace(spec, seed=args.seed)
    spec.validate()
    if (args.windows < 1 or args.epochs < 1 or args.cache_mib < 1
            or args.windows % spec.gradient_accumulation):
        parser.error("positive windows must be divisible by gradient accumulation")
    if spec.qlora_train_last_layers is None or spec.qlora_full_warmup_steps:
        parser.error("Use an already-frozen upper-layer spec without a warmup schedule")
    if not spec.data.validation_path:
        parser.error("A separate validation path is required")
    if not 0 < args.start_temperature_c < spec.thermal_target_c:
        parser.error("start temperature must be positive and below the pacing target")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    source_snapshot = output.with_suffix(".source.py")
    if source_snapshot.exists():
        raise FileExistsError(source_snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_text = Path(__file__).read_text(encoding="utf-8")
    source_snapshot.write_text(source_text, encoding="utf-8")
    random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    accounting = UpdateAccounting()
    result: dict[str, object] = {
        "state": "failed", "spec": spec.to_dict(), "probe": vars(args),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "peft", "bitsandbytes")},
        "updates": [], "evaluations": [],
        "limitations": ["Repeated subset, not a complete-dataset or generalization test",
                        "Imports, output serialization and process teardown excluded",
                        "CPU temperature unavailable; no resumable checkpoint in this probe",
                        "Upper-layer adaptation, not full-parameter training"],
    }
    telemetry = NVMLTelemetry(0.05)
    telemetry.start()
    started = time.perf_counter()
    pause_seconds = 0.0
    peak_temperature: float | None = None

    def read():
        nonlocal peak_temperature
        point = telemetry.thermal_point()
        if point is not None and point.gpu_temperature_c is not None:
            peak_temperature = max(peak_temperature or point.gpu_temperature_c, point.gpu_temperature_c)
        return point

    def gate() -> bool:
        nonlocal pause_seconds
        decision = wait_for_thermal_headroom(read, target_c=spec.thermal_target_c,
                                            abort_c=spec.thermal_abort_c)
        pause_seconds += decision.pause_seconds
        if decision.stop_reason:
            raise ThermalStop(decision.stop_reason)
        return True

    try:
        cold = passive_cooldown(read, recovery_c=args.start_temperature_c)
        pause_seconds += cold.pause_seconds
        if cold.stop_reason:
            raise ThermalStop(cold.stop_reason)
        result["start_temperature_c"] = cold.temperature_c
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = _build_qlora_model(spec)
        optimizer = _build_qlora_optimizer(model, spec)
        train_data = MMapTokenBatcher(spec.data, seed=spec.seed, device="cuda")
        epoch_start = train_data.state_dict()
        result["setup_seconds"] = time.perf_counter() - started

        @torch.no_grad()
        def evaluate(step: int) -> None:
            model.eval()
            data = MMapTokenBatcher(replace(spec.data, path=spec.data.validation_path,
                validation_path=None), seed=spec.seed + 1, device="cuda")
            losses = []
            try:
                for _ in range(spec.qlora_validation_batches):
                    gate()
                    x, y = data.batch(1)
                    with guarded_decoder_evaluation(model, gate):
                        loss = _shifted_causal_loss(model, x, y, chunk_size=spec.qlora_loss_chunk_size)
                    losses.append(float(loss))
                    gate()
                result["evaluations"].append({"step": step, "window_nll": losses,
                    "nll": sum(losses) / len(losses), "seconds": time.perf_counter() - started,
                    "board_joules": integrate_board_energy(telemetry.points)})
            finally:
                model.train()

        evaluate(0)
        boundary = int(model.get_base_model().config.num_hidden_layers) - spec.qlora_train_last_layers
        context = (FrozenPrefixReplay(model, boundary=boundary, max_bytes=args.cache_mib * 1024**2)
                   if args.cache else nullcontext())
        step = 0
        with context as cache:
            try:
                for epoch in range(args.epochs):
                    train_data.load_state_dict(epoch_start)
                    for _ in range(args.windows // spec.gradient_accumulation):
                        optimizer.zero_grad(set_to_none=True)
                        transaction = UpdateTransaction.capture(train_data, cuda=True)
                        update_started = time.perf_counter()
                        pause_before = pause_seconds
                        pending = 0
                        committed = False
                        total_loss = 0.0
                        try:
                            for _ in range(spec.gradient_accumulation):
                                gate()
                                x, y = train_data.batch(spec.batch_size)
                                pending += y.numel()
                                guard = (guarded_decoder_evaluation(model, gate)
                                         if args.forward_guard else nullcontext())
                                with guard:
                                    loss = _shifted_causal_loss(model, x, y,
                                        autocast=spec.qlora_autocast, chunk_size=spec.qlora_loss_chunk_size,
                                        precompute_head_gradient=spec.qlora_precompute_head_gradient
                                    ) / spec.gradient_accumulation
                                torch.cuda.synchronize()
                                gate()
                                loss.backward()
                                total_loss += float(loss.detach())
                                del loss
                                gate()
                            optimizer.step()
                            torch.cuda.synchronize()
                            committed = True
                            step += 1
                        finally:
                            if not committed:
                                transaction.rollback(train_data, optimizer)
                            elapsed = time.perf_counter() - update_started
                            paused = pause_seconds - pause_before
                            accounting.record(elapsed_seconds=elapsed, pause_seconds=paused,
                                              tokens=pending, committed=committed)
                            result["updates"].append({"epoch": epoch, "step": step,
                                "committed": committed, "seconds": elapsed, "pause_seconds": paused,
                                "loss": total_loss})
                        gate()
            finally:
                if cache is not None:
                    result["cache"] = {"hits": cache.hits, "misses": cache.misses,
                                       "resident_tensor_bytes": cache.resident_bytes}
        evaluate(step)
        result["state"] = "completed"
    except ThermalStop as exc:
        result.update(state="thermal_stop", error=str(exc))
    except Exception as exc:
        result.update(state="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    finally:
        measured = telemetry.stop()
        elapsed = time.perf_counter() - started
        result.update(accounting.summary())
        result.update(seconds=elapsed, committed_tokens=accounting.committed_tokens,
            discarded_tokens=accounting.discarded_tokens, thermal_pause_seconds=pause_seconds,
            end_to_end_tokens_per_second=accounting.committed_tokens / elapsed,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
            peak_gate_temperature_c=peak_temperature, telemetry=measured)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in {"telemetry", "spec", "updates"}}, indent=2))
    return 0 if result["state"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
