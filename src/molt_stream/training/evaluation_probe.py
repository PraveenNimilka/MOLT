"""Thermally paced QLoRA evaluation for research comparisons."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from molt_stream.core.specs import load_spec
from molt_stream.data.bytes import MMapTokenBatcher
from molt_stream.experiments.store import AtomicCheckpointStore
from molt_stream.measurement.telemetry import NVMLTelemetry, integrate_board_energy
from molt_stream.measurement.thermal import passive_cooldown
from molt_stream.training.qlora import _build_qlora_model, _shifted_causal_loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--recovery-c", type=float, default=55.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    root = Path(args.run)
    spec = load_spec(root / "spec.resolved.json")
    telemetry = NVMLTelemetry(0.1)
    telemetry.start()
    started = time.perf_counter()
    from peft import set_peft_model_state_dict

    model = _build_qlora_model(spec)
    state = AtomicCheckpointStore(root).load(map_location="cpu")
    set_peft_model_state_dict(model, state["adapters"])
    data_spec = type(spec.data)(**{
        **asdict(spec.data), "path": spec.data.validation_path or spec.data.path,
        "validation_path": None,
    })
    batcher = MMapTokenBatcher(data_spec, seed=spec.seed + 1, device="cuda")
    model.eval()
    losses: list[float] = []
    cooling = 0.0
    with torch.no_grad():
        for index in range(args.batches):
            x, y = batcher.batch(1)
            losses.append(float(_shifted_causal_loss(
                model, x, y, chunk_size=spec.qlora_loss_chunk_size)))
            if index + 1 < args.batches:
                decision = passive_cooldown(
                    telemetry.thermal_point, recovery_c=args.recovery_c,
                )
                cooling += decision.pause_seconds
                if decision.stop_reason:
                    raise RuntimeError(decision.stop_reason)
    measured = telemetry.stop()
    nll = sum(losses) / len(losses)
    result = {
        "run": str(root), "batches": args.batches, "losses": losses,
        "validation_nll": nll, "validation_perplexity": math.exp(nll),
        "seconds": time.perf_counter() - started, "passive_cooling_seconds": cooling,
        "board_energy_joules": integrate_board_energy(telemetry.points),
        "peak_gpu_temperature_c": measured.get("peak_gpu_temperature_c"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
