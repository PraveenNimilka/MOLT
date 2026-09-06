"""Paired all-layer test of versioned BF16 LoRA compute shadows."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from molt_stream.measurement.target import interpolate_nll_crossing


SEEDS = (1337, 2027, 4099)
ORDERS = (("baseline", "shadow"), ("shadow", "baseline"), ("baseline", "shadow"))


def _temperature_c() -> float:
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "nvidia-smi failed")
    return float(process.stdout.strip().splitlines()[0])


def _wait_for_temperature(maximum_c: float, timeout_seconds: float) -> float:
    deadline = time.monotonic() + timeout_seconds
    while True:
        temperature = _temperature_c()
        if temperature <= maximum_c:
            return temperature
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPU remained at {temperature} C")
        time.sleep(2.0)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    process = subprocess.run(
        [sys.executable, "-m", "molt_stream.cli", "--json", "train", "--config", str(path), "-y"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode:
        raise RuntimeError((process.stderr or process.stdout)[-4000:])
    run = Path(json.loads(process.stdout)["run"])
    metrics = json.loads((run / "metrics.summary.json").read_text(encoding="utf-8"))
    metrics["run"] = str(run)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-nll", type=float, default=1.54)
    parser.add_argument("--start-temperature-c", type=float, default=50.0)
    parser.add_argument("--cooldown-timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if not math.isfinite(args.target_nll):
        parser.error("target NLL must be finite")
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    template["qlora_scheduled_nf4_down_projection"] = False
    pairs: list[dict[str, Any]] = []
    for seed, order in zip(SEEDS, ORDERS, strict=True):
        trials: dict[str, dict[str, Any]] = {}
        for arm in order:
            start_c = _wait_for_temperature(
                args.start_temperature_c, args.cooldown_timeout_seconds
            )
            spec = copy.deepcopy(template)
            spec.update(
                seed=seed,
                qlora_bf16_adapter_shadows=arm == "shadow",
                artifacts_dir=str(root / "runs"),
            )
            measured = _run(spec, root / f"seed{seed}-{arm}.json")
            crossing = interpolate_nll_crossing(measured["evaluations"], args.target_nll)
            trials[arm] = {
                "run": measured["run"],
                "state": measured["state"],
                "start_temperature_c": start_c,
                "peak_temperature_c": measured["telemetry"]["peak_gpu_temperature_c"],
                "trainable_parameters": measured["active_trainable_parameter_count"],
                "compute_tokens_per_second": measured["committed_update_compute_tokens_per_second"],
                "end_to_end_tokens_per_second": measured["tokens_per_second"],
                "allocated_bytes": measured["cuda_peak_allocated_bytes"],
                "nvml_used_bytes": measured["telemetry"]["peak_gpu_used_bytes"],
                "final_nll": measured["evaluations"][-1]["nll"],
                "crossing": crossing,
            }
        reasons: list[str] = []
        baseline, shadow = trials["baseline"], trials["shadow"]
        if baseline["state"] != "completed" or shadow["state"] != "completed":
            reasons.append("incomplete run")
        if baseline["crossing"] is None or shadow["crossing"] is None:
            reasons.append("target NLL not reached")
        if shadow["final_nll"] > baseline["final_nll"] * 1.01:
            reasons.append("quality regression exceeds 1%")
        if shadow["peak_temperature_c"] > 72:
            reasons.append("thermal boundary exceeded")
        if shadow["trainable_parameters"] != baseline["trainable_parameters"]:
            reasons.append("trainable parameter mismatch")
        comparison: dict[str, float] = {}
        if not reasons:
            comparison = {
                "time_improvement_percent": 100.0 * (1.0 - shadow["crossing"]["seconds"] / baseline["crossing"]["seconds"]),
                "energy_improvement_percent": 100.0 * (1.0 - shadow["crossing"]["joules"] / baseline["crossing"]["joules"]),
                "final_nll_change_percent": 100.0 * (shadow["final_nll"] / baseline["final_nll"] - 1.0),
            }
            if comparison["time_improvement_percent"] <= 0:
                reasons.append("no time-to-target improvement")
            if comparison["energy_improvement_percent"] <= 0:
                reasons.append("no energy-to-target improvement")
        pairs.append({
            "seed": seed,
            "order": list(order),
            "trials": trials,
            "comparison": comparison,
            "passed": not reasons,
            "reasons": reasons,
        })
        _atomic_json(root / "results.partial.json", pairs)
        if reasons:
            break
    result = {
        "schema_version": 1,
        "experiment": "all-layer-bf16-lora-shadow-ab",
        "target_nll": args.target_nll,
        "pairs": pairs,
        "passed": all(pair["passed"] for pair in pairs),
        "promotion_allowed": all(pair["passed"] for pair in pairs),
        "limitations": [
            "Eight-update screen; a passing result still requires endurance validation",
            "The experiment stops after the first failed required pair",
        ],
    }
    _atomic_json(root / "results.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
