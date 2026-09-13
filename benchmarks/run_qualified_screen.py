"""Persist randomized forward/reverse order before launching isolated screens."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unsloth-python", required=True)
    parser.add_argument("--seeds", default="1337,2027,4099")
    parser.add_argument("--arms", default="hf,reference,qualified,unsloth")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--startup-max-c", type=float, default=50.0)
    parser.add_argument("--startup-dwell-seconds", type=float, default=20.0)
    parser.add_argument("--emergency-guard-c", type=float, default=83.0)
    parser.add_argument("--qualified-micro-paced", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    arms = args.arms.split(",")
    if len(arms) != len(set(arms)):
        parser.error("duplicate arm")
    order = []
    rng = random.Random(912026)
    for seed in map(int, args.seeds.split(",")):
        permutation = rng.sample(arms, len(arms))
        for repeat, sequence in enumerate((permutation, permutation[::-1])):
            order.extend({"seed": seed, "repeat": repeat, "arm": arm} for arm in sequence)
    plan = {"randomization_seed": 912026, "order": order, "steps": args.steps,
            "validation_batches": args.validation_batches, "stage": "screening",
            "thermal_start": {"maximum_c": args.startup_max_c,
                              "dwell_seconds": args.startup_dwell_seconds,
                              "emergency_guard_c": args.emergency_guard_c},
            "promotion_requires": ["five-seed 1M-token matched repetitions",
                "positive paired 95% CIs for end-to-end speed and energy savings",
                "no allocated/reserved/whole-GPU memory regression",
                "final NLL no worse than reference by more than 1%",
                "no thermal abort", "controlled clocks and power", "matching initial adapters"]}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve()) + os.pathsep + str(Path.cwd())
    fingerprints = {}
    for index, entry in enumerate(order):
        name = f"{index:02d}-{entry['arm']}-{entry['seed']}-{entry['repeat']}"
        executable = args.unsloth_python if entry["arm"] == "unsloth" else sys.executable
        command = [executable, "-m", "benchmarks.qualified_path_measure", "--config", args.config,
                   "--output", str(args.output / name), "--arm", entry["arm"], "--seed", str(entry["seed"]),
                   "--steps", str(args.steps), "--validation-batches", str(args.validation_batches),
                   "--startup-max-c", str(args.startup_max_c), "--startup-dwell-seconds",
                   str(args.startup_dwell_seconds), "--emergency-guard-c",
                   str(args.emergency_guard_c)]
        if args.qualified_micro_paced and entry["arm"] == "qualified":
            command.append("--micro-paced")
        print(f"Starting {index+1}/{len(order)}: {name}", flush=True)
        with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            print(f"STOP: child failed; inspect {name}.log", flush=True)
            return completed.returncode
        result = json.loads((args.output / name / "metrics.json").read_text(encoding="utf-8"))
        expected = fingerprints.setdefault(entry["seed"], result["initial_adapter_sha256"])
        if result["initial_adapter_sha256"] != expected:
            raise RuntimeError(f"Initial adapter mismatch in {name}; comparison rejected")
        state = result.get("machine_state", {})
        pre_run = state.get("pre_run_gpu", {})
        if state.get("ac_power") is not True:
            raise RuntimeError(f"AC power was not verified in {name}; comparison rejected")
        if pre_run.get("mean_utilization_percent") is None:
            raise RuntimeError(f"Pre-run GPU utilization was not measured in {name}")
        if pre_run["mean_utilization_percent"] > 5.0:
            raise RuntimeError(
                f"Pre-run GPU utilization exceeded 5% in {name}; comparison rejected"
            )
        print(f"Completed {name}: compute {result['compute_tok_s']:.1f}; "
              f"end-to-end {result['end_to_end_tok_s']:.1f}; "
              f"allocated {result['peak_allocated_bytes']/2**30:.3f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
