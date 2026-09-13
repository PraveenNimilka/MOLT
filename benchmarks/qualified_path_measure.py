"""Isolated fixed-token Qwen comparison. Launch each arm in a fresh process."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("hf", "reference", "qualified", "unsloth"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-capture-memory",
        action="store_true",
        help="Record allocator stacks and the live allocation set at capture peak",
    )
    parser.add_argument(
        "--attribute-updates",
        action="store_true",
        help="Low-overhead per-update CUDA/host/estimated-board-energy attribution",
    )
    parser.add_argument("--separate-graphs", action="store_true",
                        help="Capture forward and backward separately with one shared pool")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument(
        "--trim-after-graph-capture",
        action="store_true",
        help="Release inactive default-pool cache after graph instantiation",
    )
    parser.add_argument("--split-loss", action="store_true", help="Unpromoted split-loss experiment")
    parser.add_argument(
        "--asymmetric-gated-replay",
        action="store_true",
        help="Experimental one-branch gated-MLP activation rematerialization",
    )
    parser.add_argument(
        "--pointwise-gated-replay",
        action="store_true",
        help="Experimental exact gated activation/down lifetime fusion",
    )
    parser.add_argument(
        "--gated-replay-blocks",
        type=int,
        help="Replay this many compatible up projections; zero is EGLF",
    )
    parser.add_argument(
        "--gated-offload-blocks",
        type=int,
        help="Offload this many compatible up activations to pinned host slots",
    )
    parser.add_argument(
        "--gated-gate-offload-blocks",
        type=int,
        help="Offload this many compatible gate activations to pinned host slots",
    )
    parser.add_argument(
        "--loss-chunk-size-override",
        type=int,
        help="Experimental frozen-loss chunk size; requires --split-loss",
    )
    parser.add_argument(
        "--cache-scheduled-backward-weights",
        action="store_true",
        help="Experimental dense BF16 backward cache for qualified scheduled projections",
    )
    parser.add_argument("--paced", action="store_true", help="Closed-loop between-update pacing, 70C/70W targets")
    parser.add_argument("--micro-paced", action="store_true", help="Bounded 78-82C micro-pause governor")
    parser.add_argument(
        "--pace-validation",
        action="store_true",
        help="Apply the same micro-pause controller between validation batches",
    )
    parser.add_argument("--emergency-guard-c", type=float, default=83.0,
                        help="Identical pre-update emergency guard for every comparison arm")
    parser.add_argument(
        "--heat-soak-guard-c",
        type=float,
        help="Optional stricter guard during the initial training heat-soak window",
    )
    parser.add_argument(
        "--heat-soak-seconds",
        type=float,
        help="Training seconds before switching from heat-soak to steady guard",
    )
    parser.add_argument("--startup-max-c", type=float, default=50.0)
    parser.add_argument("--startup-dwell-seconds", type=float, default=20.0)
    parser.add_argument("--pretraining-max-c", type=float)
    parser.add_argument("--pretraining-dwell-seconds", type=float, default=5.0)
    parser.add_argument("--electricity-usd-per-kwh", type=float, default=0.20)
    args = parser.parse_args()
    if args.pace_validation and not args.micro_paced:
        parser.error("validation pacing requires --micro-paced")
    if args.split_loss and args.arm not in ("reference", "qualified"):
        parser.error("split loss requires a MOLT arm")
    if args.asymmetric_gated_replay and args.arm != "qualified":
        parser.error("asymmetric gated replay requires the qualified arm")
    if args.pointwise_gated_replay and args.arm != "qualified":
        parser.error("pointwise gated replay requires the qualified arm")
    if args.asymmetric_gated_replay and args.pointwise_gated_replay:
        parser.error("select only one gated replay mode")
    if args.gated_replay_blocks is not None and args.arm != "qualified":
        parser.error("budgeted gated replay requires the qualified arm")
    if args.gated_offload_blocks is not None and args.arm != "qualified":
        parser.error("gated activation offload requires the qualified arm")
    if args.gated_gate_offload_blocks is not None and args.arm != "qualified":
        parser.error("gate activation offload requires the qualified arm")
    if args.gated_replay_blocks is not None and args.gated_replay_blocks < 0:
        parser.error("gated replay block count cannot be negative")
    if args.gated_offload_blocks is not None and args.gated_offload_blocks < 0:
        parser.error("gated offload block count cannot be negative")
    if args.gated_gate_offload_blocks is not None and args.gated_gate_offload_blocks < 0:
        parser.error("gate offload block count cannot be negative")
    if args.gated_replay_blocks is not None and args.gated_offload_blocks is not None:
        parser.error("gated replay and gated offload are mutually exclusive")
    if args.gated_replay_blocks is not None and (
        args.asymmetric_gated_replay or args.pointwise_gated_replay
    ):
        parser.error("budgeted gated replay selects its own gated replay mode")
    if args.loss_chunk_size_override is not None and not args.split_loss:
        parser.error("loss chunk override requires --split-loss")
    if args.loss_chunk_size_override is not None and args.loss_chunk_size_override < 1:
        parser.error("loss chunk override must be positive")
    if args.cache_scheduled_backward_weights and args.arm != "qualified":
        parser.error("scheduled backward cache requires the qualified arm")
    if args.paced and args.micro_paced:
        parser.error("select only one pacing experiment")
    if args.pace_validation and not args.micro_paced:
        parser.error("validation pacing requires micro pacing")
    if args.no_graphs and args.separate_graphs:
        parser.error("separate graphs cannot be selected when graphs are disabled")
    if args.attribute_updates and (
        args.arm not in ("reference", "qualified")
        or args.no_graphs
        or args.separate_graphs
    ):
        parser.error("update attribution requires a MOLT joint-graph path")
    if args.attribute_updates and args.profile:
        parser.error("low-overhead update attribution cannot be combined with profiler tracing")
    if args.electricity_usd_per_kwh < 0:
        parser.error("electricity price cannot be negative")
    if not 0 < args.emergency_guard_c < 84:
        parser.error("emergency guard must be below the fixed 84C abort boundary")
    if (args.heat_soak_guard_c is None) != (args.heat_soak_seconds is None):
        parser.error("heat-soak guard and duration must be supplied together")
    if args.heat_soak_guard_c is not None and not (
        0 < args.heat_soak_guard_c <= args.emergency_guard_c
        and args.heat_soak_seconds > 0
    ):
        parser.error("heat-soak guard must be at most the steady guard with positive duration")
    if not 0 < args.startup_max_c < args.emergency_guard_c:
        parser.error("startup maximum must be below the emergency guard")
    if args.startup_dwell_seconds < 0:
        parser.error("startup dwell cannot be negative")
    if args.pretraining_max_c is not None and not 0 < args.pretraining_max_c < 84:
        parser.error("pretraining maximum must be below the abort boundary")
    if args.pretraining_dwell_seconds < 0:
        parser.error("pretraining dwell cannot be negative")
    if min(args.steps, args.validation_batches) < 1:
        parser.error("positive steps and validation batches required")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.arm == "unsloth":
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
        from unsloth import FastLanguageModel
    from contextlib import nullcontext
    from dataclasses import replace

    import torch

    from molt_stream.core.specs import load_spec
    from molt_stream.data.bytes import MMapTokenBatcher
    from molt_stream.measurement.telemetry import NVMLTelemetry
    from molt_stream.measurement.thermal import HeatSoakGuardEnvelope
    from molt_stream.measurement.update_regions import (
        CapturedCudaRegionTimer,
        attribute_update_energy,
    )
    from molt_stream.methods.nf4_lora import enable_scheduled_nf4_lora
    from molt_stream.training.qlora import _build_qlora_model, _shifted_causal_loss
    from molt_stream.training.static_cuda_graph import StaticCudaMicrobatch
    if args.split_loss:
        from molt_stream.kernels.split_frozen_loss import split_frozen_head_forward

        kernel_module = importlib.import_module("molt_stream.kernels.frozen_linear_cross_entropy")
        kernel_module._triton_frozen_head_forward = split_frozen_head_forward

    spec = replace(load_spec(args.config), seed=args.seed)
    if spec.batch_size != 1 or spec.gradient_accumulation != 1 or spec.data.context_length != 256:
        raise ValueError("This registered comparison requires batch/accumulation/context=1/1/256")
    def write(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    def digest(path):
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"status": "running", "arm": args.arm, "seed": args.seed,
              "steps": args.steps, "validation_batches": args.validation_batches,
              "targets_per_update": 256, "prediction_targets": args.steps * 256,
              "instrumented": args.profile or args.attribute_updates or args.profile_capture_memory,
              "instrumentation_mode": (
                  "torch-profiler" if args.profile else
                  "capture-memory-history" if args.profile_capture_memory else
                  "cuda-events-plus-nvml" if args.attribute_updates else None
              ),
              "energy_target_joules_per_target": 0.0511,
              "promoted": False,
              "experimental_split_loss": args.split_loss,
              "experimental_asymmetric_gated_replay": args.asymmetric_gated_replay,
              "experimental_pointwise_gated_replay": args.pointwise_gated_replay,
              "experimental_gated_replay_blocks": args.gated_replay_blocks,
              "experimental_gated_offload_blocks": args.gated_offload_blocks,
              "experimental_gated_gate_offload_blocks": args.gated_gate_offload_blocks,
              "experimental_loss_chunk_size": args.loss_chunk_size_override,
              "experimental_scheduled_backward_cache": args.cache_scheduled_backward_weights,
              "cuda_graph_topology": (
                  "disabled" if args.no_graphs else
                  "separate-shared-pool" if args.separate_graphs else "joint"
              ),
              "trim_after_graph_capture": args.trim_after_graph_capture,
              "software_pacing": "duty-cycle" if args.paced else "micro-guard" if args.micro_paced else None,
              "validation_pacing": args.pace_validation,
              "source_sha256": digest(__file__),
              "dataset_sha256": digest(spec.data.path),
              "validation_sha256": digest(spec.data.validation_path),
              "model_config_sha256": digest(Path(spec.base_model) / "config.json"),
              "versions": {}, "phases": {},
              "thermal_policy": {"startup_c": args.startup_max_c,
                                 "dwell_seconds": args.startup_dwell_seconds,
                                  "pretraining_c": args.pretraining_max_c,
                                  "pretraining_dwell_seconds": args.pretraining_dwell_seconds,
                                  "heat_soak_guard_c": args.heat_soak_guard_c,
                                  "heat_soak_seconds": args.heat_soak_seconds,
                                  "guard_c": args.emergency_guard_c,
                                  "abort_c": 84},
              "limitations": ["uncontrolled GPU clocks", "single GPU", "board energy, not wall-meter energy",
                              "process import time excluded from measured session; setup and graph capture included"]}
    implementation_files = {
        "harness": Path(__file__),
        "qlora": Path(__file__).parents[1] / "src/molt_stream/training/qlora.py",
        "static_graph": Path(__file__).parents[1] / "src/molt_stream/training/static_cuda_graph.py",
        "nf4_lora": Path(__file__).parents[1] / "src/molt_stream/methods/nf4_lora.py",
        "frozen_loss": Path(__file__).parents[1] / "src/molt_stream/kernels/frozen_linear_cross_entropy.py",
        "split_loss": Path(__file__).parents[1] / "src/molt_stream/kernels/split_frozen_loss.py",
        "thermal": Path(__file__).parents[1] / "src/molt_stream/measurement/thermal.py",
    }
    result["implementation_sha256"] = {
        name: digest(path) for name, path in implementation_files.items()
    }
    import psutil

    battery = psutil.sensors_battery()
    try:
        power_scheme = subprocess.run(
            ["powercfg", "/getactivescheme"], capture_output=True, check=False,
            text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        power_scheme = None
    result["machine_state"] = {
        "ac_power": None if battery is None else battery.power_plugged,
        "battery_percent": None if battery is None else battery.percent,
        "windows_power_scheme": power_scheme,
        "ambient_temperature_c": None,
    }
    for package in ("torch", "transformers", "peft", "bitsandbytes", "triton-windows", "unsloth", "cut-cross-entropy"):
        try:
            result["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    write("metrics.json", result)
    telemetry = NVMLTelemetry(0.1)
    telemetry.start()
    session_start = time.perf_counter()
    paused = 0.0
    cooling_halts = 0
    pacing_seconds = 0.0
    validation_pacing_seconds = 0.0
    attributed_updates = []
    result["memory_milestones"] = {}
    heat_soak_envelope = (
        HeatSoakGuardEnvelope(
            heat_soak_c=args.heat_soak_guard_c,
            steady_c=args.emergency_guard_c,
            heat_soak_seconds=args.heat_soak_seconds,
            abort_c=84.0,
        )
        if args.heat_soak_guard_c is not None
        else None
    )

    def record_memory(name):
        result["memory_milestones"][name] = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    if args.paced:
        from molt_stream.measurement.closed_loop_pacing import ClosedLoopPacer

        pacer = ClosedLoopPacer(target_c=70, target_watts=70)
        result["pacing_targets"] = {"gpu_c": 70, "steady_max_c": 72, "watts": 70,
            "session_tok_s": 1500, "hardware_cap_applied": False}
    elif args.micro_paced:
        from molt_stream.measurement.thermal import MicroPauseThermalController

        micro_pacer = MicroPauseThermalController(start_c=78, protective_c=82,
            abort_c=84, low_pause_seconds=.001, middle_pause_seconds=.0065,
            high_pause_seconds=.012, protective_pause_seconds=.100,
            release_c=80, release_samples=3, power_control_start_c=65,
            target_average_watts=48, idle_watts=12)
        result["pacing_targets"] = {"start_c": 78, "protective_c": 82, "abort_c": 84,
            "power_control_start_c": 65, "target_average_watts": 48,
            "idle_watts_assumption": 12, "maximum_pause_ms": 100,
            "session_tok_s": 1181, "hardware_cap_applied": False}
        result["micro_pacing_decisions"] = {}
    def guard(startup=False, *, maximum_c=None, dwell_seconds=None, target_c=None):
        nonlocal paused, cooling_halts
        start = time.perf_counter()
        dwell = None
        slept = False
        while True:
            point = telemetry.thermal_point()
            if point is None or point.gpu_temperature_c is None:
                raise RuntimeError("GPU temperature unavailable")
            temperature = point.gpu_temperature_c
            if temperature >= 84:
                raise RuntimeError(f"Thermal abort: {temperature} C")
            if startup:
                threshold = args.startup_max_c if maximum_c is None else maximum_c
                required_dwell = (
                    args.startup_dwell_seconds if dwell_seconds is None else dwell_seconds
                )
                dwell = (dwell or time.perf_counter()) if temperature <= threshold else None
                if dwell is not None and time.perf_counter() - dwell >= required_dwell:
                    break
            elif temperature < (
                args.emergency_guard_c if target_c is None else target_c
            ):
                break
            if time.perf_counter() - start > 300:
                raise RuntimeError("Thermal wait timed out")
            slept = True
            time.sleep(0.1 if startup else 0.04)
        if slept:
            paused += time.perf_counter() - start
            cooling_halts += int(not startup)
    def phase(name, operation):
        started = time.perf_counter()
        value = operation()
        torch.cuda.synchronize()
        result["phases"][name] = {"start": started, "end": time.perf_counter()}
        result["phases"][name]["seconds"] = result["phases"][name]["end"] - started
        return value

    try:
        phase("startup_cooling", lambda: guard(True))
        idle_points = telemetry.points[-min(len(telemetry.points), 50):]
        result["machine_state"]["pre_setup_gpu"] = {
            "temperature_c": idle_points[-1].gpu_temperature_c if idle_points else None,
            "mean_utilization_percent": (
                sum(p.gpu_utilization_percent for p in idle_points if p.gpu_utilization_percent is not None)
                / sum(p.gpu_utilization_percent is not None for p in idle_points)
                if any(p.gpu_utilization_percent is not None for p in idle_points) else None
            ),
            "mean_power_watts": (
                sum(p.gpu_power_watts for p in idle_points if p.gpu_power_watts is not None)
                / sum(p.gpu_power_watts is not None for p in idle_points)
                if any(p.gpu_power_watts is not None for p in idle_points) else None
            ),
        }
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        def setup():
            if args.arm == "unsloth":
                model, _ = FastLanguageModel.from_pretrained(
                    model_name=spec.base_model, max_seq_length=256, dtype=torch.bfloat16,
                    load_in_4bit=True, fix_tokenizer=False, trust_remote_code=False,
                    use_gradient_checkpointing=False, random_state=args.seed)
                model = FastLanguageModel.get_peft_model(model, r=8, lora_alpha=16,
                    lora_dropout=0, bias="none", target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
                    use_gradient_checkpointing=False, random_state=args.seed, max_seq_length=256)
                FastLanguageModel.for_training(model, use_gradient_checkpointing=False)
            else:
                selected = spec
                if args.arm == "hf":
                    selected = replace(spec, qlora_frozen_rmsnorm_bf16=False,
                        qlora_restore_tied_embedding_bf16=False, qlora_cache_frozen_head=False)
                model = _build_qlora_model(selected)
                if args.arm == "qualified":
                    result["patched_modules"] = enable_scheduled_nf4_lora(
                        model,
                        module_suffixes=("down_proj", "o_proj"),
                        cache_backward_weights=args.cache_scheduled_backward_weights,
                    )
                    if (
                        args.asymmetric_gated_replay
                        or args.pointwise_gated_replay
                        or args.gated_replay_blocks is not None
                        or args.gated_offload_blocks is not None
                        or args.gated_gate_offload_blocks is not None
                    ):
                        from molt_stream.methods.gated_replay import (
                            enable_asymmetric_gated_replay,
                        )

                        result["patched_gated_mlp_blocks"] = (
                            enable_asymmetric_gated_replay(
                                model,
                                rematerialize_up=(
                                    args.asymmetric_gated_replay
                                    or args.gated_replay_blocks is not None
                                ),
                                rematerialize_blocks=args.gated_replay_blocks,
                                offload_blocks=args.gated_offload_blocks or 0,
                                offload_gate_blocks=args.gated_gate_offload_blocks or 0,
                            )
                        )
            model.train()
            if any(bool(getattr(module, "gradient_checkpointing", False)) for module in model.modules()):
                raise RuntimeError("Unexpected activation checkpointing")
            return model
        model = phase("model_setup", setup)
        record_memory("model_setup")
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        result["trainable_parameters"] = sum(p.numel() for _, p in named)
        if result["trainable_parameters"] != 9232384:
            raise RuntimeError("Adapter geometry mismatch")
        # Library versions consume random numbers differently during setup.
        # Define one sorted-name CPU initialization independently of loading.
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        with torch.no_grad():
            for name, parameter in sorted(named):
                if parameter.dtype != torch.float32:
                    raise RuntimeError("Expected FP32 adapter masters")
                canonical = torch.empty(parameter.shape, dtype=torch.float32)
                if ".lora_A." in name:
                    bound = parameter.shape[1] ** -0.5
                    canonical.uniform_(-bound, bound, generator=generator)
                elif ".lora_B." in name:
                    canonical.zero_()
                else:
                    raise RuntimeError(f"Unexpected trainable parameter: {name}")
                parameter.copy_(canonical)
        result["adapter_initialization"] = "sorted-name CPU uniform(-1/sqrt(fan_in),+1/sqrt(fan_in)) A; zero B"
        fingerprint = hashlib.sha256()
        for name, p in sorted(named):
            fingerprint.update(name.encode() + b"\0" + p.detach().float().cpu().numpy().tobytes())
        result["initial_adapter_sha256"] = fingerprint.hexdigest()
        optimizer = torch.optim.AdamW([p for _, p in named], lr=0.0002,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, fused=True)
        record_memory("optimizer_created_before_lazy_state")
        batcher = MMapTokenBatcher(spec.data, seed=args.seed, device="cuda")
        validation = MMapTokenBatcher(replace(spec.data, path=spec.data.validation_path),
                                     seed=args.seed, device="cuda")
        validation_state = validation.state_dict()
        if args.arm == "unsloth":
            from cut_cross_entropy import linear_cross_entropy
            from xformers.ops.fmha.attn_bias import LowerTriangularMask
            base = model.get_base_model()
            head = base.get_output_embeddings().weight
            mask = LowerTriangularMask()
        region_timer = (
            CapturedCudaRegionTimer(
                ("graph_start", "transformer_end", "forward_end", "graph_end")
            )
            if args.attribute_updates
            else None
        )
        def loss_builder(x, y):
            if args.arm == "unsloth":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = base.model(input_ids=x, causal_mask=mask, use_cache=False,
                                        return_dict=True).last_hidden_state
                    return linear_cross_entropy(hidden, head, y, shift=False,
                                                reduction="mean", filter_eps=None)
            return _shifted_causal_loss(model, x, y, autocast=True,
                chunk_size=(
                    None if args.arm == "hf" else
                    args.loss_chunk_size_override or spec.qlora_loss_chunk_size
                ),
                precompute_head_gradient=args.arm != "hf", loss_backend=spec.qlora_loss_backend,
                sdpa_kernel_name=spec.qlora_sdpa_kernel,
                cuda_region_marker=region_timer.record if region_timer is not None else None)
        @torch.no_grad()
        def evaluate():
            nonlocal validation_pacing_seconds
            model.eval()
            validation.load_state_dict(validation_state)
            total = 0.0
            try:
                for _ in range(args.validation_batches):
                    guard()
                    batch_started = time.perf_counter()
                    total += float(loss_builder(*validation.batch(1)))
                    batch_seconds = time.perf_counter() - batch_started
                    if args.pace_validation:
                        samples = telemetry.points
                        latest = samples[-1] if samples else None
                        now = time.perf_counter()
                        if (
                            latest is None
                            or now - latest.monotonic_seconds > 1
                            or latest.gpu_power_watts is None
                        ):
                            raise RuntimeError("Missing or stale validation pacing telemetry")
                        decision = micro_pacer.update(latest, step_seconds=batch_seconds)
                        if decision.abort:
                            raise RuntimeError("Validation micro-pacing thermal abort")
                        counts = result.setdefault("validation_pacing_decisions", {})
                        counts[decision.phase] = counts.get(decision.phase, 0) + 1
                        if decision.pause_seconds:
                            before_sleep = time.perf_counter()
                            time.sleep(decision.pause_seconds)
                            validation_pacing_seconds += time.perf_counter() - before_sleep
            finally:
                model.train()
            return total / args.validation_batches
        result["initial_validation_nll"] = phase("initial_validation", evaluate)
        record_memory("initial_validation")
        runner = None
        if args.arm in ("reference", "qualified") and not args.no_graphs:
            example = validation.batch(1)
            guard()
            if args.profile_capture_memory:
                torch.cuda.memory._record_memory_history(
                    enabled="all", context="alloc", stacks="python", max_entries=100_000
                )
            try:
                runner = phase("graph_capture", lambda: StaticCudaMicrobatch(loss_builder, example,
                    tuple(p for _, p in named), joint_forward_backward=not args.separate_graphs,
                    region_timer=region_timer,
                    trim_unused_default_pool=args.trim_after_graph_capture))
                if args.profile_capture_memory:
                    from molt_stream.training.static_cuda_graph import (
                        cuda_allocation_peak_ledger,
                    )

                    snapshot = torch.cuda.memory._snapshot()
                    result["capture_allocation_peak_ledger"] = (
                        cuda_allocation_peak_ledger(snapshot)
                    )
                    torch.cuda.memory._dump_snapshot(
                        str(args.output / "capture-memory.pickle")
                    )
            finally:
                if args.profile_capture_memory:
                    torch.cuda.memory._record_memory_history(enabled=None)
            record_memory("graph_capture")
        if args.pretraining_max_c is not None:
            phase(
                "pretraining_cooling",
                lambda: guard(
                    True,
                    maximum_c=args.pretraining_max_c,
                    dwell_seconds=args.pretraining_dwell_seconds,
                ),
            )
        if args.profile:
            from molt_stream.training.static_cuda_graph import (
                summarize_cuda_memory_pools,
            )

            result["memory_pools_after_capture"] = summarize_cuda_memory_pools()
        updates = []
        losses = []
        intervals = []
        event_pairs = (
            {
                name: (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for name in ("dataset_transfer", "graph_input_staging", "optimizer_update")
            }
            if args.attribute_updates
            else {}
        )
        def begin_cuda_region(name):
            if args.attribute_updates:
                event_pairs[name][0].record()
        def end_cuda_region(name):
            if args.attribute_updates:
                event_pairs[name][1].record()
        def cuda_region_seconds(name):
            return event_pairs[name][0].elapsed_time(event_pairs[name][1]) / 1000.0
        profiler_context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) if args.profile else nullcontext()
        def train():
            nonlocal pacing_seconds
            training_started = time.perf_counter()
            result["_training_initial_cooling_halts"] = cooling_halts
            result["_training_initial_cooling_seconds"] = paused
            result["training_cooling_halts"] = 0
            result["training_cooling_seconds"] = 0.0
            with profiler_context as profiler:
                for step in range(args.steps):
                    interval_start = time.perf_counter()
                    telemetry_seconds = 0.0
                    thermal_seconds = 0.0
                    synchronization_seconds = 0.0
                    before_pause = paused
                    host_started = time.perf_counter()
                    guard_target_c = (
                        heat_soak_envelope.target_c(
                            time.perf_counter() - training_started
                        )
                        if heat_soak_envelope is not None
                        else None
                    )
                    guard(target_c=guard_target_c)
                    guard_seconds = time.perf_counter() - host_started
                    guard_pause = paused - before_pause
                    thermal_seconds += guard_pause
                    telemetry_seconds += max(0.0, guard_seconds - guard_pause)
                    begin_cuda_region("dataset_transfer")
                    host_started = time.perf_counter()
                    x, y = batcher.batch(1)
                    dataset_host_seconds = time.perf_counter() - host_started
                    end_cuda_region("dataset_transfer")
                    host_started = time.perf_counter()
                    torch.cuda.synchronize()
                    synchronization_seconds += time.perf_counter() - host_started
                    started = time.perf_counter()
                    if runner is None:
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_builder(x, y)
                        loss.backward()
                    else:
                        runner.zero_grad()
                        if args.attribute_updates:
                            begin_cuda_region("graph_input_staging")
                            host_started = time.perf_counter()
                            runner.stage_inputs(x, y)
                            staging_host_seconds = time.perf_counter() - host_started
                            end_cuda_region("graph_input_staging")
                            loss = runner.replay_staged()
                        else:
                            loss = runner.replay(x, y)
                    begin_cuda_region("optimizer_update")
                    host_started = time.perf_counter()
                    optimizer.step()
                    optimizer_host_seconds = time.perf_counter() - host_started
                    end_cuda_region("optimizer_update")
                    host_started = time.perf_counter()
                    torch.cuda.synchronize()
                    synchronization_seconds += time.perf_counter() - host_started
                    updates.append(time.perf_counter() - started)
                    value = float(loss.detach())
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss")
                    losses.append(value)
                    before_pause = paused
                    host_started = time.perf_counter()
                    guard_target_c = (
                        heat_soak_envelope.target_c(
                            time.perf_counter() - training_started
                        )
                        if heat_soak_envelope is not None
                        else None
                    )
                    guard(target_c=guard_target_c)
                    guard_seconds = time.perf_counter() - host_started
                    guard_pause = paused - before_pause
                    thermal_seconds += guard_pause
                    telemetry_seconds += max(0.0, guard_seconds - guard_pause)
                    if args.paced or args.micro_paced:
                        host_started = time.perf_counter()
                        point = telemetry.thermal_point()
                        telemetry_seconds += time.perf_counter() - host_started
                        if point is None or point.gpu_temperature_c is None:
                            raise RuntimeError("Missing pacing temperature")
                        now = time.perf_counter()
                        samples = telemetry.points
                        latest = samples[-1] if samples else None
                        if latest is None or now - latest.monotonic_seconds > 1 or latest.gpu_power_watts is None:
                            raise RuntimeError("Missing or stale pacing telemetry")
                        if args.micro_paced:
                            decision = micro_pacer.update(latest, step_seconds=updates[-1])
                            if decision.abort:
                                raise RuntimeError("Micro-pacing thermal abort")
                            delay = decision.pause_seconds
                            counts = result["micro_pacing_decisions"]
                            counts[decision.phase] = counts.get(decision.phase, 0) + 1
                        else:
                            delay = pacer.delay(now, point.gpu_temperature_c,
                                latest.gpu_power_watts, updates[-1], now - interval_start)
                        if delay:
                            before_sleep = time.perf_counter()
                            time.sleep(delay)
                            slept = time.perf_counter() - before_sleep
                            pacing_seconds += slept
                            thermal_seconds += slept
                    interval_end = time.perf_counter()
                    intervals.append({"start": interval_start, "end": interval_end,
                                      "targets": 256})
                    if args.attribute_updates:
                        transformer_forward_seconds = region_timer.elapsed_seconds(
                            "graph_start", "transformer_end"
                        )
                        frozen_loss_seconds = region_timer.elapsed_seconds(
                            "transformer_end", "forward_end"
                        )
                        backward_seconds = region_timer.elapsed_seconds(
                            "forward_end", "graph_end"
                        )
                        dataset_cuda_seconds = cuda_region_seconds("dataset_transfer")
                        staging_cuda_seconds = cuda_region_seconds("graph_input_staging")
                        optimizer_cuda_seconds = cuda_region_seconds("optimizer_update")
                        accounted_host_seconds = (
                            dataset_host_seconds
                            + staging_host_seconds
                            + optimizer_host_seconds
                            + synchronization_seconds
                            + telemetry_seconds
                            + thermal_seconds
                        )
                        attributed_updates.append({
                            "step": step + 1,
                            "start": interval_start,
                            "end": interval_end,
                            "targets": 256,
                            "loss": value,
                            "graph_replay_total_cuda_seconds": (
                                transformer_forward_seconds
                                + frozen_loss_seconds
                                + backward_seconds
                            ),
                            "transformer_forward_cuda_seconds": transformer_forward_seconds,
                            "transformer_backward_cuda_seconds": backward_seconds,
                            "regions": {
                                "dataset_transfer": {
                                    "host_seconds": dataset_host_seconds,
                                    "cuda_seconds": dataset_cuda_seconds,
                                },
                                "graph_input_staging": {
                                    "host_seconds": staging_host_seconds,
                                    "cuda_seconds": staging_cuda_seconds,
                                },
                                "transformer_graph_replay": {
                                    "host_seconds": 0.0,
                                    "cuda_seconds": (
                                        transformer_forward_seconds + backward_seconds
                                    ),
                                },
                                "frozen_vocabulary_loss": {
                                    "host_seconds": 0.0,
                                    "cuda_seconds": frozen_loss_seconds,
                                },
                                "optimizer_update": {
                                    "host_seconds": optimizer_host_seconds,
                                    "cuda_seconds": optimizer_cuda_seconds,
                                },
                                "cuda_synchronization": {
                                    "host_seconds": synchronization_seconds,
                                    "cuda_seconds": None,
                                },
                                "telemetry": {
                                    "host_seconds": telemetry_seconds,
                                    "cuda_seconds": None,
                                },
                                "thermal_pacing": {
                                    "host_seconds": thermal_seconds,
                                    "cuda_seconds": None,
                                },
                                "python_dispatch": {
                                    "host_seconds": max(
                                        0.0,
                                        interval_end
                                        - interval_start
                                        - accounted_host_seconds,
                                    ),
                                    "cuda_seconds": None,
                                },
                            },
                        })
                    if (step + 1) % 256 == 0 or step + 1 == args.steps:
                        print(f"{args.arm} seed={args.seed} {step+1}/{args.steps} loss={value:.6f}", flush=True)
            if profiler is not None:
                profiler.export_chrome_trace(str(args.output / "trace.json"))
            result["training_cooling_halts"] = (
                cooling_halts - result["_training_initial_cooling_halts"]
            )
            result["training_cooling_seconds"] = (
                paused - result["_training_initial_cooling_seconds"]
            )
        phase("training", train)
        record_memory("training_complete")
        if runner is not None:
            runner.close()
            runner = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        result["final_validation_nll"] = phase("final_validation", evaluate)
        result["validation_pacing_seconds"] = validation_pacing_seconds
        phase("save", lambda: model.save_pretrained(args.output / "adapter", safe_serialization=True))
        result.update(status="completed", update_compute_seconds=sum(updates),
            compute_tok_s=args.steps * 256 / sum(updates),
            training_tok_s=args.steps * 256 / result["phases"]["training"]["seconds"],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        training_seconds = result["phases"]["training"]["seconds"]
        compute_seconds = result["update_compute_seconds"]
        cooling_seconds = result["training_cooling_seconds"]
        result["training_time_attribution"] = {
            "compute_seconds": compute_seconds,
            "scheduled_pacing_seconds": pacing_seconds,
            "emergency_cooling_seconds": cooling_seconds,
            "other_host_and_staging_seconds": max(
                0.0, training_seconds - compute_seconds - pacing_seconds - cooling_seconds
            ),
            "compute_to_training_throughput_conversion": (
                result["training_tok_s"] / result["compute_tok_s"]
            ),
            "pacing_share_of_compute_to_training_gap": (
                pacing_seconds / (training_seconds - compute_seconds)
                if training_seconds > compute_seconds else None
            ),
        }
        write("updates.json", {"seconds": updates, "losses": losses})
        write("intervals.json", intervals)
        if intervals:
            cutoff = intervals[-1]["end"] - 60
            tail = [entry for entry in intervals if entry["start"] >= cutoff]
            duration = tail[-1]["end"] - tail[0]["start"]
            points = [p for p in telemetry.points if tail[0]["start"] <= p.monotonic_seconds <= tail[-1]["end"]]
            temperatures = [p.gpu_temperature_c for p in points if p.gpu_temperature_c is not None]
            result["tail_training_window"] = {"seconds": duration,
                "tok_s": len(tail) * 256 / duration,
                "maximum_c": max(temperatures) if temperatures else None,
                "mean_c": sum(temperatures) / len(temperatures) if temperatures else None,
                "scope": "last approximately 60s; observed window, not proof of thermal equilibrium"}
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        raise
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        raise
    finally:
        if "_training_initial_cooling_halts" in result:
            initial_halts = result.pop("_training_initial_cooling_halts")
            initial_seconds = result.pop("_training_initial_cooling_seconds")
            result["training_cooling_halts"] = cooling_halts - initial_halts
            result["training_cooling_seconds"] = paused - initial_seconds
        result["session_seconds"] = time.perf_counter() - session_start
        result["end_to_end_tok_s"] = args.steps * 256 / result["session_seconds"] if result["status"] == "completed" else None
        result["thermal_pause_seconds"] = paused
        result["pacing_seconds"] = pacing_seconds
        result["cooling_halts"] = cooling_halts
        summary = telemetry.stop()
        points = summary.pop("points")
        write("telemetry.json", points)
        if args.attribute_updates and attributed_updates:
            idle_board_watts = result["machine_state"]["pre_setup_gpu"]["mean_power_watts"]
            if idle_board_watts is None:
                raise RuntimeError("Idle board power unavailable for update attribution")
            enriched_updates = []
            for update in attributed_updates:
                energy = attribute_update_energy(
                    update, points, idle_board_watts=float(idle_board_watts)
                )
                enriched_updates.append({
                    key: value for key, value in update.items() if key != "regions"
                } | energy)
            region_totals = {}
            for update in enriched_updates:
                for name, values in update["regions"].items():
                    total = region_totals.setdefault(name, {
                        "host_seconds": 0.0,
                        "cuda_seconds": 0.0,
                        "estimated_board_joules": 0.0,
                    })
                    total["host_seconds"] += float(values.get("host_seconds") or 0.0)
                    total["cuda_seconds"] += float(values.get("cuda_seconds") or 0.0)
                    total["estimated_board_joules"] += float(
                        values.get("estimated_board_joules") or 0.0
                    )
            attributed_targets = sum(update["targets"] for update in enriched_updates)
            for values in region_totals.values():
                values["estimated_joules_per_target"] = (
                    values["estimated_board_joules"] / attributed_targets
                )
            measured_update_joules = sum(
                float(update["measured_board_joules"])
                for update in enriched_updates
                if update["measured_board_joules"] is not None
            )
            result["update_attribution"] = {
                "updates": len(enriched_updates),
                "targets": attributed_targets,
                "measured_update_board_joules": measured_update_joules,
                "measured_update_joules_per_target": (
                    measured_update_joules / attributed_targets
                ),
                "idle_board_watts_assumption": idle_board_watts,
                "regions": region_totals,
                "method": (
                    "Reusable CUDA events, including external event nodes inside the "
                    "captured graph. NVML update-window joules are partitioned between "
                    "CUDA regions by device time and host-idle regions by measured host "
                    "time under the pre-setup idle-power estimate. Synchronization is "
                    "reported as an inclusive wait with zero additional joules."
                ),
            }
            write("update_attribution.json", {
                "summary": result["update_attribution"],
                "updates": enriched_updates,
            })
        result["telemetry"] = summary
        joules = summary["gpu_board_energy_joules"]
        result["energy_wh"] = joules / 3600 if joules is not None else None
        result["electricity_price_usd_per_kwh"] = args.electricity_usd_per_kwh
        result["estimated_board_energy_cost_usd"] = (
            result["energy_wh"] / 1000 * args.electricity_usd_per_kwh
            if result["energy_wh"] is not None else None
        )
        result["projected_board_energy_cost_per_million_targets_usd"] = (
            result["estimated_board_energy_cost_usd"] * 1_000_000 / result["prediction_targets"]
            if result["estimated_board_energy_cost_usd"] is not None else None
        )
        if (args.paced or args.micro_paced) and result["status"] == "completed":
            tail = result.get("tail_training_window", {})
            checks = {
                "session_target_met": result["end_to_end_tok_s"] >= (1181 if args.micro_paced else 1500),
                "tail_window_at_least_50_seconds": tail.get("seconds", 0) >= 50,
                "tail_temperature_target_met": tail.get("maximum_c") is not None and tail["maximum_c"] <= (83 if args.micro_paced else 72),
                "zero_reactive_training_halts": result["training_cooling_halts"] == 0,
                "zero_reactive_session_halts": cooling_halts == 0,
                "telemetry_error_free": not summary["errors"],
                "uninstrumented": not args.profile,
            }
            result["paced_preliminary_checks"] = checks
            result["paced_preliminary_objective_met"] = all(checks.values())
        write("metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
