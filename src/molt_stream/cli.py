from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from molt_stream import __version__
from molt_stream.core.contracts import ProgressEvent
from molt_stream.core.errors import MoltStreamError
from molt_stream.training.stream_benchmark import inspect_capabilities, stream_tune
from molt_stream.training.engine import evaluate_run, generate_run, load_spec, train
from molt_stream.training.throughput import fusion_memory_benchmark, training_throughput_benchmark
from molt_stream.training.loss_partition import benchmark_exact_loss_partitioning
from molt_stream.training.qlora import benchmark_qlora_adapter
from molt_stream.measurement.frontier import QualityEnergyPoint, frontier_dict
from molt_stream.measurement.comparison import (
    aggregate_comparisons,
    compare_runs,
    write_comparison,
)
from molt_stream.measurement.telemetry import manage_power_limit
from molt_stream.training.curriculum import run_curriculum_experiment


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


class TerminalUI:
    GREEN = "\x1b[38;2;34;197;94m"
    MUTED = "\x1b[38;2;107;114;128m"
    WHITE = "\x1b[38;2;245;245;245m"
    BLACK = "\x1b[38;2;0;0;0m"
    GREEN_BG = "\x1b[48;2;34;197;94m"
    BLACK_BG = "\x1b[48;2;0;0;0m"
    RESET = "\x1b[0m"

    def __init__(self, *, enabled: bool, color: bool = True):
        self.enabled = enabled
        self.color = color
        self._progress_active = False

    def _style(self, value: str, foreground: str = "", background: str = "") -> str:
        if not self.color:
            return value
        return f"{self.BLACK_BG}{foreground}{background}{value}{self.RESET}"

    def badge(self, label: str) -> str:
        return self._style(f"[ {label} ]", self.BLACK, self.GREEN_BG)

    def check(self, message: str) -> None:
        self.finish_progress()
        print(f"{self._style('✓', self.GREEN)} {self._style(message, self.WHITE)}")

    def card(self, title: str, rows: list[tuple[str, object]]) -> None:
        self.finish_progress()
        clean = []
        for label, value in rows:
            raw = "—" if value is None else str(value)
            visible = f"[ {raw.upper()} ]" if str(label).lower() == "state" else raw
            clean.append((str(label), raw, visible))
        label_width = max((len(label) for label, _, _ in clean), default=0)
        inner_width = max(
            42,
            len(title) + 7,
            max((1 + label_width + 3 + len(visible) + 1 for _, _, visible in clean), default=0),
        )
        inner_width = min(inner_width, 96)
        badge_width = len(title) + 4
        header_tail = max(1, inner_width - 3 - badge_width)
        print(
            self._style("╭─ ", self.MUTED)
            + self.badge(title)
            + self._style(" " + "─" * header_tail + "╮", self.MUTED)
        )
        print(self._style("│", self.MUTED) + " " * inner_width + self._style("│", self.MUTED))
        for label, value, visible in clean:
            maximum_value_width = inner_width - label_width - 5
            if len(visible) > maximum_value_width:
                visible = visible[: maximum_value_width - 1] + "…"
                value = value[: maximum_value_width - 1] + "…"
            if label.lower() == "state":
                display = self.badge(value.upper())
            else:
                display = self._style(value, self.WHITE)
            padding = inner_width - label_width - 5 - len(visible)
            print(
                self._style("│", self.MUTED)
                + " "
                + self._style(label.ljust(label_width), self.MUTED)
                + "   "
                + display
                + " " * max(0, padding)
                + " "
                + self._style("│", self.MUTED)
            )
        print(self._style("│", self.MUTED) + " " * inner_width + self._style("│", self.MUTED))
        print(self._style("╰" + "─" * inner_width + "╯", self.MUTED))

    def progress(self, event: ProgressEvent) -> None:
        if event.kind == "startup":
            self.check(event.message)
            return
        if event.kind == "thermal_abort":
            self.finish_progress()
            print(
                f"{self.badge('THERMAL_ABORT')} "
                f"{self._style(event.message, self.WHITE)}"
            )
            return
        if event.kind != "step" or event.total_steps <= 0:
            return
        fraction = min(1.0, event.step / event.total_steps)
        width = 20
        filled = round(width * fraction)
        bar = self._style("█" * filled, self.GREEN) + self._style("░" * (width - filled), self.MUTED)
        session_steps = max(1, event.step - event.initial_step)
        eta = (
            event.elapsed_seconds / session_steps * (event.total_steps - event.step)
            if session_steps else 0.0
        )
        speed = f"{event.tokens_per_second:,.0f} tok/s" if event.tokens_per_second is not None else "— tok/s"
        loss = f"loss {event.loss:.4f}" if event.loss is not None else "loss —"
        vram = f"VRAM {event.vram_bytes / 1e9:.2f} GB" if event.vram_bytes is not None else "VRAM —"
        temp = f"GPU {event.gpu_temperature_c:.1f}°C" if event.gpu_temperature_c is not None else "GPU —°C"
        pause_ms = round(event.thermal_pause_seconds * 1000)
        if event.thermal_state == "protective-cooling":
            cooling = f"[❄ PROTECT {pause_ms}ms]"
        elif event.thermal_pause_seconds > 0:
            cooling = f"[❄ COOLING {pause_ms}ms]"
        else:
            cooling = "[⚡ FULL SPEED]"
        line = (
            f"{bar} {fraction * 100:5.1f}%  ETA {_duration(eta)}  {speed}  "
            f"{loss}  {temp} {cooling}  {vram}"
        )
        sys.stdout.write("\r\x1b[2K" + line)
        sys.stdout.flush()
        self._progress_active = True

    def finish_progress(self) -> None:
        if self._progress_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._progress_active = False


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{remainder:02d}" if hours else f"{minutes:02d}:{remainder:02d}"


def _bytes(value: object) -> str:
    if value is None:
        return "—"
    amount = float(value)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1000 or suffix == "TB":
            return f"{amount:.2f} {suffix}"
        amount /= 1000
    return str(value)


def _inspect_rows(value: dict[str, Any]) -> list[tuple[str, object]]:
    return [
        ("State", "ready" if value.get("cuda_available") else "limited"),
        ("Platform", value.get("platform")),
        ("Python", value.get("python")),
        ("PyTorch", value.get("torch")),
        ("GPU", value.get("gpu", "Not available")),
        ("VRAM", _bytes(value.get("gpu_vram_bytes"))),
        ("System RAM", _bytes(value.get("system_ram_bytes"))),
        ("CUDA", value.get("cuda_runtime", "Not available")),
        ("Triton", "available" if value.get("triton") else "unavailable"),
        ("QLoRA 8B", "ready" if value.get("qlora_8b_ready") else "dependencies missing"),
    ]


def _report_rows(value: dict[str, Any], run: str | Path) -> list[tuple[str, object]]:
    telemetry = value.get("telemetry", {})
    evaluations = value.get("evaluations", [])
    final = evaluations[-1] if evaluations else {}
    return [
        ("Run", Path(run).name),
        ("State", value.get("state", "unknown")),
        ("Steps", value.get("step")),
        ("Tokens", f"{int(value.get('tokens', 0)):,}"),
        ("Speed", f"{float(value.get('tokens_per_second', 0)):,.0f} tokens/s"),
        ("Validation NLL", final.get("nll")),
        ("Perplexity", final.get("perplexity")),
        ("Peak VRAM", _bytes(value.get("cuda_peak_allocated_bytes"))),
        ("Peak temperature", f"{telemetry.get('peak_gpu_temperature_c', '—')}°C"),
        ("Board energy", f"{telemetry.get('gpu_board_energy_joules', '—')} J"),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molt", description="MOLT local training research engine")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the installed MOLT version and exit",
    )
    presentation = parser.add_mutually_exclusive_group()
    presentation.add_argument("--json", action="store_true", help="Force machine-readable JSON output")
    presentation.add_argument("--ui", action="store_true", help="Force the interactive terminal UI")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color while retaining the UI layout")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Inspect runtime and optional acceleration capabilities")
    prepare = commands.add_parser("prepare", help="Validate an mmap-ready token file")
    prepare.add_argument("--config", required=True)
    training = commands.add_parser("train", help="Run from-scratch SLM pretraining")
    training.add_argument("--config", required=True)
    training.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the configured seed; the resolved run spec records the value",
    )
    training.add_argument("--galore", action="store_true")
    resume = commands.add_parser("resume", help="Resume an interrupted run exactly")
    resume.add_argument("--run", required=True)
    resume.add_argument("--galore", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Evaluate a completed run")
    evaluate.add_argument("--run", required=True)
    report = commands.add_parser("report", help="Print a run's machine-readable summary")
    report.add_argument("--run", required=True)
    comparison = commands.add_parser(
        "compare", help="Compare candidate and baseline end-to-end evidence gates"
    )
    comparison.add_argument("baseline_run")
    comparison.add_argument("candidate_run")
    comparison.add_argument("--quality-tolerance-percent", type=float, default=1.0)
    comparison.add_argument("--minimum-improvement-percent", type=float, default=1.0)
    comparison.add_argument("--thermal-peak-tolerance-c", type=float, default=1.0)
    comparison.add_argument("--output", default=None, help="Optional atomic JSON artifact path")
    paired = commands.add_parser(
        "compare-paired", help="Aggregate paired comparisons with bootstrap intervals"
    )
    paired.add_argument(
        "--pair", action="append", nargs=2, required=True,
        metavar=("BASELINE_RUN", "CANDIDATE_RUN"),
    )
    paired.add_argument("--quality-tolerance-percent", type=float, default=1.0)
    paired.add_argument("--minimum-improvement-percent", type=float, default=1.0)
    paired.add_argument("--thermal-peak-tolerance-c", type=float, default=1.0)
    paired.add_argument("--bootstrap-samples", type=int, default=10_000)
    paired.add_argument("--bootstrap-seed", type=int, default=20260902)
    paired.add_argument("--output", default=None, help="Optional atomic JSON artifact path")
    frontier = commands.add_parser("frontier", help="Build a time/board-energy/perplexity frontier")
    frontier.add_argument("runs", nargs="+")
    generate = commands.add_parser("generate", help="Generate greedy token IDs")
    generate.add_argument("--run", required=True)
    generate.add_argument("--prompt-ids", required=True, help="Comma-separated token IDs")
    generate.add_argument("--max-new-tokens", type=int, default=32)
    tune = commands.add_parser("stream-tune", help="Benchmark double-buffered NF4+LoRA streaming")
    tune.add_argument("--width", type=int, default=1024)
    tune.add_argument("--layers", type=int, default=8)
    tune.add_argument("--sequence", type=int, default=128)
    tune.add_argument("--batch", type=int, default=1)
    tune.add_argument("--steps", type=int, default=10)
    tune.add_argument("--rank", type=int, default=8)
    tune.add_argument("--bundle-size", type=int, default=4)
    tune.add_argument("--seed", type=int, default=1337)
    tune.add_argument("--synchronous", action="store_true", help="Disable double buffering for ablation")
    benchmark = commands.add_parser("benchmark", help="Measure sustained full-update training throughput")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--warmup-steps", type=int, default=10)
    benchmark.add_argument("--steps", type=int, default=50)
    fusion = commands.add_parser("fusion-benchmark", help="Falsify the Windows max-autotune memory gate")
    fusion.add_argument("--config", required=True)
    curriculum = commands.add_parser(
        "curriculum-benchmark",
        help="Run paired fixed-context versus sequence-warmup time/energy experiments",
    )
    curriculum.add_argument("--config", required=True)
    curriculum.add_argument(
        "--seeds", default="1337,2027,4099", help="Comma-separated paired experiment seeds"
    )
    curriculum.add_argument(
        "--eval-interval", type=int, default=None, help="Held-out evaluations every N updates"
    )
    loss_partition = commands.add_parser(
        "loss-partition-benchmark",
        help="Measure exact partitioned linear-CE correctness, time, energy, and memory",
    )
    loss_partition.add_argument("--config", required=True)
    loss_partition.add_argument("--chunks", default="1024,2048,4096,8192")
    loss_partition.add_argument("--warmup-steps", type=int, default=2)
    loss_partition.add_argument("--steps", type=int, default=5)
    power = commands.add_parser("power-limit", help="Query or explicitly request an NVML power limit")
    power.add_argument("--watts", type=float, default=65.0)
    power.add_argument("--apply", action="store_true", help="Request the hardware change; may require administrator rights")
    qlora_benchmark = commands.add_parser(
        "qlora-benchmark",
        help="Compare a saved QLoRA adapter with its unchanged local base model",
    )
    qlora_benchmark.add_argument("--run", required=True)
    qlora_benchmark.add_argument("--batches", type=int, default=8)
    qlora_benchmark.add_argument("--split", choices=("train", "validation"), default="validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ui = TerminalUI(
        enabled=args.ui or (not args.json and sys.stdout.isatty()),
        color=not args.no_color,
    )
    if ui.enabled:
        # Windows PowerShell may expose a legacy code page even though Windows
        # Terminal supports Unicode. Reconfigure before emitting box glyphs.
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")

    def output(value: Any, *, title: str = "Result", rows: list[tuple[str, object]] | None = None) -> None:
        if ui.enabled:
            ui.card(title, rows if rows is not None else [("Result", json.dumps(value, default=str))])
        else:
            _print(value)

    try:
        if args.command == "inspect":
            value = inspect_capabilities()
            output(value, title="Workspace info", rows=_inspect_rows(value))
        elif args.command == "prepare":
            spec = load_spec(args.config)
            value = {"path": spec.data.path, "bytes": Path(spec.data.path).stat().st_size, "storage_backend": "mmap", "validated": True}
            if ui.enabled:
                ui.check("Zero-RAM memory-mapped dataset initialized")
            output(value, title="Dataset info", rows=[("State", "ready"), ("Path", value["path"]), ("Size", _bytes(value["bytes"])), ("Backend", "OS mmap")])
        elif args.command == "train":
            spec = load_spec(args.config)
            if args.seed is not None:
                spec = replace(spec, seed=args.seed)
            if ui.enabled:
                ui.card("Training info", [
                    ("State", "running"), ("Model", f"{spec.model.layers}L × {spec.model.width}D"),
                    ("Context", spec.model.context_length), ("Batch", f"{spec.batch_size} × {spec.gradient_accumulation}"),
                    ("Backend", spec.execution_backend), ("Thermal target", f"{spec.thermal_target_c:.1f}°C"),
                    ("Cruise ceiling", f"{spec.thermal_cruise_max_c:.1f}°C" if spec.thermal_cruise_max_c is not None else "—"),
                    ("Abort boundary", f"{spec.thermal_abort_c:.1f}°C"),
                ])
            path = train(spec, use_galore=args.galore, progress=ui.progress if ui.enabled else None)
            ui.finish_progress()
            summary = json.loads((path / "metrics.summary.json").read_text("utf-8"))
            title = "Training stopped" if summary.get("state") == "thermal_abort" else "Training complete"
            output({"run": str(path)}, title=title, rows=_report_rows(summary, path))
        elif args.command == "resume":
            root = Path(args.run)
            path = train(load_spec(root / "spec.resolved.json"), resume=root, use_galore=args.galore, progress=ui.progress if ui.enabled else None)
            ui.finish_progress()
            summary = json.loads((path / "metrics.summary.json").read_text("utf-8"))
            title = "Training stopped" if summary.get("state") == "thermal_abort" else "Training resumed"
            output({"run": str(path)}, title=title, rows=_report_rows(summary, path))
        elif args.command == "evaluate":
            value = evaluate_run(args.run)
            output(value, title="Evaluation", rows=[("State", "complete"), ("Validation NLL", value["validation_nll"]), ("Perplexity", value["validation_perplexity"])])
        elif args.command == "report":
            value = json.loads((Path(args.run) / "metrics.summary.json").read_text("utf-8"))
            output(value, title="Training report", rows=_report_rows(value, args.run))
        elif args.command == "compare":
            value = compare_runs(
                args.baseline_run,
                args.candidate_run,
                quality_tolerance_percent=args.quality_tolerance_percent,
                minimum_improvement_percent=args.minimum_improvement_percent,
                thermal_peak_tolerance_c=args.thermal_peak_tolerance_c,
            )
            if args.output is not None:
                value["artifact_path"] = str(write_comparison(args.output, value))
            output(value, title="Run comparison")
        elif args.command == "compare-paired":
            comparisons = [
                compare_runs(
                    baseline,
                    candidate,
                    quality_tolerance_percent=args.quality_tolerance_percent,
                    minimum_improvement_percent=args.minimum_improvement_percent,
                    thermal_peak_tolerance_c=args.thermal_peak_tolerance_c,
                )
                for baseline, candidate in args.pair
            ]
            value = aggregate_comparisons(
                comparisons,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
            if args.output is not None:
                value["artifact_path"] = str(write_comparison(args.output, value))
            output(value, title="Paired comparison")
        elif args.command == "frontier":
            points = []
            signature = None
            for item in args.runs:
                spec_value = json.loads((Path(item) / "spec.resolved.json").read_text("utf-8"))
                current_signature = {
                    key: spec_value[key]
                    for key in ("mode", "data", "model", "batch_size", "gradient_accumulation", "max_steps", "learning_rate", "seed")
                }
                if signature is None:
                    signature = current_signature
                elif current_signature != signature:
                    raise ValueError(f"frontier workload semantics differ: {item}")
                summary = json.loads((Path(item) / "metrics.summary.json").read_text("utf-8"))
                energy = summary["telemetry"]["gpu_board_energy_joules"]
                if energy is None:
                    raise ValueError(f"run has no board-energy measurement: {item}")
                points.append(QualityEnergyPoint(
                    Path(item).name, float(summary["seconds"]), float(energy),
                    float(summary["evaluations"][-1]["perplexity"]),
                ))
            output(frontier_dict(points), title="Efficiency frontier")
        elif args.command == "generate":
            prompt = [int(item) for item in args.prompt_ids.split(",") if item.strip()]
            output({"token_ids": generate_run(args.run, prompt, args.max_new_tokens)}, title="Generation")
        elif args.command == "stream-tune":
            output(stream_tune(width=args.width, layers=args.layers, sequence=args.sequence, batch=args.batch, steps=args.steps, rank=args.rank, double_buffer=not args.synchronous, bundle_size=args.bundle_size, seed=args.seed), title="Stream benchmark")
        elif args.command == "benchmark":
            output(training_throughput_benchmark(load_spec(args.config), warmup_steps=args.warmup_steps, measured_steps=args.steps), title="Throughput benchmark")
        elif args.command == "fusion-benchmark":
            output(fusion_memory_benchmark(load_spec(args.config)), title="Fusion benchmark")
        elif args.command == "curriculum-benchmark":
            seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
            value = run_curriculum_experiment(
                load_spec(args.config), seeds=seeds, eval_interval=args.eval_interval
            )
            aggregate = value["aggregate"]
            output(
                value,
                title="Milestone 3 experiment",
                rows=[
                    ("State", aggregate["decision"]),
                    ("Experiment", value["experiment_id"]),
                    ("Method", value["method"]),
                    ("Paired seeds", aggregate["pair_count"]),
                    ("Median time gain", f"{aggregate['median_time_improvement_percent']}%"),
                    ("Median energy gain", f"{aggregate['median_energy_improvement_percent']}%"),
                    ("Artifact", value["artifact_path"]),
                ],
            )
        elif args.command == "loss-partition-benchmark":
            chunks = tuple(int(item.strip()) for item in args.chunks.split(",") if item.strip())
            value = benchmark_exact_loss_partitioning(
                load_spec(args.config),
                chunk_sizes=chunks,
                warmup_steps=args.warmup_steps,
                measured_steps=args.steps,
            )
            decision = value["decision"]
            output(
                value,
                title="Exact-loss frontier",
                rows=[
                    ("State", "survived" if decision["survived"] else "rejected"),
                    ("Experiment", value["experiment_id"]),
                    ("Selected chunk", decision["selected_chunk_size"]),
                    ("Reason", decision["reason"]),
                    ("Artifact", value["artifact_path"]),
                ],
            )
        elif args.command == "power-limit":
            output(manage_power_limit(args.watts, apply=args.apply), title="GPU power limit")
        elif args.command == "qlora-benchmark":
            root = Path(args.run)
            value = benchmark_qlora_adapter(
                load_spec(root / "spec.resolved.json"),
                root,
                batches=args.batches,
                split=args.split,
            )
            output(value, title="QLoRA benchmark")
        return 0
    except (MoltStreamError, FileNotFoundError, ValueError, RuntimeError) as exc:
        ui.finish_progress()
        if ui.enabled:
            print(f"{ui._style('✕', ui.MUTED)} {ui._style(str(exc), ui.WHITE)}", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
