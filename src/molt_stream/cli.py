"""MOLT unified command-line and interactive console interface.

Provides hardware-aware AI training with:
- Zero-configuration interactive landing and guided training flows.
- High-level policy profiles (SPEED, BALANCED, COOL, ENERGY).
- Automatic workspace and hardware discovery.
- Atomic checkpoint resumption and empirical benchmarking.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

# Silence harmless PyTorch Windows flop_counter and re-entrant warnings
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
warnings.filterwarnings("ignore", message=".*triton not found.*")
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")

from molt_stream import __version__
from molt_stream.core.contracts import ProgressEvent
from molt_stream.core.errors import MoltError
from molt_stream.core.specs import (
    DataSpec,
    ModelSpec,
    StreamSpec,
    TrainingSpec,
    load_spec,
)
from molt_stream.core.discovery import (
    find_datasets,
    find_models,
    find_runs,
    get_hardware_info,
    init_workspace,
)
from molt_stream.core.profiles import PROFILES, apply_profile
from molt_stream.core.system_tuning import prioritized_execution


class TerminalUI:
    GREEN = "\x1b[38;2;34;197;94m"
    MUTED = "\x1b[38;2;107;114;128m"
    WHITE = "\x1b[38;2;245;245;245m"
    BLACK = "\x1b[38;2;0;0;0m"
    RED = "\x1b[38;2;239;68;68m"
    YELLOW = "\x1b[38;2;234;179;8m"
    CYAN = "\x1b[38;2;6;182;212m"
    BOLD = "\x1b[1m"
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
        columns = shutil.get_terminal_size((140, 24)).columns if sys.stdout.isatty() else 140
        width = 20 if columns >= 135 else 12 if columns >= 110 else 8
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
        cpu_temp = (
            f"CPU {event.cpu_temperature_c:.1f}°C"
            if event.cpu_temperature_c is not None
            else "CPU sensor —"
        )
        pause_ms = round(event.thermal_pause_seconds * 1000)
        if event.thermal_state == "gear-1-sprint":
            cooling = "[Gear 1]"
        elif event.thermal_state == "gear-2-cooldown":
            cooling = f"[Gear 2: {pause_ms}ms pause]"
        elif event.thermal_state == "pit-stop-cooldown":
            cooling = f"[Thermal Pause: {event.thermal_pause_seconds:.1f}s]"
        elif event.thermal_state == "protective-cooling":
            cooling = f"[Thermal Pause: {pause_ms}ms]"
        elif event.thermal_pause_seconds > 0:
            cooling = f"[Pause: {pause_ms}ms]"
        else:
            cooling = "[Normal]"
        details = [speed, loss, temp, cpu_temp, cooling, vram]
        if columns < 130:
            details.remove(vram)
        if columns < 110:
            details.remove(loss)
        line = f"{bar} {fraction * 100:5.1f}%  ETA {_duration(eta)}  " + "  ".join(details)
        sys.stdout.write("\r\x1b[2K" + line)
        sys.stdout.flush()
        self._progress_active = True

    def finish_progress(self) -> None:
        if self._progress_active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._progress_active = False

    def select(
        self,
        title: str,
        options: list[tuple[str, str]],
        *,
        default: int = 0,
        allow_cancel: bool = False,
    ) -> int | None:
        """Select an option with arrow keys on a terminal, numbers otherwise."""
        if not options:
            raise ValueError("selection requires at least one option")
        selected = min(max(default, 0), len(options) - 1)
        self.finish_progress()
        print(self._style(title, self.BOLD + self.WHITE))
        if sys.stdin.isatty() and sys.stdout.isatty() and os.name == "nt":
            import msvcrt

            def render(*, move_up: bool) -> None:
                if move_up:
                    sys.stdout.write(f"\x1b[{len(options)}A")
                for index, (label, description) in enumerate(options):
                    pointer = "❯" if index == selected else " "
                    detail = f"  {description}" if description else ""
                    foreground = self.GREEN if index == selected else self.WHITE
                    sys.stdout.write(
                        "\r\x1b[2K"
                        + self._style(f" {pointer} {label}", foreground)
                        + self._style(detail, self.MUTED)
                        + "\n"
                    )
                sys.stdout.flush()

            render(move_up=False)
            while True:
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    key = msvcrt.getwch()
                    if key == "H":
                        selected = (selected - 1) % len(options)
                    elif key == "P":
                        selected = (selected + 1) % len(options)
                    else:
                        continue
                    render(move_up=True)
                elif key in ("\r", "\n"):
                    return selected
                elif key == "\x1b" and allow_cancel:
                    return None
                elif key == "\x03":
                    raise KeyboardInterrupt

        for index, (label, description) in enumerate(options, start=1):
            suffix = f" — {description}" if description else ""
            print(f"  {index}. {label}{suffix}")
        try:
            raw = input(f"Select [{default + 1}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            if allow_cancel:
                return None
            return default
        if not raw:
            return default
        lowered = raw.lower()
        for index, (label, _description) in enumerate(options):
            if lowered in {label.lower(), label[:1].lower()}:
                return index
        try:
            value = int(raw) - 1
        except ValueError:
            return default
        return value if 0 <= value < len(options) else default

    def confirm(self, question: str, *, default: bool = True) -> bool:
        options = [("Yes", ""), ("No", "")]
        selected = self.select(question, options, default=0 if default else 1)
        return selected == 0


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{remainder:02d}" if hours else f"{minutes:02d}:{remainder:02d}"


def _bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "—"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _report_rows(summary: dict[str, Any], path: Path) -> list[tuple[str, object]]:
    evaluations = summary.get("evaluations", [])
    initial = evaluations[0] if evaluations else {}
    final = evaluations[-1] if evaluations else {}
    final_perplexity = final.get("perplexity")
    telemetry = summary.get("telemetry", {})
    return [
        ("State", summary.get("state", "complete")),
        ("Run", str(path)),
        ("Tokens", f"{summary.get('tokens', 0):,}"),
        ("Wall duration", f"{summary.get('seconds', 0.0):.2f}s"),
        ("Throughput", f"{summary.get('tokens_per_second', 0.0):,.2f} tok/s"),
        ("Validation loss", f"{final.get('nll', '—')} (init: {initial.get('nll', '—')})"),
        ("Perplexity", f"{final_perplexity:.2f}" if isinstance(final_perplexity, (int, float)) else "—"),
        ("Board energy", f"{telemetry.get('gpu_board_energy_joules', '—')} J"),
    ]


def _resolve_execution_mode(requested: str | None, ui: TerminalUI) -> str:
    if requested in ("normal", "prioritize"):
        return requested
    if not ui.enabled or not sys.stdin.isatty():
        return "normal"
    choice = ui.select(
        "Execution mode",
        [
            ("Normal", "Recommended; leaves other applications unchanged"),
            ("Prioritized", "Temporarily raises MOLT process priority"),
        ],
    )
    return "prioritize" if choice == 1 else "normal"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="molt",
        description="MOLT: Hardware-Aware AI Training Runtime for Local Hardware",
    )
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
    parser.add_argument("--debug", action="store_true", help="Show full exception tracebacks on error")

    commands = parser.add_subparsers(dest="command", required=False)

    commands.add_parser("doctor", help="Show installation identity, environment, and optional capabilities")
    setup = commands.add_parser("setup", help="Install and verify the complete NVIDIA training runtime")
    setup.add_argument("-y", "--yes", action="store_true", help="Install without confirmation")
    setup.add_argument("--dry-run", action="store_true", help="Show what would be installed")
    for name, help_text in (
        ("update", "Update the managed per-user MOLT runtime"),
        ("repair", "Reinstall and verify the managed per-user MOLT runtime"),
        ("uninstall", "Remove the managed MOLT runtime, launcher, and package cache"),
    ):
        management = commands.add_parser(name, help=help_text)
        management.add_argument("--dry-run", action="store_true", help="Show the planned action")
        if name == "uninstall":
            management.add_argument("-y", "--yes", action="store_true", help="Remove without confirmation")
    commands.add_parser("runs", help="List saved runs and verified checkpoint status")
    fit = commands.add_parser("fit-test", help="Run two optimizer steps at configured geometry; not a sustained benchmark")
    fit.add_argument("--config", required=True)
    export = commands.add_parser("export", help="Export a verified run as a MOLT bundle, PEFT adapter, or GGUF adapter")
    export.add_argument("--run", required=True)
    export.add_argument("--output", "--output-dir", dest="output_dir", required=True)
    export.add_argument("--format", choices=("auto", "molt", "hf", "gguf"), default="auto")
    export.add_argument("--llama-cpp", help="llama.cpp checkout containing convert_lora_to_gguf.py (GGUF only)")

    # 1. info
    commands.add_parser("info", help="Display hardware diagnostics, VRAM, and suggested profiles")

    # 2. inspect (backward compatibility)
    commands.add_parser("inspect", help="Inspect runtime and optional acceleration capabilities")

    # 3. config
    config_cmd = commands.add_parser("config", help="Workspace and configuration management")
    config_cmd.add_argument("--init", action="store_true", help="Initialize standard molt-workspace/ directory")
    config_cmd.add_argument("--list", action="store_true", help="List discovered models, datasets, and configurations")

    # 4. prepare
    prepare = commands.add_parser("prepare", help="Prepare text, JSONL, or Parquet as an mmap token dataset")
    prepare.add_argument("source", nargs="?", help="Input .txt, .jsonl, or .parquet file")
    prepare.add_argument("--config", help="Validate an existing binary dataset configuration")
    prepare.add_argument("--text-file", help="Legacy alias for the source text path")
    prepare.add_argument("--model", help="Local model directory; supplies tokenizer and emits a starter config")
    prepare.add_argument("--tokenizer", help="Local tokenizer/model directory")
    prepare.add_argument("--base-model", help="Local base model; also emit a starter QLoRA training config")
    prepare.add_argument("--output", dest="output_dir", help="New output directory (alias: --output-dir)")
    prepare.add_argument("--output-dir", dest="output_dir", help="New output directory; never overwritten")
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--schema", choices=("auto", "text", "messages", "prompt-completion"), default="auto")
    prepare.add_argument("--text-column", default="text")
    prepare.add_argument("--messages-column", default="messages")
    prepare.add_argument("--prompt-column", default="prompt")
    prepare.add_argument("--completion-column", default="completion")

    # 5. train
    training = commands.add_parser("train", help="Run model training with hardware-aware thermal pacing")
    training.add_argument("--config", default=None, help="Path to JSON training configuration")
    training.add_argument("--model", default=None, help="Path or name of base model directory")
    training.add_argument(
        "--dataset",
        default=None,
        help="Path to .txt, .jsonl, .parquet, or prepared .bin training data",
    )
    training.add_argument(
        "--profile",
        choices=("speed", "balanced", "cool", "energy"),
        default=None,
        help="High-level policy profile (speed, balanced, cool, energy)",
    )
    training.add_argument("--steps", type=int, default=None, help="Override maximum optimization steps")
    training.add_argument("--batch-size", type=int, default=None, help="Micro-batch size")
    training.add_argument("--context-length", type=int, default=None, help="Sequence context length")
    training.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    training.add_argument("--seed", type=int, default=None, help="Random seed")
    training.add_argument("--galore", action="store_true", help="Enable GaLore memory-efficient projection")
    training.add_argument(
        "--mode-select",
        choices=("normal", "prioritize"),
        default=None,
        help="Select execution profile: normal or prioritize",
    )
    training.add_argument("--dry-run", action="store_true", help="Validate specification without training or host tuning; does not verify VRAM fit")
    training.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    # 6. resume
    resume = commands.add_parser("resume", help="Resume an interrupted run from verified atomic checkpoint")
    resume.add_argument("--run", default=None, help="Path to run directory containing checkpoint.pt")
    resume.add_argument("--galore", action="store_true")
    resume.add_argument(
        "--mode-select",
        choices=("normal", "prioritize"),
        default=None,
        help="Select execution profile: normal or prioritize",
    )
    resume.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    # 7. benchmark
    benchmark = commands.add_parser("benchmark", help="Measure sustained throughput and run smoke tests")
    benchmark.add_argument("--config", default=None, help="Optional custom benchmark configuration")
    benchmark.add_argument("--smoke", action="store_true", help="Run fast 20-step smoke benchmark")
    benchmark.add_argument("--warmup-steps", type=int, default=5)
    benchmark.add_argument("--steps", type=int, default=20)

    # 8. evaluate & report
    evaluate = commands.add_parser("evaluate", help="Evaluate a completed run on validation dataset")
    evaluate.add_argument("--run", required=True)
    report = commands.add_parser("report", help="Print a run's machine-readable summary")
    report.add_argument("--run", required=True)

    # 9. comparison & research benchmarks
    comparison = commands.add_parser("compare", help="Compare candidate and baseline end-to-end evidence gates")
    comparison.add_argument("baseline_run")
    comparison.add_argument("candidate_run")
    comparison.add_argument("--quality-tolerance-percent", type=float, default=1.0)
    comparison.add_argument("--minimum-improvement-percent", type=float, default=1.0)
    comparison.add_argument("--thermal-peak-tolerance-c", type=float, default=1.0)
    comparison.add_argument("--output", default=None, help="Optional atomic JSON artifact path")

    paired = commands.add_parser("compare-paired", help="Aggregate paired comparisons with bootstrap intervals")
    paired.add_argument("--pair", action="append", nargs=2, required=True, metavar=("BASELINE_RUN", "CANDIDATE_RUN"))
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
    prompts = generate.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt-ids", help="Comma-separated token IDs")
    prompts.add_argument("--prompt", help="Text prompt using the model tokenizer")
    generate.add_argument("--tokenizer", help="Local tokenizer, required for text with scratch models")
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

    fusion = commands.add_parser("fusion-benchmark", help="Falsify the Windows max-autotune memory gate")
    fusion.add_argument("--config", required=True)

    curriculum = commands.add_parser("curriculum-benchmark", help="Run paired fixed-context versus sequence-warmup experiments")
    curriculum.add_argument("--config", required=True)
    curriculum.add_argument("--seeds", default="1337,2027,4099")
    curriculum.add_argument("--eval-interval", type=int, default=None)

    loss_partition = commands.add_parser("loss-partition-benchmark", help="Measure partitioned linear-CE correctness")
    loss_partition.add_argument("--config", required=True)
    loss_partition.add_argument("--chunks", default="1024,2048,4096,8192")
    loss_partition.add_argument("--warmup-steps", type=int, default=2)
    loss_partition.add_argument("--steps", type=int, default=5)

    power = commands.add_parser("power-limit", help="Query or explicitly request an NVML power limit")
    power.add_argument("--watts", type=float, default=65.0)
    power.add_argument("--apply", action="store_true", help="Request the hardware change")

    optimize_gpu = commands.add_parser(
        "optimize-gpu", help="Run training inside an explicit, automatically restored GPU profile"
    )
    optimize_gpu.add_argument("--profile", choices=("endurance",), default="endurance")
    optimize_gpu.add_argument("--config", required=True, help="Training configuration")
    optimize_gpu.add_argument("-y", "--yes", action="store_true", help="Confirm the temporary clock change")

    qlora_benchmark = commands.add_parser("qlora-benchmark", help="Compare a saved QLoRA adapter with local base model")
    qlora_benchmark.add_argument("--run", required=True)
    qlora_benchmark.add_argument("--batches", type=int, default=8)
    qlora_benchmark.add_argument("--split", choices=("train", "validation"), default="validation")

    for command_parser in commands.choices.values():
        for option in ("--json", "--ui", "--no-color", "--debug"):
            command_parser.add_argument(option, action="store_true", default=argparse.SUPPRESS,
                                        help=argparse.SUPPRESS)
    research = commands.add_parser("research", help="Experimental benchmarks and evidence comparisons")
    experiments = research.add_subparsers(dest="research_command", required=True)
    for name in ("compare", "compare-paired", "frontier", "stream-tune", "fusion-benchmark",
                 "curriculum-benchmark", "loss-partition-benchmark", "qlora-benchmark"):
        experiments.add_parser(name, parents=[commands.choices[name]], add_help=False)
    return parser


def handle_doctor(ui: TerminalUI, output_func: Any) -> int:
    from molt_stream.training.diagnostics import doctor
    value = doctor()
    hardware = get_hardware_info()
    output_func(value, title="MOLT Doctor", rows=[
        ("Version", value["version"]), ("Commit", value["git_commit"] or "Installed package"),
        ("Python executable", value["python_executable"]), ("Package", value["package_path"]),
        ("Environment", (
            "Managed per-user runtime"
            if value.get("managed_runtime")
            else "Virtual environment"
            if value["virtual_environment"]
            else "Global Python — run managed installer"
        )),
        ("GPU", value.get("gpu", "CUDA unavailable")),
        ("VRAM", _bytes(value.get("gpu_vram_bytes"))),
        ("System RAM", _bytes(value["system_ram_bytes"])),
        ("CPU temperature", f"{hardware['cpu_temperature_c']}°C" if hardware["cpu_temperature_c"] is not None else "Sensor unavailable"),
        ("QLoRA packages", "Detected; workload not verified" if value["qlora_8b_ready"] else "Missing optional packages"),
        ("Triton", "Detected; compile probe required" if value["triton"] else "Optional; not installed"),
        ("Next", "molt"),
    ])
    return 0


def handle_info(ui: TerminalUI, output_func: Any) -> int:
    """Execute the info command to inspect hardware and environment."""
    hw = get_hardware_info()
    rows = [
        ("Platform", hw["os"]),
        ("Python", hw["python"]),
        ("CPU", f"{hw['cpu']} ({hw['cpu_cores_logical']} logical threads)"),
        ("System RAM", f"{hw['total_ram_gb']} GB ({hw['available_ram_gb']} GB free)"),
        ("PyTorch", f"{hw['pytorch_version'] or 'Not installed'}"),
        ("CUDA Runtime", f"{hw['cuda_version'] or 'None'}"),
        ("NVIDIA GPU", hw["gpu_name"] or "No discrete NVIDIA GPU detected"),
        ("VRAM", f"{hw['gpu_vram_total_gb']} GB ({hw['gpu_vram_free_gb']} GB free)" if hw["gpu_vram_total_gb"] else "—"),
        ("GPU Temp", f"{hw['gpu_temperature_c']}°C" if hw["gpu_temperature_c"] is not None else "—"),
        ("CPU Temp", f"{hw['cpu_temperature_c']}°C" if hw["cpu_temperature_c"] is not None else "Sensor unavailable"),
        ("Power Limit", f"{hw['gpu_power_limit_w']} W" if hw["gpu_power_limit_w"] else "—"),
        ("GPU Driver", hw["gpu_driver"] or "—"),
        ("Suggested Profile", hw["suggested_profile"]),
        ("Memory Budget", f"{hw['suggested_memory_budget_gb']} GB"),
    ]
    output_func(hw, title="MOLT Hardware Diagnostics", rows=rows)
    return 0


_TRAINING_REQUIREMENTS = {
    "accelerate": "accelerate>=1.10,<2",
    "bitsandbytes": "bitsandbytes>=0.48,<1",
    "peft": "peft>=0.17,<1",
    "transformers": "transformers>=5,<6",
    "pyarrow": "pyarrow>=19,<24",
}
if sys.platform == "win32":
    _TRAINING_REQUIREMENTS["triton"] = "triton-windows>=3.4,<3.5"
    _TRAINING_REQUIREMENTS["wmi"] = "WMI==1.5.1"


def _missing_training_requirements() -> list[str]:
    from packaging.requirements import Requirement

    missing: list[str] = []
    for module, requirement_text in _TRAINING_REQUIREMENTS.items():
        requirement = Requirement(requirement_text)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(requirement_text)
            continue
        if importlib.util.find_spec(module) is None or version not in requirement.specifier:
            missing.append(requirement_text)
    return missing


def handle_setup(args: argparse.Namespace, ui: TerminalUI, output_func: Any) -> int:
    """Install the complete optional runtime in the active Python environment."""
    missing = _missing_training_requirements()
    try:
        import torch
        cuda_ready = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        cuda_ready = False
    planned = ([] if cuda_ready else ["torch==2.8.0+cu128"]) + missing
    if not planned:
        if ui.enabled:
            ui.check("MOLT training runtime is already complete")
        return handle_doctor(ui, output_func)
    setup_value = {"python": sys.executable, "install": planned, "dry_run": args.dry_run}
    if ui.enabled:
        ui.card("MOLT Setup", [("Python", sys.executable), ("Install", ", ".join(planned))])
    elif args.dry_run:
        output_func(setup_value, title="MOLT Setup")
    if args.dry_run:
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            raise MoltError("Non-interactive setup requires --yes")
        if not ui.confirm("Install the complete MOLT training runtime?", default=True):
            return 0
    installer = [sys.executable, "-m", "pip", "install"]
    if not cuda_ready:
        result = subprocess.run(
            installer + [
                "torch==2.8.0",
                "--index-url", "https://download.pytorch.org/whl/cu128",
            ],
            check=False,
        )
        if result.returncode:
            raise MoltError("CUDA PyTorch installation failed")
    if missing:
        result = subprocess.run(installer + missing, check=False)
        if result.returncode:
            raise MoltError("MOLT training dependency installation failed")
    checked = subprocess.run(
        [sys.executable, "-m", "pip", "check"], check=False
    )
    if checked.returncode:
        raise MoltError("Installation completed, but dependency verification failed")
    if ui.enabled:
        ui.check("Installation complete")
        ui.card("Next step", [("Command", "molt")])
    else:
        output_func({"status": "complete", "next": "molt"}, title="MOLT Setup")
    return 0


def handle_runtime_management(
    args: argparse.Namespace, ui: TerminalUI, output_func: Any
) -> int:
    """Update, repair, or uninstall the single managed Windows runtime."""

    from molt_stream.runtime_manager import run_action, runtime_home

    action = str(args.command)
    if action == "uninstall" and not args.dry_run and not getattr(args, "yes", False):
        if not sys.stdin.isatty():
            raise MoltError("Non-interactive uninstall requires --yes")
        if not ui.confirm(
            f"Remove MOLT and its cached packages from {runtime_home()}?",
            default=False,
        ):
            ui.check("MOLT was not removed")
            return 0
    result = run_action(action, plan=bool(args.dry_run))
    if result:
        raise MoltError(f"MOLT {action} failed with exit code {result}")
    if args.dry_run:
        return 0
    if action == "uninstall":
        output_func(
            {"status": "scheduled", "runtime": str(runtime_home())},
            title="MOLT Uninstall",
        )
    else:
        output_func(
            {"status": "verified", "action": action, "runtime": str(runtime_home())},
            title=f"MOLT {action.title()}",
        )
    return 0


def _ensure_training_runtime(ui: TerminalUI) -> None:
    missing = _missing_training_requirements()
    if not missing:
        return
    if sys.stdin.isatty() and ui.confirm(
        "Some training components are missing. Install them now?", default=True
    ):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", *missing], check=False
        )
        if result.returncode == 0:
            ui.check("Training components installed")
            return
    raise MoltError("Training components are missing. Run `molt setup` once, then retry.")


def handle_config(args: argparse.Namespace, ui: TerminalUI, output_func: Any) -> int:
    """Handle config command for workspace setup and listing."""
    if args.init:
        created = init_workspace()
        if ui.enabled:
            ui.check(f"Initialized standard MOLT workspace at {created['root'].resolve()}")
            ui.card("Workspace Structure", [
                ("Root", str(created["root"])),
                ("Models", str(created["models"])),
                ("Datasets", str(created["datasets"])),
                ("Runs", str(created["runs"])),
                ("Configs", str(created["configs"])),
            ])
        else:
            _print({"status": "initialized", "root": str(created["root"])})
        return 0

    models = find_models()
    datasets = find_datasets()
    runs = find_runs()
    rows = [
        ("Workspace models", f"{len(models)} detected"),
        ("Workspace datasets", f"{len(datasets)} detected"),
        ("Completed/active runs", f"{len(runs)} detected"),
        ("Profiles available", ", ".join(PROFILES.keys()).upper()),
    ]
    output_func(
        {"models": models, "datasets": datasets, "runs": len(runs), "profiles": list(PROFILES.keys())},
        title="MOLT Workspace Status",
        rows=rows,
    )
    return 0


def _verify_cuda_or_prompt_install(ui: TerminalUI, device: str) -> bool:
    """Ensure CUDA is available if requested, offering automatic installation if running CPU PyTorch."""
    if device != "cuda":
        return True
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return True

    hw = get_hardware_info()
    if ui.enabled and sys.stdin.isatty():
        ui.card("⚠️ CUDA Acceleration Required", [
            ("NVIDIA GPU", hw.get("gpu_name") or "Discrete GPU detected"),
            ("Installed PyTorch", f"torch {torch.__version__} (CPU-only)"),
            ("Cause", "Standard Windows pip installs CPU-only PyTorch by default"),
            ("Fix", "Install PyTorch with NVIDIA CUDA 12.8 wheel"),
        ])
        try:
            if ui.confirm("Install CUDA PyTorch now?", default=True):
                print("[MOLT] Installing CUDA-accelerated PyTorch... (downloading official PyTorch cu128 wheel)")
                uv = shutil.which("uv")
                if uv:
                    cmd = [uv, "pip", "install", "--python", sys.executable]
                elif importlib.util.find_spec("pip") is not None:
                    cmd = [sys.executable, "-m", "pip", "install"]
                else:
                    raise MoltError(
                        "CUDA PyTorch cannot be installed automatically because neither uv nor "
                        "pip is available. Install uv from https://docs.astral.sh/uv/ and retry."
                    )
                cmd.extend([
                    "torch==2.8.0", "--index-url",
                    "https://download.pytorch.org/whl/cu128", "--force-reinstall",
                ])
                res = subprocess.run(cmd, check=False)
                if res.returncode == 0:
                    ui.check("CUDA PyTorch installed successfully! Please re-run: molt")
                    return False
        except (EOFError, KeyboardInterrupt):
            print()

    raise MoltError(
        f"CUDA acceleration is required for GPU training, but this Python environment has a CPU-only build of PyTorch (torch=={torch.__version__}).\n"
        "To enable GPU training on your NVIDIA GPU, run:\n"
        "pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall"
    )


def _guided_path(
    ui: TerminalUI,
    title: str,
    items: list[dict[str, Any]],
    *,
    prompt: str,
) -> str:
    if items and ui.enabled:
        choices = [
            (
                str(item["name"]),
                str(item["path"]),
            )
            for item in items
        ] + [("Choose another path…", "")]
        selected = ui.select(title, choices)
        if selected is not None and selected < len(items):
            return str(items[selected]["path"])
    value = input(prompt).strip().strip("'\"")
    if not value:
        raise MoltError(f"{title} is required")
    return value


def _model_spec_from_local_config(model_path: str, context_length: int) -> ModelSpec:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise MoltError(f"Model config not found: {config_path}")
    config = json.loads(config_path.read_text("utf-8"))

    def required(*names: str) -> int:
        for name in names:
            value = config.get(name)
            if isinstance(value, int) and value > 0:
                return value
        raise MoltError(f"Model config does not provide {names[0]}")

    model_limit = int(config.get("max_position_embeddings", context_length))
    return ModelSpec(
        vocab_size=required("vocab_size"),
        context_length=min(context_length, model_limit),
        layers=required("num_hidden_layers", "n_layer"),
        width=required("hidden_size", "n_embd"),
        heads=required("num_attention_heads", "n_head"),
        hidden_width=required("intermediate_size", "n_inner"),
    )


def _prepare_guided_dataset(dataset_path: str, model_path: str, ui: TerminalUI) -> tuple[Path, Path | None, int]:
    source = Path(dataset_path).resolve()
    if not source.is_file():
        raise MoltError(f"Dataset not found: {source}")
    if source.suffix.lower() == ".bin":
        validation = source.with_name("validation.bin")
        tokens = source.stat().st_size // 4
        if not validation.is_file() or validation == source:
            raise MoltError(
                "Prepared train.bin requires a separate validation.bin in the same directory"
            )
        return source, validation, tokens
    if source.suffix.lower() not in {".txt", ".jsonl", ".parquet"}:
        raise MoltError("Dataset must be .txt, .jsonl, .parquet, or prepared .bin")
    destination = Path("molt-workspace/datasets") / (
        f"{source.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    ui.check("Preparing and validating the selected dataset")
    if source.suffix.lower() in {".jsonl", ".parquet"}:
        from molt_stream.data.records import prepare_records

        result = prepare_records(source, model_path, destination, base_model=model_path)
    else:
        from molt_stream.data.text import prepare_text

        result = prepare_text(source, model_path, destination, base_model=model_path)
    return (
        Path(result["directory"]) / "train.bin",
        Path(result["directory"]) / "validation.bin",
        int(result["train_tokens"]),
    )


def _interactive_interrupt_decision(ui: TerminalUI) -> tuple[bool, bool]:
    ui.finish_progress()
    print()
    if not ui.confirm("Stop training?", default=False):
        ui.check("Training continues")
        return False, True
    save = ui.confirm("Save a verified checkpoint to resume later?", default=True)
    ui.check("Stopping safely at the completed update boundary")
    return True, save


def handle_guided_train(args: argparse.Namespace, ui: TerminalUI, output_func: Any) -> int:
    """Interactive guided training flow for users without full command arguments."""
    from molt_stream.training.engine import train

    spec: TrainingSpec
    guided_spec = not bool(args.config)
    if args.config:
        spec = load_spec(args.config)
    else:
        _ensure_training_runtime(ui)
        model_path = args.model or _guided_path(
            ui, "Select a local model", find_models(), prompt="Local model directory: "
        )
        dataset_path = args.dataset or _guided_path(
            ui, "Select training data", find_datasets(),
            prompt="Training data (.txt, .jsonl, .parquet, or .bin): ",
        )
        ctx = args.context_length or 256
        bs = args.batch_size or 1
        lr = args.learning_rate or 0.0002
        seed = args.seed or 1337
        profile_name = args.profile or "balanced"
        if ui.enabled and not any(
            value is not None
            for value in (args.context_length, args.batch_size, args.learning_rate, args.steps, args.profile)
        ):
            settings = ui.select(
                "Training settings",
                [
                    ("Recommended", "Context 256, batch 1, balanced thermal policy"),
                    ("Customize", "Change the important training settings"),
                ],
            )
            if settings == 1:
                ctx = int(input(f"Context length [{ctx}]: ").strip() or ctx)
                bs = int(input(f"Batch size [{bs}]: ").strip() or bs)
                lr = float(input(f"Learning rate [{lr}]: ").strip() or lr)
                profiles_list = list(PROFILES.items())
                selected_profile = ui.select(
                    "Thermal profile",
                    [(value["name"], value["description"]) for _, value in profiles_list],
                    default=1,
                )
                profile_name = profiles_list[selected_profile or 0][0]

        train_path, validation_path, token_count = _prepare_guided_dataset(
            dataset_path, model_path, ui
        )
        steps = args.steps or max(1, (token_count - 1) // (ctx * bs))
        model_spec = _model_spec_from_local_config(model_path, ctx)

        spec = TrainingSpec(
            mode="qlora",
            base_model=str(Path(model_path).resolve()),
            artifacts_dir="runs",
            batch_size=bs,
            gradient_accumulation=1,
            learning_rate=lr,
            max_steps=steps,
            seed=seed,
            model=model_spec,
            data=DataSpec(
                path=str(train_path.resolve()),
                validation_path=str(validation_path.resolve()) if validation_path else None,
                context_length=ctx,
                storage_dtype="int32",
                sequential=True,
                packing="contiguous",
            ),
            stream=StreamSpec(
                compute_dtype="bfloat16",
                lora_rank=8,
                lora_alpha=16.0,
                lora_target_modules="all-linear",
                pin_host_memory=True,
                cuda_graphs=True,
            ),
            qlora_autocast=True,
            qlora_fused_optimizer=True,
            qlora_joint_cuda_graph=True,
            thermal_startup_max_c=55.0,
            thermal_cpu_startup_max_c=70.0,
            thermal_startup_dwell_seconds=5.0,
            thermal_startup_timeout_seconds=300.0,
        )
        spec = apply_profile(spec, profile_name)

    # CLI overrides take precedence over the saved configuration.
    if args.batch_size is not None:
        spec = replace(spec, batch_size=args.batch_size)
    if args.learning_rate is not None:
        spec = replace(spec, learning_rate=args.learning_rate)
    if args.context_length is not None:
        spec = replace(spec, data=replace(spec.data, context_length=args.context_length),
                       model=replace(spec.model, context_length=args.context_length))

    # Overrides if explicitly provided on command line
    if args.seed is not None:
        spec = replace(spec, seed=args.seed)
    if args.steps is not None:
        spec = replace(spec, max_steps=args.steps)
    if args.profile is not None:
        spec = apply_profile(spec, args.profile)

    # Execution Mode (Normal vs Prioritize)
    selected_mode = _resolve_execution_mode(args.mode_select, ui)
    spec.validate()
    # Pre-Flight Card
    hw = get_hardware_info()
    if ui.enabled:
        ui.card("MOLT Pre-Flight Configuration", [
            ("Mode", spec.mode.upper()),
            ("Model", Path(spec.base_model).name if spec.base_model else "SLM"),
            ("Dataset", Path(spec.data.path).name),
            ("Max Steps", spec.max_steps),
            ("Context Length", spec.data.context_length),
            ("Batch Size", f"{spec.batch_size} × {spec.gradient_accumulation}"),
            ("Thermal Target", f"{spec.thermal_target_c:.1f}°C (pause: {int(spec.thermal_pause_seconds*1000)}ms)"),
            ("Thermal Policy", spec.thermal_control_mode),
            ("Thermal Abort", f"{spec.thermal_abort_c:.1f}°C"),
            ("Start Cooling", (
                f"GPU ≤ {spec.thermal_startup_max_c}°C; CPU ≤ {spec.thermal_cpu_startup_max_c}°C"
                if spec.thermal_startup_max_c is not None or spec.thermal_cpu_startup_max_c is not None
                else "Not configured"
            )),
            ("GPU Detected", hw["gpu_name"] or "CUDA Device"),
            ("Current CPU Temp", f"{hw['cpu_temperature_c']}°C" if hw["cpu_temperature_c"] is not None else "Sensor unavailable"),
            ("VRAM requirement", "Measured during training; fit not guaranteed"),
        ])

    if args.dry_run:
        if ui.enabled:
            ui.check("Dry-run complete. Training specification verified successfully.")
        else:
            _print({"status": "dry_run_success", "spec": spec.to_dict()})
        return 0

    if not _verify_cuda_or_prompt_install(ui, spec.stream.device):
        return 0

    if not args.yes and ui.enabled and sys.stdin.isatty():
        if not ui.confirm("Run the safety check and begin training?", default=True):
            print("[MOLT] Training cancelled by user.")
            return 0

    with prioritized_execution(selected_mode == "prioritize") as elevated:
        if ui.enabled and selected_mode == "prioritize":
            ui.check("Temporary MOLT priority enabled" if elevated else "Using standard process priority")
        if guided_spec:
            ui.check("Running a two-step model, memory, and optimizer safety check")
            fit_spec = replace(
                spec,
                max_steps=2,
                qlora_validation_batches=min(4, spec.qlora_validation_batches),
                artifacts_dir=str(Path(spec.artifacts_dir) / "fit-tests"),
            )
            fit_path = train(fit_spec, use_galore=args.galore, progress=ui.progress if ui.enabled else None)
            fit_summary = json.loads((fit_path / "metrics.summary.json").read_text("utf-8"))
            if fit_summary.get("state") != "completed":
                raise MoltError("The automatic fit test did not complete; full training was not started")
            ui.finish_progress()
            ui.check("Safety check passed; starting the full run")
        path = train(
            spec,
            use_galore=args.galore,
            progress=ui.progress if ui.enabled else None,
            interrupt_decision=(lambda: _interactive_interrupt_decision(ui))
            if ui.enabled and sys.stdin.isatty() else None,
        )
    ui.finish_progress()
    summary = json.loads((path / "metrics.summary.json").read_text("utf-8"))
    title = "Training complete" if summary.get("state") == "completed" else "Training stopped"
    output_func({"run": str(path)}, title=title, rows=_report_rows(summary, path))
    return 0 if summary.get("state") == "completed" else 1


def handle_guided_resume(args: argparse.Namespace, ui: TerminalUI, output_func: Any) -> int:
    """Interactive guided resume flow."""
    from molt_stream.training.engine import train

    run_path = args.run
    if not run_path:
        runs = [run for run in find_runs() if run["state"] != "completed" and run["checkpoint"]]
        if not runs:
            raise MoltError("No prior training runs detected in workspace to resume.")
        if ui.enabled:
            shown = runs[:8]
            selected = ui.select(
                "Select a run to resume",
                [
                    (
                        str(run["run_id"]),
                        f"Step {run['step']} · {run['tokens']:,} tokens · {run['integrity']}",
                    )
                    for run in shown
                ] + [("Choose another run…", "")],
            )
            if selected is not None and selected < len(shown):
                run_path = shown[selected]["path"]
            else:
                run_path = input(ui._style("Enter run directory path: ", ui.GREEN)).strip()
        else:
            run_path = runs[0]["path"]

    root = Path(run_path)
    spec = load_spec(root / "spec.resolved.json")
    selected_mode = _resolve_execution_mode(args.mode_select, ui)

    if ui.enabled:
        ui.check(f"Resuming run from {root.name}")

    if not _verify_cuda_or_prompt_install(ui, spec.stream.device):
        return 0

    with prioritized_execution(selected_mode == "prioritize") as elevated:
        if ui.enabled and selected_mode == "prioritize":
            ui.check("Temporary MOLT priority enabled" if elevated else "Using standard process priority")
        path = train(
            spec,
            resume=root,
            use_galore=args.galore,
            progress=ui.progress if ui.enabled else None,
            interrupt_decision=(lambda: _interactive_interrupt_decision(ui))
            if ui.enabled and sys.stdin.isatty() else None,
        )
    ui.finish_progress()
    summary = json.loads((path / "metrics.summary.json").read_text("utf-8"))
    title = "Training resumed" if summary.get("state") == "completed" else "Training stopped"
    output_func({"run": str(path)}, title=title, rows=_report_rows(summary, path))
    return 0 if summary.get("state") == "completed" else 1


def handle_benchmark_cmd(args: argparse.Namespace, ui: TerminalUI, output_func: Any) -> int:
    """Execute smoke or throughput benchmarks."""
    from molt_stream.training.throughput import training_throughput_benchmark as benchmark_stream_throughput
    from molt_stream.training.engine import train

    if args.smoke or not args.config:
        if ui.enabled:
            ui.card("MOLT Smoke Benchmark", [
                ("Mode", "Pretrain Smoke Test"),
                ("Steps", "20"),
                ("Context", "32"),
                ("Batch", "8"),
                ("Device", "CUDA"),
            ])
        from molt_stream.training.smoke import create_smoke_spec
        spec = create_smoke_spec(Path("artifacts/molt-stream/smoke"))
        t0 = time.perf_counter()
        res = train(spec, progress=ui.progress if ui.enabled else None)
        ui.finish_progress()
        duration = time.perf_counter() - t0
        summary_path = res / "metrics.summary.json"
        summary = json.loads(summary_path.read_text("utf-8"))
        rate = summary.get("session_tokens", 0) / duration if duration else 0.0
        rows = [
            ("State", "PASS" if summary.get("state") == "completed" else "INCOMPLETE"),
            ("Workload", "Synthetic random tokens; not a model-quality benchmark"),
            ("Run", str(res)),
            ("Duration", f"{duration:.2f}s"),
            ("Throughput", f"{rate:,.1f} tok/s"),
            ("Peak VRAM", _bytes(summary.get("cuda_peak_allocated_bytes"))),
            ("Board Energy", f"{summary['telemetry']['gpu_board_energy_joules']:.2f} J"
             if summary.get("telemetry", {}).get("gpu_board_energy_joules") is not None else "Unavailable"),
        ]
        output_func(summary, title="Benchmark Result", rows=rows)
        return 0 if summary.get("state") == "completed" else 1

    spec = load_spec(args.config)
    value = benchmark_stream_throughput(spec, warmup_steps=args.warmup_steps, measured_steps=args.steps)
    rows = [
        ("Throughput", f"{(value.get('sustained_tokens_per_second') or 0):,.2f} tok/s"),
        ("Duration", f"{value['total_seconds_including_setup_and_warmup']:.2f}s"),
    ]
    output_func(value, title="Sustained Throughput Benchmark", rows=rows)
    return 1 if value.get("thermal_abort") else 0


def handle_guided_prepare(ui: TerminalUI, output_func: Any) -> int:
    """Prepare a common text/record dataset without exposing token-binary details."""
    source = input("Data file (.txt, .jsonl, or .parquet): ").strip().strip('"')
    model = input("Local model directory: ").strip().strip('"')
    if not source or not model:
        raise ValueError("Both a data file and local model directory are required")
    default_output = str(Path("molt-workspace/datasets") / Path(source).stem)
    destination = input(f"Output directory [{default_output}]: ").strip().strip('"') or default_output
    if Path(source).suffix.lower() in {".jsonl", ".parquet"}:
        from molt_stream.data.records import prepare_records
        result = prepare_records(source, model, destination, base_model=model)
    else:
        from molt_stream.data.text import prepare_text
        result = prepare_text(source, model, destination, base_model=model)
    if ui.enabled:
        ui.check("Training and validation data prepared")
    output_func(result, title="Dataset ready", rows=[
        ("Directory", result["directory"]),
        ("Training tokens", result["train_tokens"]),
        ("Validation tokens", result["validation_tokens"]),
        ("Next", f"molt fit-test --config {result['training_config']}"),
    ])
    return 0


def guided_landing(ui: TerminalUI) -> int:
    """Interactive landing menu for users launching bare 'molt'."""
    hw = get_hardware_info()
    gpu_label = hw["gpu_name"] or "CPU Mode"
    vram_label = f"{hw['gpu_vram_total_gb']} GB" if hw["gpu_vram_total_gb"] else ""

    ui.card("MOLT AI Infrastructure", [
        ("Version", f"v{__version__} (Alpha)"),
        ("Hardware", f"{gpu_label} {vram_label}".strip()),
        ("Temperatures", f"GPU {hw['gpu_temperature_c'] if hw['gpu_temperature_c'] is not None else '—'}°C · CPU {hw['cpu_temperature_c'] if hw['cpu_temperature_c'] is not None else 'sensor unavailable'}"),
        ("Recommended Profile", hw["suggested_profile"]),
        ("Architecture", "Dual-Gear Thermal Control • 4-bit NF4 QLoRA • Memory-Mapped I/O"),
    ])

    choice = ui.select(
        "Select an action",
        [
            ("Train", "Choose a model and dataset, then start safely"),
            ("Resume", "Continue from a verified checkpoint"),
            ("Benchmark", "Run a synthetic hardware smoke test"),
            ("Doctor", "Check installation and hardware"),
            ("Prepare Data", "Convert text, JSONL, or Parquet"),
            ("Configuration", "Initialize workspace and list assets"),
            ("Exit", ""),
        ],
    )

    dummy_args = argparse.Namespace(
        config=None, model=None, dataset=None, profile=None, steps=None,
        batch_size=None, context_length=None, learning_rate=None, seed=None,
        galore=False, mode_select=None, dry_run=False, yes=False, smoke=True,
        init=False, list=True, run=None,
    )

    def output_func(v: Any, title: str = "Result", rows: Any = None):
        if ui.enabled:
            ui.card(title, rows if rows is not None else [("Result", json.dumps(v, default=str))])
        else:
            _print(v)

    if choice == 0:
        return handle_guided_train(dummy_args, ui, output_func)
    elif choice == 1:
        return handle_guided_resume(dummy_args, ui, output_func)
    elif choice == 2:
        return handle_benchmark_cmd(dummy_args, ui, output_func)
    elif choice == 3:
        return handle_doctor(ui, output_func)
    elif choice == 4:
        return handle_guided_prepare(ui, output_func)
    elif choice == 5:
        return handle_config(dummy_args, ui, output_func)
    elif choice == 6 or choice is None:
        print("[MOLT] Exiting.")
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    if args.json and args.ui:
        parser.error("--json and --ui cannot be combined")
    if args.command == "research":
        args.command = args.research_command

    ui = TerminalUI(
        enabled=args.ui or (not args.json and sys.stdout.isatty()),
        color=not args.no_color,
    )
    if ui.enabled:
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
        # Guided actions share the same error boundary as explicit commands.
        if args.command is None:
            return guided_landing(ui)
        if args.command == "doctor":
            return handle_doctor(ui, output)
        elif args.command == "setup":
            return handle_setup(args, ui, output)
        elif args.command in {"update", "repair", "uninstall"}:
            return handle_runtime_management(args, ui, output)
        elif args.command == "runs":
            output(find_runs(), title="Saved runs")
        elif args.command == "export":
            from molt_stream.experiments.export import export_run
            output(export_run(args.run, args.output_dir, format=args.format,
                              llama_cpp=args.llama_cpp), title="Export complete")
        elif args.command == "fit-test":
            from molt_stream.training.engine import train
            spec = load_spec(args.config)
            spec = replace(spec, max_steps=2, artifacts_dir=str(Path(spec.artifacts_dir) / "fit-tests"))
            root = train(spec, progress=ui.progress if ui.enabled else None)
            summary = json.loads((root / "metrics.summary.json").read_text("utf-8"))
            output({"run": str(root), "state": summary["state"],
                    "scope": "Two-step geometry check only; not sustained reliability"}, title="Fit test")
            return 0 if summary["state"] == "completed" else 1
        elif args.command == "info":
            return handle_info(ui, output)
        elif args.command == "config":
            return handle_config(args, ui, output)
        elif args.command == "inspect":
            from molt_stream.training.stream_benchmark import inspect_capabilities
            value = inspect_capabilities()
            output(value, title="Workspace info", rows=[(k, v) for k, v in value.items()])
        elif args.command == "prepare":
            source = args.source or args.text_file
            tokenizer = args.tokenizer or args.model
            base_model = args.base_model or args.model
            if source:
                if args.config or not tokenizer or not args.output_dir:
                    raise ValueError("Preparation needs SOURCE, --model (or --tokenizer), and --output")
                if Path(source).suffix.lower() in {".jsonl", ".parquet"}:
                    from molt_stream.data.records import prepare_records
                    value = prepare_records(
                        source, tokenizer, args.output_dir, args.validation_fraction, base_model,
                        schema=args.schema, text_column=args.text_column,
                        messages_column=args.messages_column, prompt_column=args.prompt_column,
                        completion_column=args.completion_column,
                    )
                else:
                    from molt_stream.data.text import prepare_text
                    value = prepare_text(source, tokenizer, args.output_dir,
                                         args.validation_fraction, base_model)
                output(value, title="Dataset prepared")
                return 0
            if not args.config:
                raise ValueError("Provide --config, or SOURCE with --model and --output")
            spec = load_spec(args.config)
            spec.validate()
            value = {"path": spec.data.path, "bytes": Path(spec.data.path).stat().st_size, "storage_backend": "mmap", "validated": True}
            if ui.enabled:
                ui.check("Memory-mapped dataset initialized")
            output(value, title="Dataset info", rows=[("State", "ready"), ("Path", value["path"]), ("Size", _bytes(value["bytes"])), ("Backend", "OS mmap")])
        elif args.command == "train":
            return handle_guided_train(args, ui, output)
        elif args.command == "optimize-gpu":
            from molt_stream.measurement.gpu_profile import (
                PROFILES as GPU_PROFILES,
                temporary_graphics_clock,
                verify_measured_clock_profile,
            )
            from molt_stream.training.engine import train

            if not args.yes:
                if not ui.enabled or not sys.stdin.isatty():
                    raise ValueError("Temporary GPU clock changes require -y in non-interactive mode")
                if not ui.confirm(
                    "Temporarily lock GPU clocks to 1500-1650 MHz for this run?",
                    default=False,
                ):
                    return 0
            spec = load_spec(args.config)
            profile = GPU_PROFILES[args.profile]
            with temporary_graphics_clock(profile):
                run = train(spec, progress=ui.progress if ui.enabled else None)
            summary = json.loads((run / "metrics.summary.json").read_text("utf-8"))
            verification = verify_measured_clock_profile(summary.get("telemetry", {}), profile)
            value = {"run": str(run), "clock_profile": verification}
            output(value, title="GPU-optimized training complete")
            return 0 if verification["verified"] else 1
        elif args.command == "resume":
            return handle_guided_resume(args, ui, output)
        elif args.command == "benchmark":
            return handle_benchmark_cmd(args, ui, output)
        elif args.command == "evaluate":
            from molt_stream.training.engine import evaluate_run
            value = evaluate_run(args.run)
            output(value, title="Evaluation", rows=[("State", "complete"), ("Validation NLL", value["validation_nll"]), ("Perplexity", value["validation_perplexity"])])
        elif args.command == "report":
            value = json.loads((Path(args.run) / "metrics.summary.json").read_text("utf-8"))
            output(value, title="Training report", rows=_report_rows(value, Path(args.run)))
        elif args.command == "compare":
            from molt_stream.measurement.comparison import compare_runs
            value = compare_runs(
                args.baseline_run,
                args.candidate_run,
                quality_tolerance_percent=args.quality_tolerance_percent,
                minimum_improvement_percent=args.minimum_improvement_percent,
                thermal_peak_tolerance_c=args.thermal_peak_tolerance_c,
            )
            if args.output:
                from molt_stream.measurement.comparison import write_comparison
                write_comparison(args.output, value)
            output(value, title="Run Comparison")
        elif args.command == "compare-paired":
            from molt_stream.measurement.comparison import aggregate_comparisons, compare_runs
            comparisons = [compare_runs(a, b,
                quality_tolerance_percent=args.quality_tolerance_percent,
                minimum_improvement_percent=args.minimum_improvement_percent,
                thermal_peak_tolerance_c=args.thermal_peak_tolerance_c) for a, b in args.pair]
            value = aggregate_comparisons(
                comparisons,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
            if args.output:
                from molt_stream.measurement.comparison import write_comparison
                write_comparison(args.output, value)
            output(value, title="Paired Comparison")
        elif args.command == "frontier":
            from molt_stream.measurement.frontier import build_frontier
            value = build_frontier(args.runs)
            output(value, title="Frontier")
        elif args.command == "generate":
            from molt_stream.training.engine import generate_run
            if args.prompt is not None:
                from molt_stream.training.text import generate_text
                value = generate_text(args.run, args.prompt, args.tokenizer, args.max_new_tokens)
                output(value, title="Generated Text")
                return 0
            tokens = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
            if not tokens or args.max_new_tokens <= 0:
                raise ValueError("Provide prompt IDs and positive max_new_tokens")
            value = generate_run(args.run, tokens, max_new_tokens=args.max_new_tokens)
            output(value, title="Generated Tokens")
        elif args.command == "stream-tune":
            from molt_stream.training.stream_benchmark import stream_tune
            value = stream_tune(
                width=args.width, layers=args.layers, sequence=args.sequence,
                batch=args.batch, steps=args.steps, rank=args.rank,
                bundle_size=args.bundle_size, seed=args.seed, double_buffer=not args.synchronous,
            )
            output(value, title="Stream Tune")
        elif args.command == "fusion-benchmark":
            from molt_stream.training.throughput import fusion_memory_benchmark
            value = fusion_memory_benchmark(load_spec(args.config))
            output(value, title="Fusion Benchmark")
        elif args.command == "curriculum-benchmark":
            from molt_stream.training.curriculum import run_curriculum_experiment
            seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
            value = run_curriculum_experiment(load_spec(args.config), seeds=tuple(seeds), eval_interval=args.eval_interval)
            output(value, title="Curriculum Benchmark")
        elif args.command == "loss-partition-benchmark":
            from molt_stream.training.loss_partition import benchmark_exact_loss_partitioning
            chunks = [int(x.strip()) for x in args.chunks.split(",") if x.strip()]
            value = benchmark_exact_loss_partitioning(load_spec(args.config), chunk_sizes=tuple(chunks), warmup_steps=args.warmup_steps, measured_steps=args.steps)
            output(value, title="Loss Partition Benchmark")
        elif args.command == "power-limit":
            from molt_stream.measurement.telemetry import manage_power_limit
            value = manage_power_limit(args.watts, apply=args.apply)
            output(value, title="Power Limit")
            return 1 if args.apply and not value.get("verified") else 0
        elif args.command == "qlora-benchmark":
            from molt_stream.training.qlora import benchmark_qlora_adapter
            spec = load_spec(Path(args.run) / "spec.resolved.json")
            if spec.mode != "qlora":
                raise ValueError("qlora-benchmark requires a QLoRA run")
            value = benchmark_qlora_adapter(spec, args.run, batches=args.batches, split=args.split)
            output(value, title="QLoRA Benchmark")
        return 0

    except KeyboardInterrupt:
        print("\n[MOLT] Operation cancelled by user.")
        return 130
    except Exception as e:
        if args.debug or "--debug" in sys.argv:
            raise
        if ui.enabled:
            ui.card("⚠️ MOLT Error", [
                ("Type", type(e).__name__),
                ("Message", str(e)),
                ("Tip", "Run with --debug for full traceback or consult docs/cli.md"),
            ])
        else:
            print(f"Error: {e}", file=sys.stderr)
            print("Run with --debug for complete traceback.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
