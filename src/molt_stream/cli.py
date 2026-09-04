"""MOLT unified command-line and interactive console interface.

Provides hardware-aware AI training with:
- Zero-configuration interactive landing and guided training flows.
- High-level policy profiles (SPEED, BALANCED, COOL, ENERGY).
- Automatic workspace and hardware discovery.
- Atomic checkpoint resumption and empirical benchmarking.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import replace, asdict
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
from molt_stream.core.profiles import PROFILES, apply_profile, get_profile
from molt_stream.core.system_tuning import prioritized_execution
from molt_stream.experiments.store import sha256


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
        line = (
            f"{bar} {fraction * 100:5.1f}%  ETA {_duration(eta)}  {speed}  "
            f"{loss}  {temp}  {cooling}  {vram}"
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
    initial_perplexity = initial.get("perplexity")
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
    ui.card(
        "Execution Mode",
        [
            ("1. Normal Mode", "Standard OS priority and background defaults (Recommended)"),
            ("2. Prioritized Mode", "Temporary MOLT priority; other apps and Defender unchanged"),
        ],
    )
    try:
        choice = input(ui._style("Select mode [1=Normal, 2=Prioritized] (default: 1): ", ui.GREEN)).strip()
        if choice == "2":
            return "prioritize"
    except (EOFError, KeyboardInterrupt):
        print()
    return "normal"


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

    # 1. info
    commands.add_parser("info", help="Display hardware diagnostics, VRAM, and suggested profiles")

    # 2. inspect (backward compatibility)
    commands.add_parser("inspect", help="Inspect runtime and optional acceleration capabilities")

    # 3. config
    config_cmd = commands.add_parser("config", help="Workspace and configuration management")
    config_cmd.add_argument("--init", action="store_true", help="Initialize standard molt-workspace/ directory")
    config_cmd.add_argument("--list", action="store_true", help="List discovered models, datasets, and configurations")

    # 4. prepare
    prepare = commands.add_parser("prepare", help="Validate an mmap-ready token file")
    prepare.add_argument("--config", required=True)

    # 5. train
    training = commands.add_parser("train", help="Run model training with hardware-aware thermal pacing")
    training.add_argument("--config", default=None, help="Path to JSON training configuration")
    training.add_argument("--model", default=None, help="Path or name of base model directory")
    training.add_argument("--dataset", default=None, help="Path to binary token dataset (.bin)")
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

    qlora_benchmark = commands.add_parser("qlora-benchmark", help="Compare a saved QLoRA adapter with local base model")
    qlora_benchmark.add_argument("--run", required=True)
    qlora_benchmark.add_argument("--batches", type=int, default=8)
    qlora_benchmark.add_argument("--split", choices=("train", "validation"), default="validation")

    return parser


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
        ("Power Limit", f"{hw['gpu_power_limit_w']} W" if hw["gpu_power_limit_w"] else "—"),
        ("GPU Driver", hw["gpu_driver"] or "—"),
        ("Suggested Profile", hw["suggested_profile"]),
        ("Memory Budget", f"{hw['suggested_memory_budget_gb']} GB"),
    ]
    output_func(hw, title="MOLT Hardware Diagnostics", rows=rows)
    return 0


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
            choice = input(ui._style("\nWould you like MOLT to install CUDA PyTorch now? [Y/n]: ", ui.GREEN)).strip().lower()
            if choice in ("", "y", "yes"):
                print("[MOLT] Installing CUDA-accelerated PyTorch... (downloading official PyTorch cu128 wheel)")
                import subprocess
                cmd = [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu128", "--force-reinstall"]
                res = subprocess.run(cmd)
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


def handle_guided_train(args: argparse.Namespace, ui: TerminalUI, output_func: Any) -> int:
    """Interactive guided training flow for users without full command arguments."""
    from molt_stream.training.engine import train

    spec: TrainingSpec
    if args.config:
        spec = load_spec(args.config)
    else:
        # 1. Select Model
        model_path = args.model
        if not model_path:
            models = find_models()
            if models and ui.enabled:
                print(ui._style("\n--- Detected Models ---", ui.BOLD))
                for idx, m in enumerate(models, start=1):
                    print(f"  {idx}. {m['name']} ({m['path']})")
                print(f"  {len(models) + 1}. Enter custom path...")
                choice = input(ui._style(f"Select a model [1-{len(models) + 1}] (default: 1): ", ui.GREEN)).strip()
                clean_choice = choice.strip('\'"')
                if clean_choice.isdigit() and 1 <= int(clean_choice) <= len(models):
                    model_path = models[int(clean_choice) - 1]["path"]
                elif clean_choice == str(len(models) + 1):
                    model_path = input(ui._style("Enter model directory path: ", ui.GREEN)).strip('\'"')
                elif clean_choice and (Path(clean_choice).exists() or "/" in clean_choice or "\\" in clean_choice):
                    model_path = clean_choice
                elif not clean_choice:
                    model_path = models[0]["path"]
                else:
                    model_path = clean_choice
            elif not models:
                model_path = input(ui._style("Enter base model directory path: ", ui.GREEN)).strip('\'"')
                if not model_path:
                    raise MoltError("No model specified. Place models in models/ or pass --model.")

        # 2. Select Dataset
        dataset_path = args.dataset
        if not dataset_path:
            datasets = find_datasets()
            if datasets and ui.enabled:
                print(ui._style("\n--- Detected Datasets ---", ui.BOLD))
                for idx, d in enumerate(datasets, start=1):
                    tokens_label = f"{d['estimated_tokens']:,} tokens" if d["estimated_tokens"] else "binary"
                    print(f"  {idx}. {d['name']} ({tokens_label}) [{d['path']}]")
                print(f"  {len(datasets) + 1}. Enter custom path...")
                choice = input(ui._style(f"Select a dataset [1-{len(datasets) + 1}] (default: 1): ", ui.GREEN)).strip()
                clean_choice = choice.strip('\'"')
                if clean_choice.isdigit() and 1 <= int(clean_choice) <= len(datasets):
                    dataset_path = datasets[int(clean_choice) - 1]["path"]
                elif clean_choice == str(len(datasets) + 1):
                    dataset_path = input(ui._style("Enter dataset path (.bin): ", ui.GREEN)).strip('\'"')
                elif clean_choice and (Path(clean_choice).exists() or "/" in clean_choice or "\\" in clean_choice):
                    dataset_path = clean_choice
                elif not clean_choice:
                    dataset_path = datasets[0]["path"]
                else:
                    dataset_path = clean_choice
            elif not datasets:
                dataset_path = input(ui._style("Enter dataset path (.bin): ", ui.GREEN)).strip('\'"')
                if not dataset_path:
                    raise MoltError("No dataset specified. Place token files in datasets/ or pass --dataset.")

        # 3. Select Profile
        profile_name = args.profile
        if not profile_name and ui.enabled:
            print(ui._style("\n--- Training Profiles ---", ui.BOLD))
            profiles_list = list(PROFILES.items())
            for idx, (k, p) in enumerate(profiles_list, start=1):
                rec = " (Recommended for Laptop)" if k == "balanced" else ""
                print(f"  {idx}. {p['name']}: {p['description']}{rec}")
            choice = input(ui._style("Select profile [1-4] (default: 2 - BALANCED): ", ui.GREEN)).strip()
            if choice.isdigit() and 1 <= int(choice) <= len(profiles_list):
                profile_name = profiles_list[int(choice) - 1][0]
            else:
                profile_name = "balanced"
        elif not profile_name:
            profile_name = "balanced"

        # Build spec
        ctx = args.context_length or 1024
        bs = args.batch_size or 1
        steps = args.steps or 878
        lr = args.learning_rate or 0.00015
        seed = args.seed or 2026

        spec = TrainingSpec(
            mode="qlora",
            base_model=str(Path(model_path).resolve()),
            artifacts_dir="runs",
            batch_size=bs,
            gradient_accumulation=1,
            learning_rate=lr,
            max_steps=steps,
            seed=seed,
            model=ModelSpec(context_length=ctx, layers=24, width=896, hidden_width=4864, heads=14, vocab_size=151936),
            data=DataSpec(
                path=str(Path(dataset_path).resolve()),
                validation_path=None,
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
            ),
        )
        spec = apply_profile(spec, profile_name)

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
            ("GPU Detected", hw["gpu_name"] or "CUDA Device"),
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
        proceed = input(ui._style("\nReady to begin training? [Y/n]: ", ui.GREEN)).strip().lower()
        if proceed and proceed not in ("y", "yes"):
            print("[MOLT] Training cancelled by user.")
            return 0

    with prioritized_execution(selected_mode == "prioritize") as elevated:
        if ui.enabled and selected_mode == "prioritize":
            ui.check("Temporary MOLT priority enabled" if elevated else "Using standard process priority")
        path = train(spec, use_galore=args.galore, progress=ui.progress if ui.enabled else None)
    ui.finish_progress()
    summary = json.loads((path / "metrics.summary.json").read_text("utf-8"))
    title = "Training stopped" if summary.get("state") == "thermal_abort" else "Training complete"
    output_func({"run": str(path)}, title=title, rows=_report_rows(summary, path))
    return 0


def handle_guided_resume(args: argparse.Namespace, ui: TerminalUI, output_func: Any) -> int:
    """Interactive guided resume flow."""
    from molt_stream.training.engine import train

    run_path = args.run
    if not run_path:
        runs = find_runs()
        if not runs:
            raise MoltError("No prior training runs detected in workspace to resume.")
        if ui.enabled:
            print(ui._style("\n--- Prior Training Runs ---", ui.BOLD))
            for idx, r in enumerate(runs[:8], start=1):
                ckpt_status = f"Checkpoint: {r['integrity']}" if r["checkpoint"] else "No checkpoint"
                print(f"  {idx}. {r['run_id']} │ Step {r['step']} ({r['tokens']:,} tok) │ {ckpt_status}")
            print(f"  {min(len(runs), 8) + 1}. Enter custom run path...")
            choice = input(ui._style(f"Select a run to resume [1-{min(len(runs), 8) + 1}] (default: 1): ", ui.GREEN)).strip()
            if choice.isdigit() and 1 <= int(choice) <= min(len(runs), 8):
                run_path = runs[int(choice) - 1]["path"]
            elif choice == str(min(len(runs), 8) + 1):
                run_path = input(ui._style("Enter run directory path: ", ui.GREEN)).strip()
            else:
                run_path = runs[0]["path"]
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
        path = train(spec, resume=root, use_galore=args.galore, progress=ui.progress if ui.enabled else None)
    ui.finish_progress()
    summary = json.loads((path / "metrics.summary.json").read_text("utf-8"))
    title = "Training stopped" if summary.get("state") == "thermal_abort" else "Training resumed"
    output_func({"run": str(path)}, title=title, rows=_report_rows(summary, path))
    return 0


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
        smoke_config = Path("configs/molt-stream-smoke.json")
        if smoke_config.exists():
            spec = load_spec(smoke_config)
        else:
            spec = TrainingSpec(
                mode="pretrain",
                data=DataSpec(path="data/prepared/smoke/train.bin", context_length=32, storage_dtype="uint8"),
                model=ModelSpec(context_length=32, layers=2, width=128, heads=4, hidden_width=384, vocab_size=256),
                stream=StreamSpec(device="cuda", compute_dtype="bfloat16"),
                batch_size=8,
                gradient_accumulation=1,
                max_steps=20,
                learning_rate=0.001,
                seed=1337,
                artifacts_dir="artifacts/molt-stream/runs",
            )
        t0 = time.perf_counter()
        res = train(spec, progress=ui.progress if ui.enabled else None)
        ui.finish_progress()
        duration = time.perf_counter() - t0
        summary_path = res / "metrics.summary.json"
        summary = json.loads(summary_path.read_text("utf-8"))
        rate = summary.get("session_tokens", 0) / duration if duration else 0.0
        rows = [
            ("State", "PASS"),
            ("Duration", f"{duration:.2f}s"),
            ("Throughput", f"{rate:,.1f} tok/s"),
            ("Peak VRAM", _bytes(summary.get("cuda_peak_allocated_bytes"))),
            ("Board Energy", f"{summary.get('telemetry', {}).get('gpu_board_energy_joules', 0.0):.2f} J"),
        ]
        output_func(summary, title="Benchmark Result", rows=rows)
        return 0

    spec = load_spec(args.config)
    value = benchmark_stream_throughput(spec, warmup_steps=args.warmup_steps, steps=args.steps)
    rows = [
        ("Throughput", f"{value['tokens_per_second']:,.2f} tok/s"),
        ("Active Throughput", f"{value['active_tokens_per_second']:,.2f} tok/s"),
        ("Duration", f"{value['total_seconds']:.2f}s"),
    ]
    output_func(value, title="Sustained Throughput Benchmark", rows=rows)
    return 0


def guided_landing(ui: TerminalUI) -> int:
    """Interactive landing menu for users launching bare 'molt'."""
    hw = get_hardware_info()
    gpu_label = hw["gpu_name"] or "CPU Mode"
    vram_label = f"{hw['gpu_vram_total_gb']} GB" if hw["gpu_vram_total_gb"] else ""

    ui.card("MOLT AI Infrastructure", [
        ("Version", f"v{__version__} (Alpha)"),
        ("Hardware", f"{gpu_label} {vram_label}".strip()),
        ("Recommended Profile", hw["suggested_profile"]),
        ("Architecture", "Dual-Gear Thermal Control • 4-bit NF4 QLoRA • Memory-Mapped I/O"),
    ])

    print(ui._style("\nSelect an action:", ui.BOLD))
    print(ui._style("  1. Train", ui.WHITE) + "             Start a new training run")
    print(ui._style("  2. Resume", ui.WHITE) + "            Resume from a verified checkpoint")
    print(ui._style("  3. Benchmark", ui.WHITE) + "         Run a 2-second hardware smoke test")
    print(ui._style("  4. Hardware Info", ui.WHITE) + "     Inspect GPU, VRAM, and thermal sensors")
    print(ui._style("  5. Configuration", ui.WHITE) + "     Initialize workspace and list assets")
    print(ui._style("  6. Exit", ui.MUTED))

    try:
        choice = input(ui._style("\nOption [1-6]: ", ui.GREEN)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0

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

    if choice == "1":
        return handle_guided_train(dummy_args, ui, output_func)
    elif choice == "2":
        return handle_guided_resume(dummy_args, ui, output_func)
    elif choice == "3":
        return handle_benchmark_cmd(dummy_args, ui, output_func)
    elif choice == "4":
        return handle_info(ui, output_func)
    elif choice == "5":
        return handle_config(dummy_args, ui, output_func)
    elif choice == "6":
        print("[MOLT] Exiting.")
        return 0
    else:
        print(ui._style("Invalid selection.", ui.RED))
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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

    # Bare molt invocation: show landing menu
    if args.command is None:
        return guided_landing(ui)

    try:
        if args.command == "info":
            return handle_info(ui, output)
        elif args.command == "config":
            return handle_config(args, ui, output)
        elif args.command == "inspect":
            from molt_stream.training.stream_benchmark import inspect_capabilities
            value = inspect_capabilities()
            output(value, title="Workspace info", rows=[(k, v) for k, v in value.items()])
        elif args.command == "prepare":
            spec = load_spec(args.config)
            value = {"path": spec.data.path, "bytes": Path(spec.data.path).stat().st_size, "storage_backend": "mmap", "validated": True}
            if ui.enabled:
                ui.check("Memory-mapped dataset initialized")
            output(value, title="Dataset info", rows=[("State", "ready"), ("Path", value["path"]), ("Size", _bytes(value["bytes"])), ("Backend", "OS mmap")])
        elif args.command == "train":
            return handle_guided_train(args, ui, output)
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
            output(value, title="Run Comparison")
        elif args.command == "compare-paired":
            from molt_stream.measurement.comparison import aggregate_comparisons
            value = aggregate_comparisons(
                args.pair,
                quality_tolerance_percent=args.quality_tolerance_percent,
                minimum_improvement_percent=args.minimum_improvement_percent,
                thermal_peak_tolerance_c=args.thermal_peak_tolerance_c,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
            output(value, title="Paired Comparison")
        elif args.command == "frontier":
            from molt_stream.measurement.frontier import build_frontier
            value = build_frontier(args.runs)
            output(value, title="Frontier")
        elif args.command == "generate":
            from molt_stream.training.model import generate_tokens
            tokens = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
            value = generate_tokens(args.run, tokens, max_new_tokens=args.max_new_tokens)
            output(value, title="Generated Tokens")
        elif args.command == "stream-tune":
            from molt_stream.streaming.engine import benchmark_nf4_streaming
            value = benchmark_nf4_streaming(
                width=args.width, layers=args.layers, sequence_length=args.sequence,
                batch_size=args.batch, steps=args.steps, lora_rank=args.rank,
                bundle_size=args.bundle_size, seed=args.seed, synchronous=args.synchronous,
            )
            output(value, title="Stream Tune")
        elif args.command == "fusion-benchmark":
            from molt_stream.kernels.fused import benchmark_fusion_memory
            value = benchmark_fusion_memory(load_spec(args.config))
            output(value, title="Fusion Benchmark")
        elif args.command == "curriculum-benchmark":
            from molt_stream.training.curriculum import run_curriculum_benchmark
            seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
            value = run_curriculum_benchmark(load_spec(args.config), seeds=seeds, eval_interval=args.eval_interval)
            output(value, title="Curriculum Benchmark")
        elif args.command == "loss-partition-benchmark":
            from molt_stream.training.loss_partition import benchmark_loss_partitioning
            chunks = [int(x.strip()) for x in args.chunks.split(",") if x.strip()]
            value = benchmark_loss_partitioning(load_spec(args.config), chunk_sizes=chunks, warmup_steps=args.warmup_steps, steps=args.steps)
            output(value, title="Loss Partition Benchmark")
        elif args.command == "power-limit":
            from molt_stream.measurement.telemetry import query_power_limit, set_power_limit
            if args.apply:
                set_power_limit(args.watts)
            value = query_power_limit()
            output(value, title="Power Limit")
        elif args.command == "qlora-benchmark":
            from molt_stream.training.qlora import benchmark_saved_qlora
            value = benchmark_saved_qlora(args.run, batches=args.batches, split=args.split)
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
