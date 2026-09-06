# MOLT Command-Line Interface (CLI) Reference

MOLT provides both a **guided interactive menu** and **direct command-line flags** for advanced scripting.

## 0.10.0 alpha customer workflow

Use the checkout's `.venv\Scripts\molt.exe` if `molt doctor` shows an unexpected
global Python installation. `doctor` is the primary combined diagnostic view.

### Safe GPU endurance profile

From an Administrator PowerShell, run training with the measured temporary
graphics-clock range and guaranteed restoration:

```powershell
molt optimize-gpu --profile endurance --config configs\molt-qwen2.5-1.5b-endurance.json
```

The command requires explicit confirmation, or `-y` in automation. It refuses
non-elevated execution and verifies sampled clocks before returning success.
`info` and `inspect` retain their legacy output for compatibility.

```powershell
molt doctor --ui
molt config --init
molt runs
```

For preparation, replace the paths below with your local data and model. Nothing
is downloaded implicitly:

```powershell
molt prepare stories.jsonl --model models/my-model --output molt-workspace/datasets/stories
molt train --config molt-workspace/datasets/stories/training.json --dry-run
molt fit-test --config molt-workspace/datasets/stories/training.json
molt train --config molt-workspace/datasets/stories/training.json --ui
```

JSONL/Parquet preparation streams records, writes little-endian int32 tokens, and
uses a deterministic record-level validation tail so a document never crosses
the split. It recognizes `text`, `messages`, or `prompt` plus `completion`; column
names and schema can be overridden with the corresponding flags. Plain `.txt`
retains the older bounded-chunk tokenizer and token-tail split. Existing output
folders are refused. `--model` emits a conservative QLoRA starter config; inspect
it before a substantive run.

Use the actual run path printed by training in place of `RUN_DIRECTORY`:

```powershell
molt evaluate --run RUN_DIRECTORY
molt generate --run RUN_DIRECTORY --prompt "Once upon a time" --max-new-tokens 32
molt export --run RUN_DIRECTORY --output-dir exported-adapter
```

Scratch models require `--tokenizer` for text generation. `--prompt-ids` remains
available. QLoRA auto-export produces a standard PEFT safetensors adapter; scratch
auto-export produces a MOLT recovery bundle. Both verify the source checkpoint,
hash their exported files, refuse existing destinations, and exclude base weights
and data. Optional GGUF adapter conversion uses the official llama.cpp converter:

```powershell
molt export --run RUN_DIRECTORY --format gguf --llama-cpp C:\src\llama.cpp --output-dir exported-gguf
```

The GGUF output is a LoRA adapter, not a merged standalone model, and requires a
compatible GGUF base model.

`molt research --help` groups experimental commands; existing top-level names are
retained. CLI batch/context/learning-rate/seed/steps overrides take precedence
over configurations. Relative paths in existing configs retain working-directory
semantics; newly generated configs use absolute paths. Workspace discovery uses
the nearest `molt-workspace.json` before legacy directory conventions.

---

## 1. Guided Interactive Mode

Simply type:

```powershell
molt
```

MOLT inspects your hardware and displays an interactive landing menu:
```text
╭─ [ MOLT AI INFRASTRUCTURE ] ───────────────────────────────────────╮
│ Version       v0.9.2 (Alpha)                                       │
│ Hardware      NVIDIA GeForce RTX 4060 Laptop GPU 8.0 GB            │
│ Recommended   BALANCED                                             │
│ Pillars       Dual-Gear Thermals • 4-bit NF4 QLoRA • OS MMap        │
╰────────────────────────────────────────────────────────────────────╯

Select an action:
  1. Train             Start a new training run
  2. Resume            Resume from a verified checkpoint
  3. Benchmark         Run a synthetic hardware smoke test
  4. Hardware Info     Inspect GPU, VRAM, and thermal sensors
  5. Configuration     Initialize workspace and list assets
  6. Exit
```

---

## 2. Command Reference

### `molt train`
Start an AI model training run.

```powershell
# Interactive training (prompts for model, dataset, profile)
molt train

# Fast training with discovered assets and policy profile
molt train --model QWEN --dataset datasets/train.bin --profile balanced

# Training from an explicit JSON configuration
molt train --config configs/molt-stream-production.json

# Pre-flight validation without starting CUDA compute
molt train --config configs/molt-stream-smoke.json --dry-run

# Non-interactive automated execution
molt train --config configs/molt-stream-smoke.json -y
```

**Options:**
* `--config <path>`: Path to JSON configuration file.
* `--model <path>`: Path or name of base model directory.
* `--dataset <path>`: Path to binary token file (`.bin`).
* `--profile {speed,balanced,cool,energy}`: High-level thermal policy profile.
* `--steps <int>`: Override max optimization steps.
* `--batch-size <int>`: Micro-batch size.
* `--dry-run`: Validate configuration/data paths without compute; does not prove VRAM fit.
* `-y, --yes`: Skip confirmation prompt.
* `--debug`: Output complete Python tracebacks on error.

---

### `molt resume`
Resume an interrupted run from its cryptographic checkpoint.

```powershell
# Interactive resume (lists recent runs and verification status)
molt resume

# Resume a specific run directory
molt resume --run runs/20260903-113243-qlora-493938ec
```

---

### `molt info`
Display real-time hardware diagnostics and suggested training configurations.

```powershell
molt info
```

---

### `molt benchmark`
Measure sustained token throughput and verify training correctness.

```powershell
# Fast 2-second non-destructive smoke benchmark
molt benchmark --smoke

# Sustained benchmark with custom config
molt benchmark --config configs/molt-stream-benchmark.json --steps 50
```

---

### `molt config`
Workspace asset discovery and initialization.

```powershell
# Initialize standard molt-workspace/ directory
molt config --init

# List detected models, datasets, and previous runs
molt config --list
```

---

## 3. High-Level Training Profiles

Profile names are pacing presets, not hardware-independent speed guarantees.

| Profile | Pacing Begins | Cooling Cadence | Intended Use Case |
| :--- | :---: | :---: | :--- |
| **`SPEED`** | `82.0°C` | `50ms` | Less pacing; verify the resolved abort boundary before running. |
| **`BALANCED`** | `74.0°C` | `220ms` | Dual-Gear pacing; not a guarantee of thermal equilibrium. |
| **`COOL`** | `68.0°C` | `300ms` | Conservative thermal limit for warm rooms or quiet fans. |
| **`ENERGY`** | `70.0°C` | `250ms` | Energy experiment preset; savings require measurement. |

## 4. Experimental QLoRA execution options

Two opt-in JSON fields are available for CUDA QLoRA. Neither changes existing
profiles by default:

```json
{
  "qlora_autocast": true,
  "qlora_fused_optimizer": true
}
```

These fields belong inside a complete training configuration, not a standalone
config. Autocast enables BF16 training matrix operations while loss reduction
and adapter parameters remain FP32. Fused AdamW requires FP32 CUDA trainable
parameters and keeps FP32 moment states. Evaluation disables autocast for a
common comparison precision; NF4 base computation still uses BF16. Precision
rounding can change updates, so compare validation quality before adopting it.

Saved QLoRA reports include `step_window_rates`; `step.intervals.json` records
training-step durations including evaluation and pacing. These window rates
exclude initial setup and final checkpoint writing; `tokens_per_second` remains
the broader run rate. Short runs do not prove sustained thermal stability.
