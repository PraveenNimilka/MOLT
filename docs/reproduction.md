# Reproduction guide

## Prerequisites

- Windows 11 x64
- NVIDIA GPU with a driver compatible with the pinned CUDA 12.8 wheel
- `uv` 0.11 or later
- at least 8 GiB free for the environment and artifacts

## Environment and tests

```powershell
uv sync
uv run pytest
uv run ai-local inspect
```

The lockfile pins the resolved environment. Do not replace CUDA with CPU or
change precision after a failure; create a new config/run instead.

## Thermal cruise reproduction

Record the idle start temperature and close unrelated GPU workloads. Runs with
different start temperatures are not paired evidence.

```powershell
nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit,utilization.gpu --format=csv
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\molt.exe benchmark --config configs\molt-stream-max-throughput.json --warmup-steps 10 --steps 50
.\.venv\Scripts\molt.exe benchmark --config configs\molt-stream-max-throughput-eager-control.json --warmup-steps 25 --steps 60
```

The compiled command is expected to reproduce a negative result on the audited
laptop unless cooling or firmware behavior changes. Inspect `initial_gpu_temperature_c`,
`measured_interval_peak_temperature_c`, `late_to_early_throughput_ratio`, raw
`step_samples`, pause time, energy, and throttle gates in the generated JSON.
Do not compare only the fastest individual step.

## Zoned cruise candidate

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\molt.exe --ui train --config configs\molt-stream-cruise.json
.\.venv\Scripts\molt.exe --json benchmark --config configs\molt-stream-cruise.json --warmup-steps 10 --steps 30
.\.venv\Scripts\molt.exe --json benchmark --config configs\molt-stream-cruise-safe.json --warmup-steps 100 --steps 200
.\.venv\Scripts\molt.exe --json benchmark --config configs\molt-stream-cruise-steady.json --warmup-steps 100 --steps 200
```

The UI displays `[⚡ FULL SPEED]`, `[❄ COOLING Nms]`, or
`[❄ PROTECT Nms]`. A safety stop displays `[ THERMAL_ABORT ]`; the run remains
resumable through `checkpoint.pt` plus `checkpoint.complete.json`. The config is
500 steps, so use the 30-step benchmark as the cheap gate before committing to a
full training run.

The exact 75 C profile is a research control, not the recommended unattended
setting. The steady-duty profile avoided thermal clock events in its measured
run but did not reach 80,000 aggregate tokens/s. Reproduce in randomized order
from comparable idle temperatures before drawing hardware conclusions.

## Smoke reproduction

```powershell
uv run ai-local prepare --config configs/smoke-cuda.json
uv run ai-local train --config configs/smoke-cuda.json
uv run ai-local report --run RUN_ID
uv run ai-local evaluate --run RUN_ID
```

Run directories contain the resolved spec, environment, append-only events,
telemetry samples, status, summary, and checkpoint files. `artifacts/` is ignored
by Git because checkpoints can be multi-gigabyte.

## One-billion-parameter mechanism

Start with the one-step gate:

```powershell
uv run ai-local prepare --config configs/one-billion-mechanism.json
uv run ai-local train --config configs/one-billion-mechanism.json
```

The longer `configs/one-billion-under-10m.json` run writes a roughly 2 GiB
checkpoint and can sustain high GPU power for several minutes. Close unrelated
GPU workloads and ensure adequate cooling before reproducing it. Its result is a
synthetic memorization test, not useful pretraining.

## Real baseline

Place a pinned TinyStories text export at `data/raw/tinystories.txt`, record its
source revision and SHA-256, then run `configs/baseline.json`. This acquisition
has not been automated yet because floating dataset revisions are prohibited.
