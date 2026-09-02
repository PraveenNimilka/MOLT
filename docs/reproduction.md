# MOLT reproduction guide

## 1. Environment

Requirements: Windows or Linux, Python 3.12, and `uv`. CUDA profiles additionally
require a compatible NVIDIA driver and GPU. From the repository root:

```powershell
uv sync
.\.venv\Scripts\molt.exe --json inspect
.\.venv\Scripts\python.exe -m pytest -q
```

The lockfile pins the resolved environment. Save the `inspect` JSON with every
external reproduction report.

## 2. CPU smoke reproduction

Smoke token files are generated artifacts and may be absent in a fresh clone.
Create an aligned `int32` token file first, or point a copied config at an
existing prepared corpus. Then use an active `molt-stream-*.json` profile:

```powershell
.\.venv\Scripts\molt.exe prepare --config configs\molt-stream-smoke.json
.\.venv\Scripts\molt.exe train --config configs\molt-stream-smoke.json
.\.venv\Scripts\molt.exe report --run artifacts\molt-stream\runs\RUN_ID
.\.venv\Scripts\molt.exe evaluate --run artifacts\molt-stream\runs\RUN_ID
```

Do not replace missing CUDA, lower precision, smaller data, or shorter training
silently. Copy the config, change it explicitly, and preserve that resolved spec
with the result.

## 3. CUDA throughput benchmark

```powershell
.\.venv\Scripts\molt.exe benchmark `
  --config configs\molt-stream-production.json `
  --warmup-steps 10 --steps 50
```

Report end-to-end time separately from the steady-state benchmark. Include GPU
model, driver/runtime versions, enforced power limit, temperature, throttle
flags, peak VRAM, warm-up count, and all measured trials—not only the best run.

## 4. Milestone 3 experiments

```powershell
.\.venv\Scripts\molt.exe --json curriculum-benchmark `
  --config configs\molt-stream-milestone3.json `
  --seeds 1337,2027,4099 --eval-interval 15

.\.venv\Scripts\molt.exe --json loss-partition-benchmark `
  --config configs\molt-stream-milestone3.json `
  --chunks 1024,2048,4096,8192 --warmup-steps 2 --steps 5
```

These commands are falsification tools. Their generated decision and raw
measurements must be kept even when the candidate is rejected.

## 5. Qwen LoRA+ holdout reproduction

Install the optional QLoRA runtime and make the local Qwen checkpoint and
prepared token files available at the paths in copied configs. Alternate arm
order and wait for the GPU to return to the same idle-temperature band before
each run.

```powershell
uv sync --extra qlora
$env:MOLT_QWEN_MODEL = "D:\path\to\your\Qwen2-0.5B-checkpoint"

uv run molt --json train `
  --config configs/molt-stream-qwen2-0.5b-qlora-loraplus.json --seed 8111
uv run molt --json train `
  --config configs/molt-stream-qwen2-0.5b-qlora-time-target-baseline.json --seed 8111

uv run molt --json compare BASELINE_RUN CANDIDATE_RUN `
  --quality-tolerance-percent 1 --minimum-improvement-percent 1 `
  --output artifacts/qwen2-0.5b/comparisons/reproduction.json
```

Repeat with seeds 12143 and 16381, reversing order for the middle pair. Do not
substitute the tuning seeds 1337, 2027, and 4099 as holdout evidence. The
comparison must report no controlled mismatch and every gate must be `true`.

Aggregate the three immutable pairs with deterministic bootstrap resampling:

```powershell
uv run molt --json compare-paired `
  --pair BASELINE_8111 CANDIDATE_8111 `
  --pair BASELINE_12143 CANDIDATE_12143 `
  --pair BASELINE_16381 CANDIDATE_16381 `
  --bootstrap-samples 10000 --bootstrap-seed 20260902 `
  --output artifacts/qwen2-0.5b/comparisons/reproduction-aggregate.json
```

## 6. Checkpoint interruption test

Resume a thermally aborted run, or a copied run directory containing a valid
incomplete checkpoint. MOLT validates the SHA-256 manifest, restores model,
optimizer, data cursor, and RNG state, and refuses to resume an already completed
run. The current engine checkpoints on completion or thermal abort; it does not
yet claim arbitrary power-loss recovery before the first save. Corrupt a copy—not
the original—to verify fallback to the previous complete checkpoint generation.

## 7. Reproduction record

Archive these together:

- source commit and dirty diff;
- original and resolved configuration;
- `molt --json inspect` output;
- token-file identity or manifest;
- checkpoint plus `.complete.json` digest manifest;
- `metrics.summary.json` and experiment artifacts;
- every seed/trial, failure, and deviation from the registered procedure.

Historical `ai_local` commands in dated result documents describe the earlier
prototype and are not part of the current executable interface.
