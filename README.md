# MOLT

MOLT is a local-first research platform for testing whether model training or
adaptation can reach useful quality with less time, energy, memory, cost, or
operational complexity on consumer hardware.

The repository now contains a working research prototype and measured CPU/CUDA
smoke runs. A dense 1,007,710,208-parameter BF16 model completed a narrow
synthetic memorization experiment and persisted a checkpoint in 256.73 seconds
on the audited RTX 4060 Laptop GPU. This is **not useful 1B pretraining**: it saw
only 16,384 repeated byte tokens at context length 8. No novelty or breakthrough
claim has been demonstrated.

## MOLT-Stream experimental engine

The isolated `molt_stream` package adds mmap-backed SLM pretraining, clean-room
NF4+LoRA streamed-layer mechanics, explicit Liger capability dispatch, a GaLore
reference optimizer, NVML thermal/energy telemetry, and SHA-256 atomic recovery.

```powershell
.\.venv\Scripts\python.exe -m molt_stream.cli inspect
.\.venv\Scripts\python.exe -m molt_stream.cli prepare --config configs\molt-stream-smoke.json
.\.venv\Scripts\python.exe -m molt_stream.cli train --config configs\molt-stream-smoke.json
.\.venv\Scripts\python.exe -m molt_stream.cli stream-tune --width 2048 --layers 8 --sequence 128 --steps 20
```

Interactive terminals use MOLT's minimalist black/green UI: verified startup
checkmarks, rounded information cards, inverted state badges, and a live slim
training bar with percent, ETA, tokens/s, loss, allocated VRAM, and sampled GPU
temperature. Redirected output remains JSON for scripts and CI.

```powershell
# Force the terminal UI (useful in captured terminals)
.\.venv\Scripts\molt.exe --ui inspect
.\.venv\Scripts\molt.exe --ui train --config configs\molt-stream-smoke.json

# Stable machine-readable output
.\.venv\Scripts\molt.exe --json inspect

# Retain the layout without color
.\.venv\Scripts\molt.exe --ui --no-color report --run RUN_ID
```

The host is not yet 8B QLoRA-ready, and double buffering was slower than its
synchronous control. See [the research record](docs/research/molt-stream-core.md)
for exact gate results.

## Quick start

```powershell
uv sync
.\.venv\Scripts\molt.exe inspect
.\.venv\Scripts\molt.exe prepare --config configs\molt-stream-smoke.json
.\.venv\Scripts\molt.exe train --config configs\molt-stream-smoke.json
.\.venv\Scripts\molt.exe benchmark --config configs\molt-stream-production.json
.\.venv\Scripts\molt.exe report --run RUN_ID
.\.venv\Scripts\molt.exe evaluate --run RUN_ID
.\.venv\Scripts\molt.exe generate --run RUN_ID --prompt-ids 1,2,3
```

Other commands include `resume`, `frontier`, `stream-tune`,
`curriculum-benchmark`, and `loss-partition-benchmark`. Token corpora are streamed
to disk and memory-mapped during training, so corpus size is not limited by RAM.
The `tune` command records OOM candidates and preserves a configurable VRAM
headroom; its output is a throughput recommendation, not a quality claim.
Commands reject missing CUDA,
invalid configs, data hash drift, completed-run resume, and corrupted checkpoints
rather than silently falling back.

The former `ai_local` prototype and its tests are preserved in
`archive/ai_local-legacy-2026-09-02.zip`. Historical documents and artifacts
remain available as research evidence, but `ai-local` is no longer installed.

## Start here

- [Vision and scope](docs/vision-and-scope.md)
- [Repository audit](docs/audits/repository-audit-2026-09-01.md)
- [Hardware capability report](docs/audits/hardware-capability-2026-09-01.md)
- [Prior-art matrix](docs/research/prior-art-matrix.md)
- [Baseline specification](docs/baseline-spec.md)
- [Architecture proposal](docs/architecture.md)
- [Hypotheses and falsification experiments](docs/research/hypotheses.md)
- [2× efficiency research program and measured memory experiment](docs/research/efficiency-2x-program.md)
- [Rejected coupled-suffix telescoping method: mathematics and ablations](docs/research/coupled-suffix-telescoping.md)
- [Evidence-gated roadmap](docs/roadmap.md)
- [1B feasibility analysis](docs/research/one-billion-feasibility.md)
- [Prototype results](docs/results/2026-09-01-prototype.md)
- [Reproduction guide](docs/reproduction.md)
- [Machine-readable experiment registry](experiments/registry.json)

## Evidence vocabulary

Documents use four labels:

- **Fact**: directly observed here or supported by a cited primary source.
- **Reported**: a result claimed by prior work but not reproduced here.
- **Interpretation**: a reasoned conclusion from facts or reported results.
- **Hypothesis**: an unverified prediction that requires an experiment.

The project does not use “novel,” “breakthrough,” or comparative superiority
without a completed prior-art review, controlled experiments, multiple seeds,
raw artifacts, and reproducible statistical evidence.

The latest real-text milestone is [MOLT AI 50M](docs/results/2026-09-01-molt-ai-50m.md):
a 49.88M-parameter ByteLevel-BPE model trained for two full TinyStories epochs
in 225.26 seconds, reaching held-out perplexity 33.70 and readable story output.
The first controlled efficiency follow-up cut framework peak allocation by
49.992% at equivalent single-seed quality, but increased time by 20.94% and
energy by 12.83%; it is therefore retained only as a constrained-fit option.

## Milestone 3 paired experiment

```powershell
.\.venv\Scripts\molt.exe --json curriculum-benchmark `
  --config configs\molt-stream-milestone3.json `
  --seeds 1337,2027,4099 --eval-interval 15
```

This command records paired fixed-context versus sequence-warmup runs, full-
context quality, NVML board energy, thermal state, and bootstrap intervals.
EXP-1016 was rejected; see
[the measured result](docs/results/2026-09-02-exp-1016.md).
