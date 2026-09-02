# MOLT architecture

Status: active architecture for package version 0.2.x.

## Design goals

MOLT is an evidence-first local training engine. Stable domain types point
inward, hardware behavior is explicit, experiment outputs are immutable
artifacts, and optional acceleration must never silently fall back to a
different workload.

## Package map

```text
src/molt_stream/
├── core/          immutable specs, progress/telemetry contracts, errors
├── data/          deterministic mmap token windows and contiguous packing
├── kernels/       fused operators, compiler dispatch, exact partitioned loss
├── measurement/   NVML telemetry, thermal controllers, Pareto frontiers
├── methods/       isolated, unpromoted research mechanisms and correctness probes
├── streaming/     host NF4 storage and grouped LoRA layer streaming
├── training/      pretraining, QLoRA, optimizers, benchmarks, experiments
├── experiments/   digest-verified atomic checkpoints
└── cli.py         the sole `molt` command-line interface
```

The automated boundary test prevents `core` from importing outward and keeps
data, kernels, measurement, streaming, and experiments independent of the
training orchestration layer. Experimental `methods` also remain inward of
training and cannot import orchestration code.

## Training path

```text
JSON config -> validated TrainingSpec -> mmap batcher -> model/kernel backend
            -> optimizer + thermal controller -> atomic checkpoint + metrics
```

Configuration is parsed once and rejects unknown fields. CUDA requests fail
when CUDA is unavailable. Compiled execution is selected explicitly; MOLT does
not substitute eager execution after a compiler failure. Input token order,
batch geometry, seeds, and optimizer state are persisted for reproducibility.

## Storage and recovery

`MMapTokenBatcher` maps token files without copying the corpus into process RAM.
Its cursor and RNG state are checkpointed. `AtomicCheckpointStore` writes a
temporary checkpoint, flushes it, calculates SHA-256, atomically replaces the
active checkpoint, and retains one verified previous generation. Loading accepts
only a data/manifest pair whose length and digest match.

## Hardware-specific execution

- Full-update SLM training uses eager or an explicitly requested PyTorch compile
  backend and fused CUDA AdamW when CUDA is selected.
- QLoRA requires its declared optional libraries and fails clearly when they are
  absent.
- Streamed LoRA keeps frozen NF4 payloads on the host, groups transfers into
  configurable layer bundles, and keeps adapters on the compute device.
- CUDA graphs are rejected for the current host-restreaming backward because
  graphed callables require stable tensor addresses.
- NVML sampling measures board power, temperature, utilization, memory, enforced
  power limits, and clock-event reasons. Power-limit mutation is opt-in only.

## Experiment boundary

Production mechanisms live outside research selection logic. Benchmarks and
paired experiments emit machine-readable artifacts under `artifacts/`; measured
negative results remain valid project outputs. A method is promoted only after
its pre-registered quality and efficiency gates survive repeated paired runs.

## Public interface

Run `molt --help` for the authoritative command list. Core workflows are:

```powershell
molt inspect
molt prepare --config configs\molt-stream-smoke.json
molt train --config configs\molt-stream-smoke.json
molt resume --run RUN_DIRECTORY
molt evaluate --run RUN_DIRECTORY
molt benchmark --config configs\molt-stream-production.json
molt report --run RUN_DIRECTORY
```

Interactive terminals render the optional UI. Redirected output and `--json`
remain stable machine-readable JSON.
