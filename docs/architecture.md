# Architecture proposal

## Design goal

One training engine should execute stable baselines and isolated methods without
duplicating loops. Experimental code may depend on stable contracts; stable
code must never import a method implementation.

## Proposed repository boundaries

```text
src/ai_local/
  core/          immutable IDs, configs, events, errors, domain types
  backends/      PyTorch/CPU/CUDA adapters and capability discovery
  training/      one engine, optimizer/precision/checkpoint orchestration
  data/          manifests, preparation, integrity, deterministic sampling
  methods/       experimental policies implementing stable hooks
  scheduling/    placement, resource limits, offload contracts
  measurement/   phase clocks, samplers, energy integration, metric schemas
  experiments/   spec validation, run lifecycle, artifact store, registry
  evaluation/    aggregate quality and comparison statistics
tests/           unit, integration, determinism, failure, and regression tests
benchmarks/      versioned benchmark specs; no ad-hoc result selection
configs/         immutable, versioned experiment configurations
docs/            scope, decisions, methodology, results, and negative results
scripts/         thin operational wrappers only
artifacts/       ignored run outputs, content-addressed where practical
```

The import direction is inward toward `core`. `core` imports no project package.
`measurement`, `data`, `training`, and `scheduling` consume core types;
`backends` implements core protocols; `methods` implements hook protocols;
`experiments` composes them. The CLI calls `experiments` and contains no training
logic.

## Stable contracts

Keep the initial public surface small:

- `ExperimentSpec`: fully validated immutable configuration plus schema version.
- `RunId` / `ExperimentId`: non-semantic identifiers; reruns never overwrite.
- `Backend`: capabilities, synchronization, memory reset/read, RNG capture/restore.
- `Trainable`: forward/loss, state serialization, exact parameter accounting.
- `TrainingMethod`: explicit hooks around backward/update with declared state and
  compatibility; baseline is a no-op method.
- `SamplerState`: dataset revision, permutation algorithm/version, epoch, cursor.
- `CheckpointBundle`: model, optimizer, scaler, scheduler, RNG, sampler, method,
  step/token counters, hashes, and atomic completion marker.
- `MetricSink`: append-only typed events; measurement failure cannot be hidden.
- `Evaluator`: token-weighted metric aggregation over an immutable manifest.

Experimental methods receive a narrow `StepContext`; they do not own the data
loader, main loop, checkpoint format, or evaluator. Unsupported capability
combinations fail validation before allocation.

## Run lifecycle

1. Parse and schema-validate config; resolve every floating reference.
2. Inspect capabilities and reject unsupported/unsafe combinations.
3. Materialize a run directory and immutable resolved spec.
4. Verify dataset/artifact hashes and acquire an exclusive run lock.
5. Seed backend and construct model, method, optimizer, evaluator, and telemetry.
6. Warm up without advancing training state; reset measurement peaks.
7. Execute the single engine, emitting phase and step events.
8. Save checkpoints through write-temporary → flush → hash → atomic rename →
   completion-marker; retain prior valid checkpoint until success.
9. Evaluate, finalize checksums, and mark the run success/failure/interrupted.
10. Compare only compatible manifests and generate machine-readable plus human
    reports from raw artifacts.

No hidden fallbacks: CPU substitution, smaller batches, changed precision,
disabled determinism, skipped records, telemetry loss, and checkpoint recovery
are explicit events or hard failures according to the spec.

## Saturating data path

Prepared corpora are written in bounded chunks and training splits are opened
through OS-backed memory maps. Batch windows are gathered as byte tensors, sent
to CUDA compactly, and widened to integer token IDs on-device. This bounds RAM
use independently of corpus size; storage/page-cache bandwidth remains a
measured constraint.

`ai-local tune` runs isolated context/micro-batch candidates, catches only CUDA
out-of-memory failures, records peak allocation/reservation and loader enqueue
share, and rejects candidates below a declared VRAM headroom. It never changes a
training configuration automatically. The selected shape must pass a separate
equal-quality experiment before promotion.

## Artifact layout

```text
artifacts/runs/<run-id>/
  spec.resolved.json
  environment.json
  status.json
  events.jsonl
  metrics.summary.json
  data.manifest.json
  checkpoints/<step>/...
  stderr.log
  report.json
```

Raw JSONL is append-only. Summaries are derived and reproducible. Every schema
is versioned; unknown versions fail loudly. Secrets and arbitrary executable
objects are forbidden in configs/checkpoints. Loading uses tensor/state formats
with explicit allowlists where possible.

## Measurement architecture

A monotonic high-resolution clock defines phase spans. GPU spans synchronize at
boundaries. Independent samplers capture process/system RAM, NVML VRAM, GPU
power/utilization, CPU utilization, and storage I/O with timestamps and error
status. Energy integration rejects gaps beyond a configured threshold.

Framework allocator peaks and device-wide NVML memory answer different questions
and are retained separately. Telemetry overhead is measured with an A/B no-op
benchmark. Compilation and warm-up are never merged into steady-state training,
though total-to-quality includes all required work for a fresh run.

## Test strategy

- Unit: config invariants, weighted metrics, energy integration, state machines,
  checksums, comparison compatibility, and each algorithm against references.
- CPU integration: overfit fixture, deterministic data order, atomic checkpoint,
  corruption detection, exact resume, and signal interruption.
- CUDA integration: AMP/scaler resume, memory metrics, OOM as a recorded failure,
  telemetry gaps, and documented numerical tolerance.
- Regression: golden schema fixtures and performance thresholds on dedicated
  benchmark specs; never fail ordinary unit tests on noisy wall-clock limits.
- Candidate tests: gradient/update equivalence before quality experiments.

## CLI mapping

The proposed package name remains `ai-local` provisionally:

```text
ai-local inspect
ai-local prepare --config CONFIG
ai-local train --config CONFIG
ai-local resume --run RUN_ID
ai-local evaluate --run RUN_ID
ai-local benchmark --config CONFIG
ai-local tune --config CONFIG --contexts 8,16,32,64,128
ai-local compare BASELINE_RUN CANDIDATE_RUN
ai-local report --run RUN_ID
```

`inspect` is read-only. `prepare` never starts training. `resume` never mutates a
completed run and refuses incompatible environment/config changes unless an
explicit migration creates a new run.

## Deferred decisions

- SQL vs file registry: begin with append-only files and atomic indexes; revisit
  only after concurrent-machine requirements exist.
- Configuration format: JSON is sufficient for machine records; human configs
  may use YAML only if strict schema validation and dependency cost are justified.
- Frameworks beyond PyTorch, distributed execution, NPU, custom kernels, and UI
  dashboards wait behind measured needs.
