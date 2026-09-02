# Evidence-gated roadmap

Milestones advance on evidence, not code completion. A failed gate remains the
current milestone and produces a diagnosis or negative result.

## Milestone 0 — audit and pre-registration

**Outcome:** repository/hardware audit, primary-source research matrix and plan,
baseline protocol, modular architecture/ADRs, machine-readable hypothesis
registry, and falsification-first roadmap.

**Gate:** every observed claim is traceable; prior results are labeled reported;
unknowns and measurement limits are explicit; baseline/candidate thresholds are
defined before execution; documents and registry validate locally.

**Status:** satisfied on 2026-09-01. JSON schema/required-field checks, local
Markdown-link checks, and independent parameter-count arithmetic passed. No
training evidence was produced or implied.

## Milestone 1 — trustworthy baseline

**Current status (2026-09-01): partially implemented, evidence gate not met.**
The pinned runtime, training engine, CPU/CUDA smoke tests, telemetry, and
checkpoint integrity exist. BL-0001 still requires a pinned TinyStories download,
three full seeds, short-repeat variance, and deterministic resume evidence.

**Exact outcome:** a pinned native-Windows Python/PyTorch environment and BL-0001
with three successful seeds, five short performance repeats, raw telemetry,
environment/dataset hashes, and an immutable target-quality manifest.

Ordered work:

1. initialize version control if authorized and add license/project metadata;
2. add `pyproject.toml`, `.python-version`, and `uv.lock` with minimal dependencies;
3. implement typed config/domain schemas, inspection, and artifact state machine;
4. implement deterministic byte data preparation and manifests;
5. implement the model and single baseline training engine;
6. implement phase timing, RAM/VRAM/NVML sampling, and weighted evaluation;
7. implement atomic checkpoint/resume with CPU exactness tests;
8. run CPU/CUDA smoke tests, telemetry-overhead test, short repeats, then seeds;
9. freeze baseline comparison manifest and publish all results/failures.

**Gate:** every acceptance criterion in `baseline-spec.md` passes. If Python or
dataset acquisition is declined, the milestone is blocked—not silently changed.

## Milestone 2 — experiment runner and failure semantics

**Current status (2026-09-01): functional prototype, full gate not met.** The CLI,
artifact schemas, corruption fallback, comparisons, and reports work. Signal
interruption, injected OOM, and clean-machine reproduction remain outstanding.

**Outcome:** complete CLI workflow, schema-versioned registry/artifacts,
interrupt/resume/corruption/OOM tests, compatible-run comparison, confidence
intervals, machine-readable and Markdown reports.

**Gate:** clean machine reproduction from lockfile; resume equivalence; fault
tests; report regeneration from raw artifacts; no hidden fallback path.

## Milestone 3 — first isolated optimization

**Outcome:** EXP-0101 cheap falsification followed, only if it survives, by a
pre-registered three-seed BL-0001 progressive-freezing comparison.

**Gate:** candidate meets the full success/regression gate, or is recorded as a
negative result. Passing only the cheap exploratory threshold does not promote.

## Milestone 4 — reproduction breadth

**Outcome:** reproduce any survivor across at least three seeds, two workloads
(byte-level and a standard subword workload), two model scales, and another
machine if one becomes available.

**Gate:** directionally consistent primary outcome with disclosed heterogeneity;
failure to transfer narrows the claim instead of being discarded.

## Milestone 5 — validated combination

**Outcome:** combine only independently validated improvements using a factorial
or staged ablation that measures interaction effects.

**Gate:** combination beats the strongest individual method and baseline without
violating regressions; otherwise retain the simpler individual method.

## Milestone 6 — external reproduction

**Outcome:** pinned public package, dataset/model license manifest, raw and
derived results, one-command reduced reproduction, full methodology, negative
results, and an honest technical report.

**Gate:** independent party reproduces the principal result within a
pre-registered tolerance. “Breakthrough” remains prohibited without evidence
far beyond a single project and machine.

## Immediate next decision

Authorize Milestone 1 environment setup and dataset download. Expected external
effects are a Python/PyTorch dependency download, dataset network traffic, and
several GiB of repository-local/cache storage; exact pins and download sizes
should be shown before installation.
