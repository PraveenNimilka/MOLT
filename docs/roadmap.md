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

**Current status (2026-09-02): Qwen QLoRA engineering track passed; project-wide
sequence not yet promoted.** EXP-1031 reproduced an isolated resident-activation
configuration across three paired seeds. Median end-to-end time and board
energy improved 21.96% and 22.30%; median interpolated time and energy to NLL
2.10 improved 21.91% and 22.79%; final NLL was identical within every pair.
Additional VRAM was an explicit accepted tradeoff. This is established
activation-checkpoint selection, not a novel algorithm. See
`docs/results/2026-09-02-qwen2-resident-activations.md`.

The main project cannot skip the still-open Milestone 1 and 2 gates. EXP-1031
therefore advances the Qwen experimental track and supplies a Milestone-3-grade
result, while clean-machine reproduction, fault semantics, and the original
BL-0001 evidence remain outstanding.

EXP-1032 then strengthened the tuned QLoRA baseline with established LoRA+
prior art. On three frozen holdout seeds, a five-update ratio-14 schedule versus
10-update vanilla QLoRA improved median end-to-end time 37.17% and GPU-board
energy 48.76%. Every NLL stayed within 1%, allocation was unchanged, all runs
completed, and candidates peaked 5–7 C cooler. This is a strong narrow baseline
result, not MOLT algorithmic novelty. It still requires transfer to other tasks,
model sizes, longer runs, and another machine under Milestone 4.

The first breadth test, EXP-1033, rejected direct transfer to context 512:
although time and board energy were lower, held-out NLL regressed 3.43%, peak
temperature rose 3 C, and a new throttle flag appeared. The EXP-1032 claim is
therefore explicitly limited to context 256. EXP-1034 established bit-exact
rollback for an experimental counterfactual LoRA-rate probe, but its three-arm
proxy cost 6.273x one optimizer step and it cannot beat an already tuned fixed
trajectory without a broader tuning-cost objective. It remains isolated.

EXP-1035 rejected a longer interpretation of EXP-1032. Ratio-14 NLL rebounded
from approximately 2.105 at update 5 to 2.153 at update 10, while the paired
vanilla control thermally aborted in the heat-soaked session. The validated
result is therefore limited to the registered early-quality horizon; it is not
evidence of a full fine-tune speedup.

**Outcome:** one isolated optimization survives a preregistered cheap test and
then a three-seed controlled time/energy-to-quality comparison.

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

Acquire or prepare a legally usable second subword dataset and a second local
model scale, then preregister EXP-1032 transfer without retuning on validation
seeds. In parallel, run longer context-256 time-to-quality soaks to determine
whether the five-update advantage persists beyond the current short horizon.
Until those assets and results exist, Milestone 4 and any defensible-general
performance claim remain open.
