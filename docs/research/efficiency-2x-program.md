# MOLT 2× efficiency research program

Date: 2026-09-01

## Objective and claim boundary

MOLT will pursue five independent promotion gates. A method does not need to
pass all five, but it must pass at least one without violating its declared
quality, time, memory, stability, and total-cost limits.

| Gate | Primary measure | Required result | Regression limit |
|---|---|---|---|
| E1 energy | GPU board joules to a fixed held-out loss | at most 50% of baseline | quality no worse than 1%; time no worse than 20% |
| E2 time | end-to-end seconds to fixed held-out loss | at most 50% of baseline | energy no worse than 20%; memory must fit |
| E3 memory | framework peak allocated VRAM at identical workload | at most 50% of baseline | time no worse than 10%; quality no worse than 1% |
| E4 quality | held-out NLL at equal tokens and estimated training FLOPs | statistically better over at least 3 seeds | time, energy, and memory no worse than 20% |
| E5 constrained fit | largest model/context/batch that completes reliably | reference OOMs and candidate completes 3/3 runs | useful loss decline; storage traffic and time disclosed |

The present 50M/TinyStories run is a mechanism test, not evidence about all
models, datasets, GPUs, or data-center economics. Consumer hardware cannot
match a data center's absolute aggregate throughput; the falsifiable target is
better efficiency or accessibility for a defined workload.

## Measured starting point

The immutable baseline and recomputation candidate used the same 49,883,520
parameter model, tokenizer, data order, 8,835,072-token budget, 719 updates,
AdamW configuration, seed 1337, FP32 parameters/optimizer state, BF16 autocast,
and RTX 4060 Laptop GPU.

| Result | Baseline | Recompute only | Change |
|---|---:|---:|---:|
| Final validation perplexity | 33.7011 | 33.7086 | +0.0223% |
| End-to-end time | 225.257 s | 272.423 s | +20.94% |
| End-to-end throughput | 39,222 tok/s | 32,431 tok/s | -17.31% |
| Framework peak allocated VRAM | 4,745,062,912 B | 2,372,921,856 B | -49.992% |
| Device-wide peak used VRAM | 6,565,064,704 B | 4,318,654,464 B | -34.22% |
| Measured GPU board energy | 16,543.64 J | 18,666.36 J | +12.83% |

Interpretation: recomputation essentially halves framework allocation and can
make otherwise impossible shapes fit, but it fails E3's time limit and does not
save energy. It is retained as an opt-in constrained-fit method, not promoted as
a general speed or energy optimization. The single seed establishes mechanism,
not statistical equivalence.

An explicit BF16 residual-stream probe did not reduce peak allocation at the
baseline shape (4.748 GB versus 4.747 GB). Its isolated hypothesis is rejected.
A combined BF16-residual plus recomputation run reached 2.239 GB and perplexity
33.7034 in 266.337 seconds. That interaction is exploratory until reproduced.

## Prior-art boundary and research matrix

“Reported” below means the source authors' result, not a MOLT result.

| Direction | Established mechanism | Reported evidence | MOLT interpretation | Untested MOLT hypothesis |
|---|---|---|---|---|
| Sequence-length warmup | begin with shorter sequences, then grow to target length | [Li et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/aac02401755a65904cf977a33136af4a-Abstract-Conference.html) report up to 2.2× fewer tokens and 3.7× less wall time on GPT-2 scales | strongest cheap candidate for E1/E2; comparison must hold final context and quality fixed | a token-weighted 128→256→512 schedule reaches baseline NLL in ≤60% of time and energy |
| Progressive depth/model growth | train a smaller network and expand it | [Progressive Stacking 2.0](https://arxiv.org/abs/2011.13635) reports over 110% BERT speedup; [Transformer Growth](https://arxiv.org/abs/2010.12562) reports 73.6–82.2% pretraining speedups | promising for E2, but expansion must preserve function and optimizer semantics | 3→6→9 layers reaches baseline NLL with ≤60% FLOPs and time |
| Muon optimizer | orthogonalized matrix updates with scale/decay rules | [Muon is Scalable](https://arxiv.org/abs/2502.16982) reports about 2× compute efficiency versus AdamW in its scaling study | promising for E4; recent evidence needs exact reference tests and multiple seeds | transparent Muon matches reference updates, then improves NLL at equal tokens/FLOPs |
| Lion optimizer | sign-based update with one momentum state | [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675) and the [official implementation](https://github.com/google/automl/tree/master/lion) report half Adam's auxiliary state and competitive language-model results | direct optimizer-memory candidate, but not a new MOLT invention | Lion cuts optimizer-state bytes roughly in half without >1% NLL or >10% time regression |
| Activation compression | store low-bit/noisy activations and reconstruct for backward | [ActNN](https://arxiv.org/abs/2104.14129) reports randomized 2-bit activation compression; [GACT](https://proceedings.mlr.press/v162/liu22v.html) generalizes compressed activation training; [ALAM](https://proceedings.iclr.cc/paper_files/paper/2024/hash/18abbeef8cfe9203fdf9053c9c4fe191-Abstract-Conference.html) studies Transformers | possible route to E3 with less recompute; quality and kernel overhead are the central risks | 4-bit per-channel saved tensors beat recomputation's memory/time Pareto point |
| Power-aware scheduling | tune GPU power/clock and batch choices using measured energy | [Zeus](https://arxiv.org/abs/2208.06102) reports 15.3–75.8% energy improvements across tested workloads | credible E1 candidate, unlikely alone to guarantee 2× on this laptop | a measured power/batch frontier cuts joules-to-loss ≥20% before quality experiments |
| Data deduplication/selection | remove redundancy or choose high-information samples | [Lee et al.](https://arxiv.org/abs/2107.06499) report benefits from deduplicating language-model data | possible E4/E1 gain, but preprocessing cost and distribution narrowing count | exact/near dedup plus diversity control reduces tokens-to-target after amortized preparation cost |
| Heterogeneous scheduling | place state/data across GPU, pinned RAM, and storage with overlap | no single source establishes a universal win; offload trades VRAM for interconnect/storage traffic | only valuable when profiling proves memory pressure and transfers can be hidden | a double-buffered, deadline-aware planner fits an OOM workload with ≥85% GPU utilization and beats naive offload |

None of these established techniques is claimed as a MOLT invention. A new
method would require a precise difference from this matrix, an ablation that
isolates that difference, and independent reproduction.

## Three highest-value hypotheses and cheapest kill tests

### EXP-1016 — sequence-length warmup (first)

- Hypothesis: a 128→256→512 curriculum reaches baseline 5.075 bits/token using
  at most 60% of baseline time and GPU energy.
- Controls: same final model, seed, tokenizer, example stream, maximum token
  budget, optimizer, effective token batch, validation at context 512, and full
  data/preparation accounting.
- Cheapest test: 90 updates per arm with baseline fixed-512 and two preregistered
  curricula; evaluate all arms at context 512.
- Kill: neither curriculum improves time-to-target by 25%, or NLL is >2% worse.
- Full promotion: three paired seeds and gates E1/E2. Schedule selection uses a
  development seed; held-out seeds cannot tune the schedule.

### EXP-1017 — optimizer Pareto test (second)

- Hypothesis: Lion reduces optimizer-state memory, or Muon improves quality per
  compute, without unacceptable instability.
- Cheapest test: deterministic quadratic and tiny Transformer update-by-update
  comparison against the authors' equations/reference implementation, followed
  by 200-update paired runs.
- Kill: numerical reference mismatch, non-finite values, <20% state-memory gain
  for Lion, or >2% NLL regression. Muon is killed if it cannot improve loss-area
  under an equal FLOP/token budget on two development seeds.
- Full promotion: three seeds at 50M and one smaller model. Do not choose between
  optimizer hyperparameters on the reporting seeds.

### EXP-1018 — memory planner (third)

- Hypothesis: selective recomputation plus compressed saved tensors can retain
  at least 50% allocated-memory reduction with at most 10% time regression.
- Cheapest test: profile bytes and recompute time per block, then enumerate
  checkpoint masks on a 4-block proxy. Compare full save, full recompute, and
  predicted Pareto mask; no model training is needed for the first kill test.
- Kill: profiler prediction error >10%, no mask meets the memory boundary, or
  compression reconstruction/update error exceeds declared tolerance.
- Full promotion: equal-token 50M runs, then an OOM control/candidate pair for
  E5. GPU, host RAM, and storage traffic are all reported.

## Ordered build plan

1. **Measurement hardening:** add repeated benchmark aggregation, confidence
   intervals, GPU power-state capture, and explicit allocated/reserved/device-
   wide memory fields. Gate: five short repeats vary <5%, or variance source is
   documented.
2. **Schedule abstraction:** add declarative per-phase context, depth, batch,
   learning rate, and stopping targets while retaining one training engine.
   Gate: a constant schedule reproduces the current run and resume is equivalent.
3. **EXP-1016:** implement sequence-length warmup and run its kill test. This is
   first because published effect size is large and implementation risk is low.
4. **Optimizer laboratory:** reference equations, state-byte accounting, and
   numerical tests before integrating Lion or Muon into production training.
5. **Activation Pareto planner:** collect per-block save/recompute profiles,
   choose masks under a VRAM constraint, then investigate low-bit saved tensors.
6. **Power controller:** sweep safe software power limits only when supported;
   otherwise measure batch/clock operating points without changing firmware.
7. **Data-efficiency pipeline:** immutable raw/validation split, leakage checks,
   dedup/diversity manifests, and amortized preprocessing energy.
8. **Heterogeneous runtime:** only after profiler evidence. Use pinned double
   buffers, bounded queues, checksummed spill files, prefetch deadlines, and an
   explicit abort instead of a hidden fallback. Compare against naive CPU
   offload and GPU-only controls.
9. **Combination:** factorially combine only independently promoted methods and
   measure interactions. A combined system must beat its strongest component.
10. **Reproduction:** three seeds, two scales, two workloads, interruption/
    resume and OOM tests, then another machine before any broad claim.

## Architecture changes required next

- `core/`: typed `TrainingPhase` and promotion-gate schemas.
- `training/`: phase scheduler and optimizer protocol; one loop only.
- `methods/`: sequence warmup, progressive growth, optimizer candidates, and
  checkpoint planners remain experimental plugins.
- `measurement/`: repeated-run statistics, power-state metadata, PCIe/storage
  bytes, and per-block activation profiles.
- `scheduling/`: bounded placement plan with observable transfer events.
- `experiments/`: preregistration locks configuration and thresholds before run.
- `evaluation/`: target-quality interpolation with tolerance, not an exact
  floating-point equality that incorrectly labels equivalent runs as misses.

## Decision after this cycle

Recomputation survives only for constrained-fit research. It is not promoted
for E1, E2, or E3. EXP-1016 is the next recommended implementation because it
has the best combination of reported effect size, low engineering cost, and a
cheap falsification path. The 2× goals remain targets, not promises.
