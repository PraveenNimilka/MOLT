# Research and prior-art plan

## Review protocol

For each candidate method, preserve the query, retrieval date, primary source,
official implementation revision, evaluation workload, hardware, claimed
outcome, variance/seeds, ablations, known failures, license, and reproduction
status. Secondary surveys may discover terminology but may not support a core
claim when a primary source is available.

Evidence strength is scored before local testing:

| Grade | Minimum evidence |
|---|---|
| A | Peer-reviewed primary work, usable code/artifacts, multiple workloads/seeds, meaningful ablations |
| B | Primary preprint with code and clear controlled experiments |
| C | Primary source but incomplete code, controls, variance, or hardware transfer |
| D | Anecdote, vendor claim without sufficient method, or speculative proposal |

Local reproduction status is independent: `not_attempted`, `blocked`,
`failed`, `partial`, or `reproduced_within_tolerance`.

## Search order

1. Measurement correctness: deterministic training, energy telemetry, memory
   semantics, checkpoint/resume equivalence, Windows/PyTorch behavior.
2. Strong baselines: AdamW+AMP+SDPA; activation checkpointing; LoRA for later
   adaptation; standard data ordering.
3. Closest methods for each hypothesis: FreezeOut and dynamic freezing; GaLore
   and low-bit optimizers; deduplication, DoReMi, and reference-model pruning.
4. Scaling and negative results: small-model transfer, kernel overhead,
   offloading under low RAM, thermal/power variability.
5. Licenses and patents at the exact revisions considered for reuse.

## Reproduction policy

A paper result is not “reproduced” when only its code runs. Pre-register a
tolerance for its principal metric, use the documented workload when feasible,
retain logs and environment hashes, and explain any hardware or scale deviation.
If faithful reproduction is unaffordable, label a reduced experiment as a
mechanism check rather than a reproduction.

## Living outputs

- `prior-art-matrix.md`: source-backed technique comparison;
- `hypotheses.md`: MOLT-specific falsifiable predictions;
- `experiments/registry.json`: immutable experiment identities and decisions;
- future `docs/results/`: reports generated from raw run artifacts;
- future `docs/negative-results.md`: rejected hypotheses and failure evidence.
