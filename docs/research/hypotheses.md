# Highest-value hypotheses and cheapest falsification experiments

These are candidates, not solutions. Each experiment changes one important
variable relative to BL-0001. “Survives” means only “worth a fuller test.”

## H1 — loss-slope-triggered progressive freezing

**Hypothesis:** freezing lower Transformer blocks only after their smoothed
gradient contribution and validation-loss response have plateaued can reduce
GPU energy-to-quality by at least 20% without worsening final validation B/B by
more than 1% relative to AdamW. This exploratory threshold is below the 2×
promotion gate.

**Rationale:** FreezeOut reported architecture-dependent speedups and failures;
the untested question is whether an explicit, pre-registered plateau signal is
safer than a fixed schedule for this small Transformer.

- Independent variable: freezing policy (`none` vs fixed policy vs plateau policy).
- Controls: initialization, data order, token budget, optimizer, precision,
  evaluation, and hardware state.
- Cheapest falsification (EXP-0101): 2-layer/width-128, 5M-token paired runs,
  seeds 1337 and 2027. Instrument per-layer gradient norms and actual backward
  time. Freeze no earlier than 20% of budget.
- Kill rule: reject if neither seed saves ≥10% training-phase time, either seed's
  final B/B worsens >2%, or optimizer state/scheduler behavior is incorrect.
- If it survives: preregister three-seed BL-0001 comparison plus fixed-schedule
  and random-freeze controls.
- Key confounders: gradients are not causal importance; freezing changes AdamW
  state and effective capacity; short runs may reward early noise.
- Prior art: FreezeOut <https://arxiv.org/abs/1706.04983>. No novelty claim.

## H2 — low-rank gradient projection under a strict memory budget

**Hypothesis:** GaLore-style gradient projection can reduce peak training VRAM
by at least 30% while keeping time-to-target and final B/B regressions within
20% and 1%, respectively, on a model whose optimizer states materially affect
memory.

**Rationale:** published reductions are promising but may not transfer to a
6.45M model, Windows, or an Ada laptop; projection overhead may dominate.

- Independent variable: AdamW vs GaLore projection; start with rank 64 and a
  fixed 200-step refresh interval derived from the paper, then preregister any
  revision.
- Controls: model, initialization, data order, effective batch, schedule,
  precision, and token budget.
- Cheapest falsification (EXP-0201): deterministic synthetic linear and 2-layer
  language-model tests, 1,000 updates, one seed for correctness then two paired
  5M-token seeds. Compare update direction to a transparent reference and
  separately time projection overhead.
- Kill rule: reject for non-finite updates, reference mismatch above declared
  tolerance, <15% peak-VRAM saving, >30% time regression, or >2% B/B regression.
- If it survives: BL-0001 at ranks 32/64/128 with a full AdamW control; only one
  rank chosen using a development seed before the held-out seeds.
- Key confounders: full gradient tensors may dominate peak before projection;
  SVD cadence and allocator caching distort memory readings.
- Prior art: GaLore <https://arxiv.org/abs/2403.03507>. This is a reproduction/
  transfer study, not a new method.

## H3 — cheap diversity-aware data selection

**Hypothesis:** removing exact/near duplicates and downweighting byte windows
that are both redundant and extremely easy for a small frozen reference model
can cut tokens-to-target by at least 20% without worsening held-out B/B by more
than 1%.

**Rationale:** deduplication and reference-model pruning have reported sample-
efficiency gains, but scoring cost and synthetic-corpus bias may erase end-to-end
benefits. The comparison must include preparation energy/time.

- Independent variable: unfiltered deterministic stream vs deduplicated stream
  vs deduplicated plus pre-registered score bands/diversity constraint.
- Controls: raw source revision, validation set, model, seed, optimizer, and
  maximum end-to-end budget.
- Cheapest falsification (EXP-0301): quantify exact and MinHash-near duplication
  in the 20MB training slice; train paired 2-layer models for 5M tokens and add
  preprocessing/scoring cost to time and GPU energy.
- Kill rule: reject if duplicate removal is <2%, if preparation cost cannot be
  amortized within ten BL-0001 runs, if no seed improves tokens-to-target ≥10%,
  or if validation overlap/leakage is detected.
- If it survives: three-seed BL-0001 runs with raw, dedup-only, random-pruned,
  and score-pruned controls at equal raw-data and compute budgets.
- Key confounders: TinyStories is synthetic and repetitive by design; a pruning
  score may narrow rather than improve the distribution.
- Prior art: deduplication <https://arxiv.org/abs/2107.06499> and perplexity
  pruning <https://arxiv.org/abs/2405.20541>. No novelty claim.

## Ordering decision

Run H1 first only after BL-0001 passes. It has the lowest dependency burden and
fastest mechanism check. H2 follows because it needs a correct reference
optimizer and possibly third-party code. H3 follows after dataset manifests and
end-to-end preparation accounting are mature. Do not combine survivors until
each passes its full isolated gate.
