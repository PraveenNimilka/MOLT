# Exact counterfactual LoRA-rate probe

**Status:** experimental primitive; correctness passed; performance promotion
rejected before trainer integration.  
**Novelty status:** not established. Generic line search and online
learning-rate adaptation are prior art.

## Hypothesis

For a LoRA optimizer with separate A and B parameter groups, evaluate several
B/A learning-rate ratios from one computed gradient by restoring parameters and
optimizer state before each candidate. Commit only the candidate with the best
deterministic score. This might adapt a fixed LoRA+ ratio to workload geometry
without repeating backward passes.

## Exactness argument

For AdamW at one optimizer state and gradient, moment updates do not depend on
the learning rate:

```text
m' = beta1*m + (1-beta1)*g
v' = beta2*v + (1-beta2)*g^2
u  = bias_correct(m') / (sqrt(bias_correct(v')) + epsilon)
theta' = theta - eta*(u + weight_decay*theta)
```

Therefore each candidate rate can start from the same `theta`, `m`, `v`, and
`g`. Restoring all four before every candidate and before the final commit makes
the committed update exactly equal to a standalone AdamW step at the selected
rates. Tests cover bit-exact parameters, nonempty optimizer state, and rollback
after invalid scoring.

With `P` trainable elements, `S` optimizer-state elements, `K` candidates, and
score-forward cost `F`, the probe requires `O(P+S)` snapshot memory and
`O(K(P+S+F))` time while reusing one backward pass. This is only attractive when
saved tuning/backward work exceeds state-copy and score-forward overhead.

## Prior art boundary

- LoRA+ already introduces unequal A/B learning rates and tunes a fixed ratio:
  <https://arxiv.org/abs/2402.12354>.
- Hypergradient descent adapts learning rates online and predates MOLT:
  <https://arxiv.org/abs/1703.04782>.
- LoRA-RITE and newer LoRA-Muon work target factor-scaling/optimizer robustness,
  further narrowing any novelty claim:
  <https://openreview.net/pdf?id=VpWki1v2P8> and
  <https://arxiv.org/abs/2606.12921>.

The exact rollback implementation is clean-room MOLT code, but that does not by
itself establish a new algorithm.

## Cheap falsification result

On the RTX 4060 Laptop GPU, a 4.4M-parameter two-group AdamW proxy with three
rate candidates measured 6.713 ms median versus 1.070 ms for one ordinary
optimizer step, **6.273x optimizer-only overhead**. A reproducible
warm-up-excluded benchmark is provided at
`benchmarks/counterfactual_rate.py`; raw output is stored under
`artifacts/benchmarks/counterfactual-rate-proxy.json`.

This proxy excludes model forward/backward, candidate validation forwards,
energy, temperature, and convergence. Against a fixed ratio already tuned on
the same workload, the probe adds strictly positive work and cannot improve the
identical trajectory if it selects that ratio. Using held-out score feedback to
select a seed-specific ratio also creates an overfitting risk.

## Decision

Keep the exact primitive isolated under `methods/` for future cross-workload
tuning-cost research. Do not wire it into stable QLoRA and do not claim novelty
or a performance win. Reconsider only when at least two model/workload
geometries are available and total tuning cost—not one selected run—is the
preregistered primary metric.
