# EXP-1019 — Coupled-Suffix Randomized Telescoping

Date: 2026-09-01  
Decision: **rejected before training**  
Novelty status: **not novel as a general estimator; no novelty claim**

## Research question

Can a decoder Transformer optimize its exact long-context language-model
objective using mostly cheap short-context gradients, while unbiased random
corrections preserve the full-context gradient in expectation?

The proposed Transformer-specific construction couples every context level on
the same suffix targets and absolute positions. This prevents a common invalid
comparison in which changing context also changes targets or positional inputs.

## Prior-art boundary

[Beatson and Adams (2019)](https://arxiv.org/abs/1905.07006) already introduced
randomized telescoping gradient estimators for costly approximation sequences,
derived adaptive probability selection, and evaluated an LSTM on long
sequences. [Shi and Cornish (2021)](https://proceedings.mlr.press/v130/shi21d.html)
applied multilevel Monte Carlo to unbiased gradients for deep latent-variable
models and analyzed finite variance. Therefore randomized telescoping,
multilevel unbiased gradients, and their use for sequence learning are prior
art.

The MOLT implementation adds the following experimental construction:

1. nested *causal context* levels for a decoder Transformer;
2. identical suffix targets at every level;
3. absolute positional indices preserved when the left prefix is removed;
4. coefficient fusion so a context level is evaluated at most once per draw;
5. an exact finite-level cost/second-moment probability frontier measured on
   the target GPU before optimizer training.

These differences may be useful engineering, but this search did not establish
that they are publication-level novelty. The generic mechanism is conclusively
not new, and the candidate failed technically.

## Mathematical specification

Let context levels satisfy

\[
  l_0 < l_1 < \dots < l_K=L.
\]

For one full window \(w\), let \(J_l(\theta;w)\) be cross entropy on the same
final \(m\le l_0\) targets, with only the last \(l\) causal input tokens visible.
Tokens retain positions \(L-l,\dots,L-1\). The exact identity is

\[
  J_L = J_{l_0} + \sum_{k=1}^{K}\left(J_{l_k}-J_{l_{k-1}}\right).
\]

Draw independent \(I_k\sim\mathrm{Bernoulli}(p_k)\), with \(0<p_k\le1\), and
define

\[
 \widehat J = J_{l_0} + \sum_{k=1}^{K}
 \frac{I_k}{p_k}\left(J_{l_k}-J_{l_{k-1}}\right).
\]

Linearity gives

\[
 \mathbb E[\nabla\widehat J]=\nabla J_L.
\]

Writing \(g=\nabla J_L\) and
\(\Delta_k=\nabla J_{l_k}-\nabla J_{l_{k-1}}\), independence gives the exact
second moment

\[
 \frac{\mathbb E\|\widehat g\|^2}{\|g\|^2}=
 1+\sum_k\left(\frac1{p_k}-1\right)
 \frac{\|\Delta_k\|^2}{\|g\|^2}.
\]

This equation exposes the central risk: rare corrections are only efficient
when adjacent gradients are extremely close.

For a dense Transformer with batch \(B\), width \(d\), layers \(N\), and context
\(l\), one level costs approximately

\[
 C(l)=\Theta(BN(ld^2+l^2d)).
\]

MOLT does not use this asymptotic expression for its decision. It measures each
\(C(l)\) on the actual RTX 4060 and exactly enumerates every correction mask to
calculate expected execution cost. The screening score is

\[
 S(p)=\frac{C(L)}{\mathbb E[C(\widehat g)]}
 \bigg/\frac{\mathbb E\|\widehat g\|^2}{\|g\|^2}.
\]

This is a conservative SGD progress proxy, not a convergence theorem or model-
quality result.

## Pseudocode

```text
input: full window w, levels l[0..K], probabilities p[1..K]
coefficients = [1, 0, ..., 0]
for k in 1..K:
    I[k] ~ Bernoulli(p[k])
    if I[k]:
        coefficients[k]   += 1 / p[k]
        coefficients[k-1] -= 1 / p[k]

loss = 0
for k in 0..K:
    if coefficients[k] != 0:
        loss += coefficients[k] * suffix_loss(w, context=l[k])
backpropagate(loss)
```

## Correctness tests

The test suite enumerates all four masks for a three-level estimator and checks
both expected loss and every expected parameter gradient against the exact
longest-context gradient. It also tests coefficient cancellation, absolute
position handling, invalid probabilities, and the analytic probability
frontier. All 27 project tests pass.

## Preregistered kill rules

Reject before optimizer training if any of the following occurs:

- raw measured step speedup is below 2×;
- variance-adjusted speedup proxy is below 1×;
- adjusted peak allocation exceeds the exact reference;
- any non-finite loss or gradient occurs.

A rejected proxy does not advance to multi-seed training. This prevents an
already-failed idea from consuming energy merely to produce a larger result
table.

## Ablations and results

Model: 49,883,520 parameters, FP32 parameters, BF16 autocast, contexts
64→128→512, 64 common suffix targets, TinyStories BPE-8192, RTX 4060 Laptop GPU.

| Probe | Batch | Correction probabilities | Raw step speedup | Gradient second moment | Variance-adjusted proxy | Decision |
|---|---:|---:|---:|---:|---:|---|
| initial instrument check | 1 | 0.10, 0.10 | 9.449× | 20.288× | 0.466× | invalid for speed: reference contained CUDA cold start; rejection still conservative |
| corrected low-utilization | 1 | 0.25, 0.25 | 0.574× | 8.698× | 0.066× | reject |
| corrected realistic batch | 8 | 0.10, 0.10 | 2.768× | 13.607× | 0.203× | reject |
| analytic-frontier run | 8 | 0.10, 0.10 sampled | 3.152× | 12.123× | 0.260× | reject |

For batch 8, measured isolated costs were:

- \(C(64)=0.0132784\) s
- \(C(128)=0.0201209\) s
- \(C(512)=0.0720178\) s

Measured adjacent squared-gradient delta ratios were 0.94554 and 0.72208.
An exact grid over every pair \(p_k\in\{0.05,0.10,\dots,1.0\}\) selected
\((1,1)\) as the best schedule. At \((1,1)\), coefficient fusion reduces the
estimator to the ordinary full-context loss: raw speedup 1×, second moment 1×,
score 1×. Every stochastic alternative was worse.

The realistic 10% probe also used 1.700 GB adjusted peak allocation versus
1.327 GB for the reference, a 28% memory regression caused by retaining several
context graphs on correction steps.

## Interpretation

The estimator is mathematically correct but economically wrong for this
workload. Short- and long-context gradients differ too much, so importance
weights amplify correction variance faster than short contexts save compute.
At batch 1, small forwards also underutilize the GPU. At batch 8, raw speed
appears, but variance and multi-graph memory eliminate it.

No full training, energy-to-quality, or multi-seed claim is made. The automatic
gate rejected the method before those runs. The target of a reproducible 2×
improvement was **not achieved**.

## Exact reproduction

From `D:\Projects\MOLT`:

```powershell
uv sync --frozen
uv run pytest -q
uv run ai-local research-telescope `
  --config configs/molt-ai-50m-pretrain.json `
  --levels 64,128,512 `
  --probabilities 0.1,0.1 `
  --target-tokens 64 `
  --micro-batch-size 8 `
  --samples 8
```

Primary artifact:
`artifacts/benchmarks/telescoping-20260901-105308.json`.

## Next evidence-based direction

Do not add stale or learned correction buffers without a new prior-art review;
they would introduce bias and overlap with variance-reduction/synthetic-gradient
work. The next lowest-risk experiment remains a rigorous reproduction of
[Sequence Length Warmup](https://proceedings.neurips.cc/paper_files/paper/2022/hash/aac02401755a65904cf977a33136af4a-Abstract-Conference.html),
which reported large time reductions but is established prior art. It can
improve MOLT while a genuinely original mechanism continues through a separate
novelty screen.
