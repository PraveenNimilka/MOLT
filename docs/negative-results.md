# Failure and negative-results log

This append-only log was created before the first experiment so rejected
hypotheses and operational failures would have a durable home.

| Date | Experiment | Outcome | Evidence | Decision |
|---|---|---|---|---|
| 2026-09-01 | Environment audit | Python/ML framework absent; no baseline runnable | Hardware audit | Remain at Milestone 0; specify environment prerequisite |
| 2026-09-01 | EXP-1002 | Direct BF16 SGD at lr 1e-4 stagnated: 8.5654→8.5629 B/B over 256 steps | Raw run artifacts | Reject low-rate configuration |
| 2026-09-01 | EXP-1004 | Error feedback improved sample efficiency but achieved only 1.36× interpolated time-to-quality, added ~2.08 GB, and nearly doubled energy | Controlled EXP-1003 comparison | Reject promotion; retain prior-art mechanism |
| 2026-09-01 | 1B useful pretraining in 10 min | Measured ~64 tokens/s implies only ~38k tokens/10 min, far below useful 1B-scale token budgets | EXP-1005 and compute bound | Reject useful-pretraining claim; retain narrow memorization result |
| 2026-09-01 | EXP-1010 equal-token saturation | Context 128 processed the same 16,384 tokens 13.41x faster, but ended at 5.0369 B/B versus 2.9216 because it performed 128 rather than 2,048 optimizer updates | Raw run and comparison artifacts | Reject as a learning-efficiency improvement; retain shape tuning as throughput infrastructure |
| 2026-09-01 | `torch.compile` mechanism probe | PyTorch 2.8.0+cu128 `reduce-overhead` failed with `torch._inductor.exc.TritonMissing` in the pinned Windows environment | Exact one-operation CUDA probe | Do not add an unreviewed compiler dependency; isolate a pinned compiler-backend experiment later |
| 2026-09-01 | MOLT AI 50M QA v1 | Unmasked causal loss learned QA surface format but generated repetitive questions instead of the factual answer | Run `20260901-094347-...` and fixed prompts | Reject unmasked SFT; add answer-only target masking |
| 2026-09-01 | MOLT AI 50M QA v2/v3 selection | Step-200 v2 passed direct and paraphrase behavior but overfit aggregate held-out answer loss; step-20 v3 had the better curve but failed the paraphrase | Both immutable runs and generation tests | Package v2 only as narrow specialist; retain v3 as early-stop control; no general-QA claim |
| 2026-09-01 | EXP-1014 explicit BF16 residual stream | Same-shape peak allocation was 4.748 GB versus 4.747 GB baseline; the expected memory saving did not appear | `saturation-20260901-101713.json` and baseline tuner artifact | Reject isolated mechanism; do not promote |
| 2026-09-01 | EXP-1015 full activation recomputation | Equal-token perplexity was 33.7086 versus 33.7011; allocation fell 49.992%, but end-to-end time rose 20.94% and GPU energy rose 12.83% | Runs `20260901-093650-...` and `20260901-102421-...`; tuner `saturation-20260901-102408.json` | Retain opt-in for constrained-fit experiments; reject E1/E2/E3 promotion and investigate selective planning |
| 2026-09-01 | EXP-1019 coupled-suffix randomized telescoping | Exact gradient estimator passed correctness tests, but the best analytic probability schedule was the ordinary full-context gradient at 1×; stochastic batch-8 variants scored only 0.203–0.260× after gradient second-moment adjustment and increased proxy peak memory | `telescoping-20260901-105151.json`, `telescoping-20260901-105308.json`, and `coupled-suffix-telescoping.md` | Reject before optimizer training; generic randomized telescoping is also prior art, so make no novelty claim |
| 2026-09-01 | EXP-1020 fixed-workload utilization frontier | Microbatch 24/accumulation 1 won the geometry sweep. Contiguous indexing, FP32, unfused AdamW, and forced alternatives did not provide a joint time/energy improvement. Windows lacked a usable Triton compiler and the laptop rejected manual power-limit control. | `artifacts/benchmarks/utilization-*.json` and `artifacts/frontiers/time-energy-20260901-142341.json` | Retain the tuned BF16 configuration; power state is an observed environmental variable, not a controllable treatment on this host. |
| 2026-09-01 | EXP-1021 loss-conditioned time-energy controller | In the preregistered 80-update kill test, controller exploration was 1.592% slower and consumed 3.054% more board energy than static microbatch 24. It returned to the static winner, so no full run was justified. | `artifacts/benchmarks/controller-20260901-142721.json` | Reject this action set/controller version; do not call it novel or promote it to the stable trainer. |
| 2026-09-01 | EXP-1022 MOLT NF4 double-buffered layer stream | After moving dequantization off the copy stream, width-2048 double buffering still measured 5,697 versus 5,934 activation vectors/s for synchronous streaming (-3.99%); sampled energy was also higher. | `artifacts/molt-stream/benchmarks/stream-tune-20260901-145218.json` and `stream-tune-20260901-145222.json` | Keep opt-in; reject performance promotion. These are not LM tokens/s. |
| 2026-09-01 | MOLT small-model 35k throughput gate | The stable 200-step, 463,488-parameter held-out run reached 22,645.5 end-to-end tokens/s, below 35,000, while perplexity decreased monotonically from 6.08e37 to 1.358. | `artifacts/molt-stream/runs/20260901-150507-pretrain-c6acb6ab` | Gate failed; profile without changing workload semantics. |
| 2026-09-01 | EXP-1023 grouped K=4 streaming | Three paired 60-update runs changed throughput by -11.77%, +4.11%, and +4.11% (median +4.11%); energy was mixed. A shorter run's +10.50% did not reproduce. | `stream-tune-20260901-153044.json` through `stream-tune-20260901-153056.json` | Keep experimental; reject the required >=10% speedup claim. |
| 2026-09-01 | Windows max-autotune/native fusion | Optional Triton-Windows made no-graph fusion operational and cut incremental forward/backward allocation 14.37%, but the required 40% failed. Graph-enabled max-autotune crashes in the Windows static launcher; native MSVC/nvcc tools remain absent. | `artifacts/molt-stream/benchmarks/fusion-memory-20260901-154400.json` | Retain no-graph backend as experimental; reject graph and 40% claims. |
| 2026-09-01 | Production geometry thermal acceptance | The 10.78M context-512/batch-24 eager profile measured 127,854 and 128,520 tokens/s in short windows, while compiled no-graph briefly measured 175,256; all crossed 70 C and auto-aborted (71 C, 72 C, and 74 C peaks). | `production-throughput-20260901-153318.json`, `production-throughput-20260901-153335.json`, `production-throughput-20260901-154457.json` | Throughput arithmetic passes for this small model; commercial reliability and thermal gate fail. |
| 2026-09-01 | EXP-1026 60 C duty cycling at requested cool geometry | The laptop rejected the 65 W NVML request and remained at 115 W. With batch 16/accumulation 2, twelve 30-50 ms pauses could not prevent a 74 C peak; the run aborted after 8/30 measured steps at 115,173 tokens/s. | `artifacts/molt-stream/benchmarks/production-throughput-20260901-181655.json` | Reject the claim that 40 ms sleeps lock temperature at 60 C. Do not spend energy on the preregistered 500-step run. |
| 2026-09-01 | EXP-1027 predictive thermal cruise | Two compiled batch-32 probes reached 73-74 C during max-autotune/warm-up and aborted before a valid interval. Eager controller variants completed, but late/early throughput ratios were 0.881, 0.798, 0.945, 0.576 and 0.586; none passed the preregistered 0.97 stability gate. The best short safe interval was 103,858 tokens/s, peak 64 C, no throttle, ratio 0.945. | `production-throughput-20260901-203016.json` through `production-throughput-20260901-223807.json` | Retain controller instrumentation and safety fixes, but reject promotion as a solved cruise profile. Compilation heat and long thermal soak remain blockers. |
| 2026-09-01 | EXP-1027-C 35 W conservative frontier point | The 40-warm-up/100-step run had not completed after more than 240 seconds and was manually interrupted under the cheap-kill rule. Atomic result output was therefore never published. | Console record; config `configs/molt-stream-sustainable-35w.json` | Unverified and not production-ready. Add bounded benchmark timeout/progress before repeating. |
| 2026-09-02 | EXP-1028 zoned/steady thermal cruise | A 30-step zone probe looked successful at 126,920 tokens/s and 68-70 C, but longer runs falsified promotion. The 75 C profile recorded SW thermal slowdown and 0.871 stability; a 72 C profile still throttled; 40/50 ms steady-duty removed throttle but delivered only 72,295/68,432 aggregate tokens/s with stability below 0.97. | `production-throughput-20260902-075735.json`, `075926.json`, `080046.json`, `080305.json`, `080544.json`, `080733.json` | Retain UI, checkpoint safety, telemetry and experimental profiles. Reject the claim that the current laptop sustains 80-100k at the requested thermal/stability gates. |

Entries are appended; they are not removed when a later revision succeeds.

# 2026-09-02 — EXP-1016 sequence-length warmup

- Three paired seeds, alternating AB/BA order, fixed model and 8,192
  tokens/update, context-512 held-out target within 1% of each baseline.
- Median improvement: 0.59% time and 3.23% GPU-board energy.
- Bootstrap 95% intervals: time [−20.13%, +15.15%], energy
  [−19.40%, +11.09%]. One seed regressed by about 20% in both metrics.
- Every arm peaked at 75–76 C and recorded the software thermal slowdown bit.
- Decision: reject. The result is neither repeatable nor thermally eligible.
- Artifact: `artifacts/molt-stream/experiments/exp-1016-20260902-084502.json`.

# 2026-09-02 — EXP-1030 exact-loss partitioning

- A clean-room, Windows-compatible exact token-partitioned linear cross-entropy
  primitive matched float64 loss and all gradients in unit tests. BF16 maximum
  relative gradient error was below 0.22% in GPU probes.
- On the 10.78M workload, chunk 1,024 reduced peak allocation 34.07% but slowed
  updates 9.03%. On the 58.01M workload, it reduced allocation only 16.11%,
  slowed updates 6.69%, and still used 5.18 GB.
- The short energy samples were sequential and temperature-biased, so they are
  retained as telemetry but are not treated as energy comparisons.
- Normal Liger installation failed because its declared Linux Triton wheel is
  unavailable on Windows. A no-dependency wheel installed zero-filled Python
  files and could not import; it was removed from the virtual environment.
- Decision: retain the exact primitive and benchmark as experimental memory
  infrastructure; reject Milestone 3 time/energy promotion.
- Artifacts: `exp-1030-20260902-091125.json` and
  `exp-1030-20260902-091225.json`.
# 2026-09-02 — fixed steady-duty pacing for EXP-1016

- Configuration: 30 ms pause after every update, 60 C target, 70 C protective
  boundary, 75 C abort; identical 10.78M fixed-token workload.
- Baseline completed at 50,060 tokens/s and 72 C peak with no NVML thermal
  throttle bit.
- The sequence-warmup candidate reached 56,974 tokens/s but crossed the abort
  boundary at 76 C after 42 updates, so it produced no valid time-to-quality
  result.
- Decision: reject fixed 30 ms duty pacing. It loses the 80k throughput gate
  and does not guarantee the configured thermal boundary. An instantaneous
  starting-temperature check was also insufficient to normalize chassis heat;
  paired experiments now require a five-second stable dwell in the start bin.
