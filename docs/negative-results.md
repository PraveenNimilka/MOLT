# Failure and negative-results log

This append-only log was created before the first experiment so rejected
hypotheses and operational failures would have a durable home.

| Date | Experiment | Outcome | Evidence | Decision |
|---|---|---|---|---|
| 2026-09-06 | Qwen2.5-1.5B midpoint intra-step synchronization (all 28 layers, boundary 14, 10 ms, B1/G4, context 512) | Exact final NLL was preserved (1.533068), but compute fell from 1,710 to 1,566 tok/s, end-to-end fell from 613 to 492 tok/s, peak temperature still reached 72 C, and one 1,536-token partial microbatch was rolled back. | `artifacts/mars-1p5b-20260904/intra-step-10ms-screen/20260906-103011-qlora-b80df5cc/metrics.summary.json` | Reject as a performance default. Retain only as an opt-in exact-math safety checkpoint; test a verified hardware operating point instead. |
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
| 2026-09-06 | All-layer Qwen2.5-1.5B CPU activation offload | Exact pinned-host saved tensors reduced CUDA allocation to 2.635 GiB but reached only 502 compute tok/s and 6.50 GiB process RSS. | `artifacts/mars-1p5b-20260904/activation-offload/20260906-082436-qlora-51ad49df` | Reject: PCIe traffic and host memory are worse than recomputation. |
| 2026-09-06 | Packed INT4 saved activations | Short arms approached the memory gate with matching early NLL, but packing hundreds of tensors reduced compute and an endurance arm grew to 3.97 GiB before thermal stop. | `artifacts/mars-1p5b-20260904/int4-activations/` | Keep experimental only; investigate lifetime and long-run allocation before any promotion. |
| 2026-09-06 | Exact BF16 RMSNorm + stride-8 endurance | The short arm measured 1,727 compute tok/s at 2.956 GiB, but the 16-step arm completed only 15 steps after ten recoveries, sustained 401 end-to-end tok/s and peaked at 73 C. | `artifacts/mars-1p5b-20260904/bf16-rms-stride8/20260906-085632-qlora-48217507` | Reject production promotion. The 1,200 end-to-end thermal gate remains open. |
| 2026-09-01 | EXP-1026 60 C duty cycling at requested cool geometry | The laptop rejected the 65 W NVML request and remained at 115 W. With batch 16/accumulation 2, twelve 30-50 ms pauses could not prevent a 74 C peak; the run aborted after 8/30 measured steps at 115,173 tokens/s. | `artifacts/molt-stream/benchmarks/production-throughput-20260901-181655.json` | Reject the claim that 40 ms sleeps lock temperature at 60 C. Do not spend energy on the preregistered 500-step run. |
| 2026-09-01 | EXP-1027 predictive thermal cruise | Two compiled batch-32 probes reached 73-74 C during max-autotune/warm-up and aborted before a valid interval. Eager controller variants completed, but late/early throughput ratios were 0.881, 0.798, 0.945, 0.576 and 0.586; none passed the preregistered 0.97 stability gate. The best short safe interval was 103,858 tokens/s, peak 64 C, no throttle, ratio 0.945. | `production-throughput-20260901-203016.json` through `production-throughput-20260901-223807.json` | Retain controller instrumentation and safety fixes, but reject promotion as a solved cruise profile. Compilation heat and long thermal soak remain blockers. |
| 2026-09-01 | EXP-1027-C 35 W conservative frontier point | The 40-warm-up/100-step run had not completed after more than 240 seconds and was manually interrupted under the cheap-kill rule. Atomic result output was therefore never published. | Console record; config `configs/molt-stream-sustainable-35w.json` | Unverified and not production-ready. Add bounded benchmark timeout/progress before repeating. |
| 2026-09-02 | EXP-1028 zoned/steady thermal cruise | A 30-step zone probe looked successful at 126,920 tokens/s and 68-70 C, but longer runs falsified promotion. The 75 C profile recorded SW thermal slowdown and 0.871 stability; a 72 C profile still throttled; 40/50 ms steady-duty removed throttle but delivered only 72,295/68,432 aggregate tokens/s with stability below 0.97. | `production-throughput-20260902-075735.json`, `075926.json`, `080046.json`, `080305.json`, `080544.json`, `080733.json` | Retain UI, checkpoint safety, telemetry and experimental profiles. Reject the claim that the current laptop sustains 80-100k at the requested thermal/stability gates. |
| 2026-09-02 | Bounded asynchronous QLoRA launch window | Four-update CUDA launch windows delayed thermal observation. The first candidate reached 79 C, thermally aborted and regressed NLL 3.94%; subsequent heat-soaked pairs also aborted. | `artifacts/qwen2-0.5b/bounded-async-paired/` | Reject and remove the configuration surface; retain per-update synchronization for thermal safety. |
| 2026-09-02 | Qwen2 0.5B attention-only LoRA | Attention-only adapters cut trainable parameters from 4.40M to 1.08M and end-to-end time from 16.20s to 11.78s, but final held-out NLL regressed from 2.054 to 2.112 (+2.8%) and did not reach the baseline target. | `artifacts/qwen2-0.5b/attention-only-runs/20260902-155054-qlora-ee871401` | Reject as an equal-quality optimization; retain explicit module targeting as experimental infrastructure. |
| 2026-09-02 | Qwen2 0.5B all-linear LoRA rank 4 | The valid 20-update rerun took 12.09s versus 16.20s for rank 8, but final held-out NLL regressed from 2.054 to 2.080 (+1.27%), outside the 1% gate. A prior attempt thermally aborted after five updates; the valid run still peaked at 77 C and recorded thermal slowdown. | `artifacts/qwen2-0.5b/rank4-runs/20260902-183511-qlora-27c36e57` | Reject as a no-regression optimization; retain rank 8 as the strongest fixed-token baseline. |
| 2026-09-02 | BF16 LoRA adapters with FP32 paged AdamW state | A contemporaneous 20-update pair matched quality (candidate NLL improved 0.052%) and reduced allocation 4.09%, but the candidate was 5.51% slower end-to-end and 6.13% slower in update compute; board energy improved only 0.08%, within noise. Both arms used the same enforced 100 W limit and recorded thermal slowdown. | Candidate `artifacts/qwen2-0.5b/bf16-adapter-runs/20260902-184023-qlora-0c8b5094`; baseline `artifacts/qwen2-0.5b/resident-activation-runs/20260902-184129-qlora-a70554ce` | Reject and remove the added backend surface; keep FP32 adapters/AdamW as the tuned baseline. |
| 2026-09-02 | LoRA+ ratio 8 and preregistered ratio-16 early stop | Ratio 8 was effectively tied with vanilla at step 10. Across three ratio-16/vanilla pairs, median time and energy improved 38.46% and 50.13%, but seed 4099 regressed held-out NLL by 1.02% and two candidates missed the absolute 2.11 target. | Ratio-8 run `artifacts/qwen2-0.5b/loraplus-runs/20260902-184932-qlora-2bcc75c9`; ratio-16 paired runs dated 18:52–18:55 under `loraplus-runs` and `time-target-baseline-runs` | Reject ratio 8 and ratio 16 under the preregistered gate; any retuned schedule requires fresh holdout seeds. LoRA+ is established prior art, not a MOLT invention. |
| 2026-09-02 | EXP-1033 context-512 LoRA+ transfer | The frozen context-256 ratio-14/five-update schedule improved time 39.74% and board energy 50.84% at context 512, but NLL regressed 3.43%, peak temperature rose from 74 C to 77 C, and the candidate newly reported thermal throttling. | `artifacts/qwen2-0.5b/comparisons/loraplus-context512-kill-seed-1337.json` | Reject transfer and skip multi-seed extension. Narrow EXP-1032 to context 256; fixed LoRA+ schedules are not geometry-invariant. |
| 2026-09-02 | EXP-1035 ratio-14 medium horizon | At context 256 the candidate improved to NLL 2.105 at step 5 but rebounded to 2.153 at step 10, 5.15% worse than the completed 20-update vanilla reference. The contemporaneous vanilla control thermally aborted at step 6 after a 78 C heat-soaked peak, so no paired time/energy result is valid. | Candidate `artifacts/qwen2-0.5b/loraplus-medium-horizon-runs/20260902-200504-qlora-b3c18415`; aborted control `resident-activation-runs/20260902-200604-qlora-26e7abe1`; invalid comparison artifact `comparisons/loraplus-medium-horizon-kill-seed-1337.json` | Reject medium-horizon promotion. EXP-1032 is an early-quality transient only; stop further GPU trials until chassis state is normalized. |

Entries are appended; they are not removed when a later revision succeeds.

## 2026-09-06 — all-layer equilibrium and LoRA scheduling screens

- Separating the predictive cruise target from a 70 C microbatch safety guard
  did not stabilize a 50 W controller point: it stopped at 7/8 updates, peaked
  at 73 C, and discarded 5,632 uncommitted tokens. The committed updates still
  measured 1,712 tok/s, confirming that recovery churn—not the healthy update
  path—caused the 597 end-to-end tok/s result.
- A conservative 62 C cruise, 66 C microbatch guard, and 40 W average-power
  pacing target completed 8/8 all-layer updates with no recovery or discarded
  work. It reached the identical 1.533068 NLL and peaked at 70 C, but required
  12.22 seconds of pacing and sustained only 624 end-to-end tok/s.
- A packed NF4 backward schedule restricted to `down_proj` was rejected after
  full-model compute fell from 1,636 to 1,464 tok/s despite a small isolated
  component win.
- The first full-checkpoint BF16-shadow run exposed a correctness defect: fused
  AdamW did not advance the Python tensor version used for lazy shadow refresh,
  leaving stale compute weights and regressing NLL to 1.608671. Explicit
  post-step invalidation was added; the faulty run is invalid for performance
  comparison and retained at `full-checkpoint-shadow-equilibrium/`.
- Versioned BF16 LoRA compute shadows produced one 1,710 tok/s screen, but the
  first temperature-normalized AB pair failed when the shadow arm stopped at
  72 C without reaching the NLL target. The remaining pairs were killed because
  an all-pairs promotion result was no longer possible.
- Artifacts: `artifacts/mars-1p5b-20260904/equilibrium-screen/`,
  `equilibrium-conservative-screen/`, and `lora-shadow-ab-20260906/`.
- Decision: keep both scheduling mechanisms experimental. Retain the separated
  microbatch guard as safety/control infrastructure; do not claim a sustained
  Unsloth win.

## 2026-09-06 — matched all-layer Unsloth thermal kill screens

- A new matched runner uses Qwen2.5-1.5B, all 28 layers, 9,232,384 rank-8
  all-linear adapter parameters, context 512, B1/G4, FP32 fused AdamW, identical
  mmap token order, exact unfiltered cross-entropy, four validation windows and
  the same 72 C boundary.
- Unsloth stopped after 4 updates at 72 C under the 62/66 C controller, after 3
  updates in a colder repeat, after 6 updates with a 30 W pacing target, and
  after 2 updates with a 25 W target plus a 58 C microbatch guard. None reached
  the registered 1.54 NLL target.
- MOLT completed one normalized 8-update arm at 70 C and NLL 1.533068, but a
  later heat-soaked automated pair stopped both engines. A five-second surface
  temperature dwell was therefore rejected as insufficient chassis-state
  normalization.
- Full-checkpoint MOLT reached the same 1.533068 NLL at 1.944 GiB allocated and
  2.803 GiB NVML board-used memory, but a delayed 72 C sample marked the run as
  a thermal stop. The corrected adapter-shadow repeat also stopped and provided
  no speed advantage.
- The fused Triton-loss/BF16-RMS/stride-8 combination stopped after 5 updates at
  74 C and was slower than the analytical-loss reference.
- Decision: the short MOLT completion is a reliability screen, not an official
  Unsloth victory. Require a long rolling-window thermal soak and a power/clock
  operating point enforceable below the current 140 W ceiling before the
  three-seed endurance protocol.

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
