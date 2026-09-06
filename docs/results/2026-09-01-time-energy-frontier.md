# MOLT 49.88M time–energy frontier and controller kill test

## Immutable comparison

The workload is the 49,883,520-parameter context-512 TinyStories BPE model with
seed 1337, 12,288 tokens/update, 719 AdamW updates, FP32 parameters and optimizer
state, identical sequential token stream, and the same context-512 validation
objective. The success boundary was fixed before candidate analysis at 1% above
the immutable baseline final NLL: `3.552706392854452`.

Energy-to-target is trapezoidal integration of sampled NVML GPU board power to
the linearly interpolated validation-NLL crossing. It excludes CPU, display and
charger losses. Each result is one seed and is therefore evidence for rejection
or continued study, not promotion.

| Run | Final NLL | Seconds to target | Board J to target | Interpretation |
|---|---:|---:|---:|---|
| Immutable BF16, high-power state | 3.517531 | 197.989 | 14,759.41 | Fastest |
| BF16 repeat, firmware-reduced state | 3.517940 | 263.553 | 10,247.23 | 30.57% less energy but 33.12% slower |
| FP16 + GradScaler, reduced state | 3.513439 | 277.433 | 10,205.77 | 0.40% less energy than reduced-state BF16 but 5.27% slower; not joint |

The runs form a tradeoff frontier because the laptop changed its enforced power
ceiling during the study. This is not evidence that MOLT caused the energy
reduction. A manual `nvidia-smi -pl` capability probe was rejected by the driver,
so safe user-set power points could not be swept.

## Isolated utilization findings

- Microbatch 24 / accumulation 1 was fastest and lowest-energy across the
  effective-batch-preserving geometry sweep.
- Indexed dense windows beat the tensor-equivalent contiguous implementation in
  the paired probe; the latter was 5.5% slower and 3.1% higher energy.
- FP32 was about 2x slower and required about 6.83 GB allocated memory.
- Unfused AdamW was slower. Forced Flash SDPA was unavailable in this Windows
  PyTorch build. Explicit efficient SDPA did not establish a repeatable joint
  gain over automatic dispatch.
- `torch.compile` remained unavailable because the pinned Windows environment
  has no supported Triton backend.

These are kill tests, not promoted benchmarks. Thermal/power state changed
between some probes, so only paired conclusions with sufficiently large effects
were used.

## Experimental controller

The implemented research rule estimates remaining time and energy to a loss
target for each semantically invariant action. It changes action only when an
uncertainty penalty still predicts at least 1% improvement in both objectives
and VRAM headroom is safe. Its proposed distinguishing feature was the joint use
of live loss progress and firmware-enforced power state, rather than optimizing
step throughput or selecting a static power/batch pair.

For action `a`, a measured block has loss progress
`d_a = max(0, loss_before - loss_after)`, duration `t_a`, and energy `e_a`.
At current loss `L` and target `L*`, the controller projects
`T_a = (L-L*) t_a/d_a` and `E_a = (L-L*) e_a/d_a`, using sample means and a
relative-standard-error penalty. It switches from `b` to `a` only if both
`T_a(1+u_a) <= 0.99 T_b` and `E_a(1+u_a) <= 0.99 E_b`. Non-positive progress,
insufficient exploration, or unsafe VRAM headroom prevents promotion. This rule
is intentionally conservative; short-window training loss nevertheless proved
too noisy for useful projections.

This is **not claimed novel**. Zeus already applies online exploration and
energy profiling to batch/power selection, while PowerTrain predicts time and
power across edge power modes. A complete patent and citation search was not
performed, and the local action set offered no surviving new behavior.

The 80-update live kill test compared the controller with static microbatch 24
at the same token/update budget and data order. The controller was 1.592% slower
and used 3.054% more GPU board energy. Its exploration selected microbatch 12 for
two blocks, then returned to 24. End losses were effectively equal. The
preregistered joint-improvement gate failed, so a costly full controller run was
not performed.

## Conclusion and next falsification

No repeatable joint time-and-energy improvement, no 2x result, and no original
mechanism have been demonstrated. The strongest honest result is a trustworthy
measurement frontier plus rejection of the current controller.

The next justified experiment is not another scheduler over dominated execution
shapes. It should change the available kernel/work graph while preserving the
mathematical update—for example a supported Linux/Triton fused training path—then
repeat paired runs in a controlled thermal state. Only after two actions exhibit
different Pareto optima across enforced-power bins should online control be
reopened.

## Reproduction

These commands record the retired `ai_local` prototype. They are preserved as
historical evidence and require the initial Git commit; they are not commands in
the current MOLT interface.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ai_local.cli utilization-probe --config configs\molt-ai-50m-frontier-bf16-control.json --micro-batches 24,12,8,6,4,3,2,1 --warmup-steps 3 --steps 20
.\.venv\Scripts\python.exe -m ai_local.cli controller-probe --config configs\molt-ai-50m-frontier-bf16-control.json --blocks 8 --steps-per-block 10
.\.venv\Scripts\python.exe -m ai_local.cli frontier "${MOLT_DATA_ROOT}\50M\runs\20260901-093650-MOLT-AI-50M-BPE-pretrain-v1-68ba7279" "artifacts\runs\20260901-114659-MOLT-AI-50M-frontier-BF16-control-43471575" "artifacts\runs\20260901-115215-MOLT-AI-50M-frontier-FP16-316c6567"
```
