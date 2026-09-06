# Commercial-readiness scorecard

Scores are evidence summaries, not targets that can be declared complete by
editing code. A 10/10 requires every listed gate and independent confirmation.

| Dimension | Current | Evidence gained | 10/10 gate still required |
|---|---:|---|---|
| Research prototype | 8.5/10 | Working pretrain/QLoRA paths, telemetry, atomic recovery, comparison CLI, positive and negative experiments | Two-model/two-workload reproduction, complete fault matrix, sustained runs |
| Source-available alpha | 8/10 | PolyForm Shield license, lockfile, Windows CI, security/contribution policy, built wheel/sdist, successful clean-environment wheel/CLI audit | Legal review, support matrix, external user install, independent reproduction |
| Scientific value | 8/10 | Preregistered gates, tuning/holdout separation, deterministic paired intervals, three-seed early-quality result, explicit context/longer-horizon falsification, negative-results log | Statistical power beyond three seeds, multiple tasks/scales/machines, wall-energy validation, independent reproduction |
| Engineering originality | 6.5/10 | Local-first evidence/thermal/recovery integration and exact comparison guards | Demonstrated technically distinct systems mechanism that beats mature alternatives across workloads |
| Algorithmic novelty | 2/10 | Original candidates were precisely tested and rejected | Completed prior-art search plus a technically distinct algorithm, ablations, theory, multi-scale gains, independent reproduction |
| Commercial readiness | 5/10 | Safer checkpoint loading, typed configs, CI, security policy, reproducible artifacts, portable paths, clean wheel audit | Stable API/versioning, GPU installer, licensing/SBOM audit, signed releases, telemetry privacy, support/upgrade plan, long-soak tests |
| Defensible performance advantage | 3.5/10 | 37.17% median time and 48.76% board-energy gain at one early context-256 target; context-512 and medium-horizon transfer failed | Advantage over strongest competitor implementations on several valuable workloads; long-run and external reproduction; defensible IP or execution moat |
| Long-term potential | 8/10 | Strong experimental discipline and a useful consumer-GPU test platform | Evidence that improvements transfer and compound without regressions |

## Non-negotiable 10/10 program

1. Freeze public benchmark manifests and competitor versions before runs.
2. Reproduce on at least two model families, three sizes, three datasets/tasks,
   and two independently administered machines.
3. Use at least five seeds for primary statistics and publish all raw runs.
4. Measure wall energy alongside NVML board energy and quantify sampler overhead.
5. Run 30–120 minute thermal/throughput soaks and interruption/OOM/corruption
   recovery campaigns.
6. Compare with tuned PEFT/Transformers, Unsloth, Axolotl/LLaMA-Factory where
   supported; disclose unsupported Windows paths rather than scoring them zero.
7. For any original method, publish mathematics, pseudocode, complexity,
   correctness tests, ablations, prior-art mapping, and a falsification package.
8. Obtain an external reproduction before using “breakthrough,” “unique,” or
   “commercially defensible algorithm” in public claims.
9. Complete package signing, SBOM/license review, vulnerability response,
   compatibility matrix, migration policy, and clean-host installer testing.
10. Promote only a simple set of independently validated improvements; keep
    failed or narrow techniques experimental.

The Qwen LoRA+ result is a strong baseline improvement but is established prior
art. It therefore cannot raise algorithmic novelty to 10/10. MOLT reaches that
score only by discovering and independently validating a distinct mechanism.
