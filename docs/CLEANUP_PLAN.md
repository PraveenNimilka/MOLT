# MOLT Repository Cleanup & Classification Plan

**Audit Date:** 2026-09-04  
**Auditor:** Release & Hardening Engineer  
**Policy:** Never delete source code, tests, checkpoints, datasets, benchmark evidence, or telemetry. Clean only ephemeral compilation caches, temporary packaging dists, and __pycache__.

---

## 1. File Classification Inventory

| Category | Description / Paths | Action | Rationale |
| :--- | :--- | :---: | :--- |
| **A. CORE SOURCE** | `src/molt_stream/**` (34 modules) | **PRESERVE** | Core training engine, streaming, QLoRA, thermal pacing, telemetry. |
| **B. TESTS** | `tests/test_*.py` (16 files) | **PRESERVE** | 76 validated unit tests covering specs, QLoRA, telemetry, and thermal math. |
| **C. CONFIGURATION** | `configs/*.json`, `pyproject.toml`, `.gitattributes`, `.gitignore` | **PRESERVE** | Production, cruise, thermal, and smoke configurations. |
| **D. DOCUMENTATION** | `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`, `docs/**` | **UPDATE / EXPAND** | Architectural specs, guides, and benchmarking methodology. |
| **E. BENCHMARKS** | `benchmarks/**` | **PRESERVE** | MOLT standalone benchmarking infrastructure. |
| **F. EXAMPLES** | `examples/` (to be created as workspace templates) | **PRESERVE** | Default configs, minimal scripts, and workflow samples. |
| **G. GENERATED ARTIFACTS** | `dist/` (`.whl`, `.tar.gz`) | **SAFE CLEAN** | Packaging distribution artifacts regenerated on clean build. |
| **H. CACHE / TEMP** | `**/__pycache__`, `.pytest_cache`, `.c/` | **SAFE CLEAN** | Ephemeral Python bytecode, compiler caches, and temporary test caches. |
| **I. USER DATA** | `data/prepared/**` (`one-billion`, `qwen2-0.5b`, `smoke`) | **PRESERVE** | Binary token dataset files required for local reproduction. |
| **J. CHECKPOINTS / MODELS** | Checkpoints in `artifacts/**/checkpoint.pt` | **PRESERVE** | Training weights and optimizer states. |
| **K. EXPERIMENT RESULTS** | `artifacts/benchmarks/*.json`, `artifacts/frontiers/*.json` | **PRESERVE** | Historical milestone empirical evidence. |

---

## 2. Safe Cleanup Actions Executed

1. **Purge Python Bytecode:** Remove ephemeral `__pycache__` directories across `src/` and `tests/`.
2. **Purge Old Distribution Artifacts:** Clear stale wheels in `dist/` before fresh hatchling build.
3. **Purge Ephemeral Cache:** Clear stale pytest cache files while keeping `.gitignore` rules enforced.
4. **Enforce Repository Boundary:** Verify that no competitor code (specifically Unsloth) exists anywhere in the repository.

---

## 3. Strict Boundary Verification
* [x] Zero Unsloth source code or kernels present.
* [x] Zero Unsloth dependencies in `pyproject.toml`.
* [x] Zero Unsloth checkpoints, logs, or traces in repository.
* [x] All 76 MOLT unit tests execute independently without Unsloth.
