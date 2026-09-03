# MOLT Benchmarking Methodology & Empirical Evidence

MOLT includes a standalone, reproducible benchmarking suite for evaluating LLM fine-tuning performance, thermal equilibrium, energy consumption, and memory footprint on consumer hardware.

---

## 1. Benchmarking Methodology

MOLT evaluates training systems across five empirical dimensions:

1. **Throughput (Tokens/s):**
   * **Wall-Clock Throughput:** Total processed tokens divided by total elapsed wall time ($T_{\text{compute}} + T_{\text{cooling}}$).
   * **Active Compute Throughput:** Total tokens divided by active forward/backward kernel compute time ($T_{\text{compute}}$).
2. **Thermal Dynamics (°C):**
   * 10 Hz NVML sampling of GPU core temperature, throttling reason masks (`reasons & 0x20` / `0x40`), and thermal equilibrium plateaus.
3. **Energy Efficiency (Joules & Wh / 1M Tokens):**
   * Real-time trapezoidal integration of calibrated NVML board power ($P(t)$):
     $$E = \int_0^T P(t) dt$$
   * Normalized energy consumption expressed in Watt-hours per 1,000,000 processed tokens.
4. **VRAM Footprint (GB):**
   * Peak PyTorch CUDA allocated memory vs. total GPU driver reserved memory.
5. **Convergence & Numerical Quality:**
   * Validation cross-entropy loss (NLL) and perplexity ($PPL = e^{\text{NLL}}$) on held-out token sequences.

---

## 2. Benchmark 001 Empirical Results `[MEASURED]`

**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM, 100W Enforced TGP)  
**Workload:** 4-bit NF4 QLoRA on Qwen2-0.5B (494M Base Parameters), 1,000,000 Contiguous Tokens  

| Framework / Configuration | Wall Speed | Compute Speed | Mean Power | Peak Temp | Steady Temp | Peak VRAM | Wh / 1M Tokens | Val PPL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MOLT (Thermal ON)** | **`1,376.7 tok/s`** | **`1,952.4 tok/s`** | 49.07 W | **`80.0°C`** | **`71.0–74.0°C`** | 6.89 GB | **9.90 Wh** | 6.325 |
| **MOLT (Thermal OFF)** | **`1,924.1 tok/s`** | **`1,956.6 tok/s`** | 57.01 W | 83.0°C | 82.0–83.0°C | 6.92 GB | **8.23 Wh** | 6.325 |
| **PEFT (Hugging Face)** | 1,070.9 tok/s | 1,071.0 tok/s | 49.42 W | **85.0°C** | **85.0°C (Throttled)** | **7.97 GB** | 13.19 Wh | 5.926 |
| **UNSLOTH (External Ref)** | **`3,027.8 tok/s`** | **`3,028.7 tok/s`** | 58.46 W | 78.0°C | 75.0–78.0°C | 4.80 GB | **5.79 Wh** | 6.399 |

---

## 3. Key Findings

1. **MOLT vs. Industry Standard (Hugging Face PEFT):**
   * MOLT Dual-Gear is **+28.6% faster** in wall-clock time and **+82.3% faster** in active compute throughput than PEFT.
   * MOLT uses **25.0% less electrical energy** (32.0 kJ vs. 42.7 kJ).
   * PEFT pinned the laptop GPU at 85.0°C for 700+ consecutive steps, suffering a -6.5% thermal downclocking decay. MOLT maintained a safe 71–74°C cruise.
   * PEFT pushed VRAM to **7.97 GB / 8.00 GB** (verge of crash); MOLT preserved a **1.08 GB safety margin**.
2. **Thermal Invariance:**
   * `MOLT_THERMAL_ON` and `MOLT_THERMAL_OFF` produced **identical validation loss (1.84454)**, proving that thermal duty cycling causes zero numerical drift or convergence degradation.

---

## 4. Repository Boundary Policy

* **Independence:** The MOLT repository contains its own standalone benchmark methodology, configs, and runners.
* **External Reference:** Unsloth and other commercial engines may be evaluated as external comparative baselines, but **must remain strictly outside the MOLT repository**.
* **Zero Dependency:** MOLT installs, runs, and benchmarks without any external competitor code.
