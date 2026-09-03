# MOLT Developer Guide

Guidelines for contributing to the MOLT architecture, maintaining strict boundaries, and running automated tests.

---

## 1. Development Setup

```powershell
git clone https://github.com/PraveenNimilka/MOLT.git
cd MOLT
pip install -e ".[dev]"
```

---

## 2. Running Automated Tests

Run the full pytest suite:

```powershell
pytest
```

Run CLI and discovery tests:

```powershell
pytest tests/test_molt_stream_cli_v1.py
```

Check architectural import boundaries:

```powershell
pytest tests/test_molt_stream_boundaries.py
```

---

## 3. Architectural Boundary Standards

MOLT enforces strict inward layer boundaries (`test_molt_stream_boundaries.py`):
```
core (0) <── data, kernels, measurement, experiments (1) <── streaming (2) <── training (3) <── cli (4)
```
* Modules at a lower rank cannot import from modules at a higher rank.
* `core` cannot import from `experiments` or `training`.

---

## 4. Competitor Isolation Policy

**UNSLOTH MUST REMAIN COMPLETELY OUTSIDE THE MOLT REPOSITORY.**
* Do not install Unsloth in the core MOLT environment.
* Do not add Unsloth to `pyproject.toml`.
* Do not import Unsloth in any MOLT module.
* Competitor benchmarking must be conducted in isolated external directories.
