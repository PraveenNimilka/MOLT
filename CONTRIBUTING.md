# Contributing to MOLT

MOLT accepts focused changes that preserve correctness, reproducibility, and a
clear distinction between measured facts, reported results, interpretations,
and hypotheses.

## Supported development environment

- Windows 10/11 x64
- Python 3.12
- The dependency versions resolved by `uv.lock`
- NVIDIA hardware only for tests explicitly marked as requiring CUDA

Set up the repository and run the core checks:

```powershell
uv sync --frozen
uv run ruff check --select E9,F63,F7,F82 src tests
uv run python -m pytest -q
git diff --check
```

The critical Ruff rule set is enforced in CI. The repository still contains
legacy formatting debt, so whole-tree `ruff format --check` is not yet a merge
gate. New or substantially rewritten Python files should be formatted with:

```powershell
uv run ruff format path\to\changed_file.py
```

Do not mix a repository-wide formatting rewrite with a behavioral change.

## Pull requests

Use a focused branch and one concern per pull request. The PR description must
state:

- the user-visible or engineering problem;
- the chosen boundary and important alternatives rejected;
- tests added or changed;
- exact verification commands and results; and
- compatibility, performance, memory, thermal, or security risks.

Use imperative commit subjects such as `fix: reject unsafe checkpoint input` or
`test: cover interrupted update recovery`. Do not include generated artifacts,
private paths, credentials, datasets, model weights, or large binaries.

## Performance and research changes

Performance claims require the workload contract, model and data hashes, raw
local artifacts, arm order, seeds, hardware state, quality objective, confidence
intervals, and all failed or excluded trials. Compare against the strongest
identical-workload baseline. Keep raw artifacts outside Git unless a maintainer
explicitly approves a compact public fixture.

Do not describe a mechanism as novel without a dedicated prior-art review. Do
not turn a diagnostic screen into a release claim when a registered promotion
gate failed.

## Licensing and conduct

By submitting a contribution, you represent that you have the right to provide
it and agree that it may be distributed under the repository's current license
and any commercial licenses issued by the maintainer. This is not a copyright
assignment or contributor license agreement.

Be precise, respectful, and evidence-led. Security reports belong in the private
channel documented in [SECURITY.md](SECURITY.md), not in a public issue.
