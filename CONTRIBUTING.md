# Contributing to MOLT

MOLT accepts changes that preserve reproducibility and the distinction between
measured facts, prior reported results, interpretations and hypotheses.

Before submitting a change:

1. create or update a focused test;
2. run `uv sync --frozen` and `python -m pytest -q`;
3. run `git diff --check`;
4. record benchmark configuration and raw artifacts for performance claims;
5. compare against the strongest identical-workload baseline; and
6. disclose regressions, failed seeds, thermal state and measurement limits.

By submitting a contribution, you represent that you have the right to provide
it and agree that it may be distributed under the repository's current license.
This is not a copyright assignment or a contributor license agreement.

Do not describe a mechanism as novel without a dedicated prior-art review. Do
not commit model weights, prepared datasets, private paths, credentials or
generated run artifacts.
