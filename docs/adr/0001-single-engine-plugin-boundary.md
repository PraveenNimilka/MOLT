# ADR-0001: Single stable training engine with experimental method hooks

- Status: Accepted for Milestone 0 proposal
- Date: 2026-09-01

## Context

Experiments must change one variable without drifting data order, metrics,
checkpointing, or loop semantics. Copying a training script per method makes
comparisons invalid and fixes hard to propagate. Importing experiments into the
stable engine couples production behavior to unfinished research.

## Decision

Implement one training state machine in `training/`. Experimental algorithms in
`methods/` implement narrow, versioned hooks and serializable state declared by
stable protocols. The engine owns data iteration, accumulation, phase events,
evaluation, interruption, and checkpointing. `core/` and `training/` never
import a concrete method.

## Consequences

- Baseline and candidate share operational semantics.
- Hook design requires early discipline and compatibility validation.
- A method that cannot fit the hooks must justify a new stable capability in a
  separate ADR; it may not fork the training loop by default.
- Experimental API churn is isolated, while stable checkpoint and metric schemas
  remain testable.

## Rejected alternatives

- Independent scripts per paper: fast initially, but comparison drift and
  duplicated failure handling are unacceptable.
- General workflow/plugin framework now: unnecessary dependency and abstraction
  cost before one baseline exists.
- Put all algorithms in the engine: creates conditional complexity and reverses
  the stable-to-experimental dependency rule.
