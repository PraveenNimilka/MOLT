# ADR-0003: Fixed byte vocabulary for the first baseline

- Status: Accepted for Milestone 0 proposal
- Date: 2026-09-01

## Context

A trained tokenizer adds preparation cost, randomness, artifacts, dependencies,
and a second experimental surface. The first baseline exists to validate system
measurement and method comparisons on one machine, not to maximize language
quality.

## Decision

BL-0001 predicts the next UTF-8 byte with a fixed 256-symbol vocabulary on a
pinned TinyStories slice and reports validation bits per byte.

## Consequences

Results are easy to hash and reproduce, but sequences represent less text than
subword sequences and quality does not transfer directly to normal LLMs. Before
generalizing a surviving method, reproduce it on at least one standard subword
model/dataset workload.
