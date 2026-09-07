# Public benchmark evidence

The checked-in figure is generated from
[`0.11.0a3-diagnostic.json`](0.11.0a3-diagnostic.json). It summarizes three
six-pair development-system screens without publishing implementation-specific
profiling records.

For each pair, elapsed-time and energy reductions are calculated as
`100 × (Unsloth − MOLT) / Unsloth` at the registered target NLL. The displayed
uncertainty is a two-sided 95% paired t-interval over six AB/BA pairs
(`df = 5`). Allocator reduction compares the mean PyTorch peak allocation for
the two arms; the observed allocator peak was constant within each family, so
no sampling interval is shown for that panel.

Positive values favor MOLT. None of these screens passed the official promotion
gate because graphics clocks were not controlled. The Gemma screen also contains
one power-envelope-invalid pair. Soup, 7B/8B endurance, and independent
reproduction remain open.

Regenerate the SVG from the repository root:

```powershell
py -3.12 tools\render_benchmark_summary.py
```

The JSON includes SHA-256 identifiers for the local result aggregate and the
registered workload contract used to calculate each row.
