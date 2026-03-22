# J48 Fast Validation Plan

This document defines the validation campaigns for `j48.fast`.

The goal is to keep the plan easy to reason about:

1. prove that `fast` preserves the behavior of `strict`,
2. verify that `fast` remains aligned with `WEKA J48`,
3. and then measure performance separately under controlled timing conditions.

## Campaigns

### F1: `fast` vs `strict`

Purpose:

- validate semantic preservation,
- detect any regression introduced by performance work,
- and keep `strict` as the behavioral baseline.

Execution:

- Layer A with the full synthetic expanded plan
- Layer B with the extended real-tabular manifest
- Layer C with the IDS manifest

Default seed policy:

- deterministic configs: `1`
- `rep`: the fixed `50`-seed paper seedset

Runners:

- [run_j48_fast_vs_strict_synthetic_matrix.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_fast_vs_strict_synthetic_matrix.sh)
- [run_j48_fast_vs_strict_matrix.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_fast_vs_strict_matrix.sh)

### F2: `fast` vs `WEKA J48`

Purpose:

- confirm that the fast backend does not drift away from the external WEKA reference,
- and preserve the comparative narrative of the project.

Execution:

- Layer B with the extended real-tabular manifest
- Layer C with the IDS manifest

Default seed policy:

- deterministic configs: `1`
- `rep`: the fixed `50`-seed paper seedset

Runner:

- [run_j48_fast_vs_weka_matrix.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_fast_vs_weka_matrix.sh)

### F3: performance timing

Purpose:

- measure `fit`, `predict`, `predict_only`, and `predict_proba`,
- report steady-state timing,
- and avoid polluting semantic runs with timing-specific choices.

Execution:

- Layer B and Layer C
- serial repeats
- warm predict kernels after `fit`

Runner:

- [run_j48_fast_timing_matrix.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_fast_timing_matrix.sh)

## Default campaign sizes

### F1 synthetic

- Layer A expanded plan: `441` runs

### F1 extended

For `8` Layer B datasets and `5` Layer C datasets:

- deterministic configs: `7`
- `rep`: `50` seeds

Counts:

- Layer B: `8 x (7 + 50) = 456`
- Layer C: `5 x (7 + 50) = 285`
- total: `741`

### F2 extended

Same shape as F1 extended:

- total: `741`

### F3 timing

With `13` datasets and `10` repeats:

- total: `130`

## Acceptance gates

### F1

- exact prediction match
- exact or near-zero probability delta
- no regression on the previous green cases

### F2

- same metrics used for the strict-vs-WEKA campaign
- equivalence thresholds remain those defined in
  [J48_ACCEPTANCE_CRITERIA.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_ACCEPTANCE_CRITERIA.md)

### F3

- report medians and IQR
- keep the timing campaign serial
- do not mix semantic and timing conclusions
