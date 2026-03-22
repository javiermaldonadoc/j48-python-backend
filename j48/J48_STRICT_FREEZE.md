# J48 Strict Freeze

This note freezes the current `strict` line as the semantic baseline before
performance work.

## Freeze basis

Current extended local validation batch:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_extended_batches/j48_extended_all_20260312_091943/layer_c_local/j48_extended_20260312_091943/summary.csv)

Current external-manifest sweep:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_extended_batches/j48_extended_all_20260312_091943/layer_b_manifest/j48_extended_20260312_092042/summary.csv)

Current synthetic expanded `Layer A` campaign:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_112530/summary.csv)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_114212/summary.csv) (`A3` + `A5` focused rerun after the `binary_nominal` routing fix)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_114644/summary.csv) (final full rerun after the `binary_nominal` routing fix)
- [J48_SYNTHETIC_EXPANDED_RESULTS.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_SYNTHETIC_EXPANDED_RESULTS.md)

## Frozen status

The `strict` line is frozen for semantic work under the current local
validation scope.

Exact or effectively exact in the extended local batch:

- `base`
- `unpruned`
- `no_subtree_raising`
- `no_collapse`
- `laplace`
- `no_mdl`

Residual low-impact differences remain only in:

- `NSL-KDD + no_split_actual`: `0.999`, `1` mismatch per seed, matching node counts
- `NSL-KDD + rep`: `0.999`, `1` mismatch per seed, matching node counts
- `CIC-IDS2017 + rep`: `0.997`, `3` mismatches per seed, node counts `67 vs 65`

These residuals are documented and currently treated as acceptable for the
`strict` freeze because:

- they are deterministic and stable across seeds,
- they do not affect the mainline configuration,
- they are concentrated in narrow `REP` and `no_split_actual` edge cases,
- and further global tuning risks breaking configurations that are already exact.

The synthetic `-B` residuals isolated by the first expanded `Layer A` run are
no longer part of the frozen residual set because the focused `A3` + `A5`
rerun closes them without changing the already exact controlled blocks, and
the final full rerun preserves exact agreement across the whole synthetic suite.

## Interpretation

The current `strict` line is suitable as:

- the semantic baseline for synthetic edge-case validation,
- the software baseline for paper discussion,
- and the reference implementation to profile before speed work.

## Next step after freeze

Immediate next work should focus on:

1. synthetic edge-case validation,
2. external tabular validation once `layer_b` datasets are available,
3. then performance profiling and optimization.
