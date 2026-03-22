# Phase 1 Status

This file tracks the `WEKA J48 -> local support` map for the strict line.

Active work guide:

- [J48_WEKA_CLOSURE_BACKLOG.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_WEKA_CLOSURE_BACKLOG.md)
- [J48_STRICT_FREEZE.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_STRICT_FREEZE.md)
- [J48_SYNTHETIC_EXPANDED_RESULTS.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_SYNTHETIC_EXPANDED_RESULTS.md)

## Supported in the local API

| WEKA option | Meaning | Local status |
| --- | --- | --- |
| `-U` | Unpruned tree | Supported |
| `-C` | Confidence factor | Supported |
| `-M` | Minimum number of instances per leaf | Supported |
| `-S` | Disable subtree raising | Supported |
| `-O` | Disable collapse | Supported |
| `-A` | Laplace smoothing | Supported |
| `-J` | Disable MDL correction | Supported |
| `-Q` | Seed | Supported |
| `-doNotMakeSplitPointActualValue` | Disable split-point relocation to observed value | Supported |
| `-R` | Reduced-error pruning | Supported |
| `-N` | Number of folds for reduced-error pruning | Supported |
| `-L` | Do not clean up after building the tree | Supported |
| `-B` | Binary splits on nominal attributes | Supported |

## Current Phase 1 priorities

1. Treat the current `strict` line as frozen for semantic work under the local validation scope.
2. Use synthetic edge-case validation to probe general boundary behavior without perturbing the frozen baseline.
3. Expand validation outside IDS once `layer_b` data is available locally.
4. Move to performance work only after the synthetic edge-case battery is in place.

## Current matrix snapshot

Latest extended local validation batch:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_extended_batches/j48_extended_all_20260312_091943/layer_c_local/j48_extended_20260312_091943/summary.csv)

Latest external-manifest sweep:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_extended_batches/j48_extended_all_20260312_091943/layer_b_manifest/j48_extended_20260312_092042/summary.csv)

Latest synthetic expanded `Layer A` batch:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_112530/summary.csv)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_114212/summary.csv) (`A3` + `A5` focused rerun after the `binary_nominal` routing fix)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_114644/summary.csv) (final full rerun after the `binary_nominal` routing fix)
- [J48_SYNTHETIC_EXPANDED_RESULTS.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_SYNTHETIC_EXPANDED_RESULTS.md)

Latest full matrix:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_phase1_matrix200_20260311_153847/summary.csv)

Latest non-default refresh:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_phase1_nondefault_refresh_20260311_170104/summary.csv)
- [comparison_vs_matrix200.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_phase1_nondefault_refresh_20260311_170104/comparison_vs_matrix200.csv)

Latest focused `REP` rerun:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_rep_rerun_20260311_161614/summary.csv)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_rep_rerun_probfix_20260311_163516/summary.csv) (`J48-009` probability fix on empty nominal branches)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_rep_refresh_20260311_170955/summary.csv) (`REP` refresh after pruning unsupported holdout branches)

Latest focused `unpruned` checks:

- [/tmp/j48_unpruned_nsl_seed1_after_leafcheck/comparison.json](/tmp/j48_unpruned_nsl_seed1_after_leafcheck/comparison.json)
- [/tmp/j48_unpruned_cic_seed1_after_leafcheck/comparison.json](/tmp/j48_unpruned_cic_seed1_after_leafcheck/comparison.json)
- [/tmp/j48_unpruned_unsw_seed1_after_leafcheck/comparison.json](/tmp/j48_unpruned_unsw_seed1_after_leafcheck/comparison.json)

Latest focused `no_subtree_raising` checks:

- [/tmp/j48_nosr_nsl_seed1/comparison.json](/tmp/j48_nosr_nsl_seed1/comparison.json)
- [/tmp/j48_nosr_cic_seed1/comparison.json](/tmp/j48_nosr_cic_seed1/comparison.json)
- [/tmp/j48_nosr_unsw_seed1_after_adderrs/comparison.json](/tmp/j48_nosr_unsw_seed1_after_adderrs/comparison.json)

Highlights:

- The extended local batch reproduces the latest `matrix200` exactly (`changed_rows = 0`) and gives `200/200 OK`.
- `base` reaches `1.000` match on all five datasets and five seeds.
- `CIC-IDS2017`, `NSL-KDD`, `UNSW-NB15`, `CIRA2020-AttNorm`, and `CIRA2020-DoHNDoH` all match WEKA structurally in `base`.
- `REP` now reaches `0.991` on `CIC-IDS2017`, `0.999` on `NSL-KDD`, `0.995` on `UNSW-NB15`, and `1.000` on both `CIRA2020` variants across the 5-seed rerun.
- `J48-009` reduced `REP` probability deltas from `0.0105 -> 0.00025` on `UNSW-NB15` and from `0.00080 -> 0.00029` on `NSL-KDD` without changing prediction match.
- Pruning branches with no holdout support moves `REP` to `1.000` on `UNSW-NB15` with matching node counts (`473 vs 473`).
- Focused `unpruned` reruns now also hit `1.000` match with matching node counts on `NSL-KDD`, `CIC-IDS2017`, and `UNSW-NB15` for seeds `1`, `7`, and `13`.
- Fixing `addErrs` for singleton perfect leaves closed the remaining focused `no_subtree_raising` gap on `UNSW-NB15`, giving `1.000` match with matching node counts for seeds `1`, `7`, and `13`.
- The non-default refresh now also aligns `unpruned` and `no_subtree_raising` across all five datasets and five seeds.
- The only remaining local residuals are tiny threshold/context drifts in `CIC-IDS2017 + rep`, `NSL-KDD + rep`, and `NSL-KDD + no_split_actual`.
- Those residuals are documented but no longer block freezing the `strict` line for semantic work.
- The synthetic expanded `Layer A` campaign completes `441/441` paired runs and initially isolates only three non-exact rows, all under `binary_nominal`.
- The focused `A3` + `A5` rerun closes those residuals and gives `69/69` exact paired runs after the `binary_nominal` routing fix.
- The final full rerun confirms `441/441` exact paired runs across the whole synthetic expanded campaign.
- `REP` is exact in `400/400` synthetic runs across the fixed `50`-seed paper set.
- Numeric, missing-value, pruning, and standard probability behaviors are exact in the controlled suite, and the previously isolated `-B` residuals are now closed in the final full rerun.
