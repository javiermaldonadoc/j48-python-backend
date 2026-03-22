# J48 Synthetic Edge Battery

This battery isolates boundary conditions that are hard to audit with the
larger IDS datasets.

The purpose is not to replace the real-data validation already completed in the
extended local batch. It is to probe whether the frozen `strict` line behaves
like `WEKA J48` on small, controlled cases where the source of any divergence is
easy to identify.

## Scope

The initial synthetic suite covers:

- deterministic numeric tie cases,
- split-point relocation to observed values,
- missing-value routing in train and test,
- nominal multi-way behavior,
- unseen nominal values declared in the domain but absent from train rows,
- binary nominal splits with `-B`,
- reduced-error pruning on small trees,
- and tiny leaves around `minNumObj`.

## Layout

Generated data lives under:

- [generated](/home/javier/VisualStudio/IDEA/tests/data/j48_synthetic/generated)

The suite is driven by:

- [generate_j48_synthetic_cases.py](/home/javier/VisualStudio/IDEA/scripts/analysis/generate_j48_synthetic_cases.py)
- [j48_synthetic_validation_manifest.csv](/home/javier/VisualStudio/IDEA/scripts/analysis/j48_synthetic_validation_manifest.csv)
- [j48_synthetic_expanded_plan.csv](/home/javier/VisualStudio/IDEA/scripts/analysis/j48_synthetic_expanded_plan.csv)
- [run_j48_synthetic_matrix.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_synthetic_matrix.sh)
- [J48_SYNTHETIC_EXPANDED_PLAN.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_SYNTHETIC_EXPANDED_PLAN.md)

## Initial case list

1. `numeric_tie_first_feature`
2. `numeric_split_actual_value`
3. `missing_fractional_train_test`
4. `nominal_multiway_basic`
5. `nominal_unseen_test_value`
6. `nominal_binary_grouping`
7. `rep_single_prune`
8. `minnumobj_tiny_leaf`

## Validation intent

Each case should be interpreted primarily through:

- `prediction_match_fraction`,
- `prediction_mismatch_count`,
- `probability_mean_abs_delta`,
- `local_node_count` vs `weka_node_count`,
- and the first structural divergence if any.

The suite is meant to answer:

- whether a residual is a general semantic issue,
- or whether it is a narrow artifact of a specific real dataset branch.

## Current initial-batch result

Latest initial synthetic run:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic/j48_synth_20260312_110222/summary.csv)

Summary:

- `32/32 OK`
- `base`: `8/8` exact
- `no_split_actual`: `8/8` exact
- `rep`: `7/8` exact
- `binary_nominal`: `6/8` exact

This means the pipeline is ready to scale into the expanded plan, and it
already highlights the two most informative residual families:

- tiny `REP` cases,
- and `binary_nominal` cases.
