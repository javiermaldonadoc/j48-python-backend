# J48 Synthetic Expanded Plan

This document turns the initial synthetic battery into an expanded validation
plan for `Layer A`.

The purpose of this expanded battery is to support a stronger paper discussion
and a more defensible claim about behavioral alignment with `WEKA J48` under
controlled conditions.

It does not replace the real-data layers. It complements them by isolating
general semantic behaviors that large datasets cannot diagnose cleanly.

## Design principles

1. Each synthetic case should test one main semantic mechanism.
2. Seeds should only be expanded where the mechanism is actually seed-sensitive.
3. Cases should be grouped by claim:
   - exact behavior expected,
   - near-exact behavior acceptable,
   - diagnostic-only exploratory cases.
4. The expanded suite should remain auditable: each failure should be easy to
   map back to a specific semantic block.

## Seed policy

The expanded synthetic plan uses two seed regimes:

- deterministic configurations:
  - `1` seed only
- stochastic configurations:
  - `50` seeds for `REP`

Deterministic configurations are:

- `base`
- `no_split_actual`
- `no_mdl`
- `binary_nominal`
- `unpruned`
- `no_subtree_raising`
- `no_collapse`
- `laplace`

The stochastic configuration currently in scope is:

- `rep`

Fixed seed list for `REP`:

- `1,2,3,4,5,6,7,8,9,10`
- `11,12,13,14,15,16,17,18,19,20`
- `21,22,23,24,25,26,27,28,29,30`
- `31,32,33,34,35,36,37,38,39,40`
- `41,42,43,44,45,46,47,48,49,50`

Paper seed set file:

- [j48_paper_seedset_50.txt](/home/javier/VisualStudio/IDEA/scripts/analysis/j48_paper_seedset_50.txt)

Operational note:

- the runners can now consume this shared seed set through `SEEDS_FILE`,
- which makes it practical to reuse the exact same seed policy across synthetic,
  tabular, and IDS experiments in the paper.

This policy is intentionally asymmetric:

- it avoids wasting computation on deterministic paths,
- and it concentrates the statistical effort on the only path where seed
  sensitivity is expected to matter.

## Proposed structure

The expanded `Layer A` is organized in five blocks.

### Block A1: Numeric split behavior

Goal:

- threshold selection,
- split-point relocation,
- MDL sensitivity,
- deterministic tie-breaking,
- `minNumObj` effects.

Cases:

1. `numeric_tie_first_feature`
2. `numeric_tie_first_threshold`
3. `numeric_split_actual_value`
4. `numeric_split_dense_duplicates`
5. `numeric_mdl_borderline`
6. `numeric_minnumobj_block`

Primary configs:

- `base`
- `no_split_actual`
- `no_mdl`

Seeds:

- `1`

### Block A2: Missing values

Goal:

- fractional propagation in train,
- routing in test,
- interaction with numeric and nominal splits,
- probability stability.

Cases:

7. `missing_fractional_train_test`
8. `missing_numeric_one_sided`
9. `missing_nominal_multiway`
10. `missing_nominal_binary`
11. `missing_high_mass_fractional`
12. `missing_with_pruning`

Primary configs:

- `base`
- `binary_nominal`
- `rep`

Seeds:

- `1`
- `1..50` only for `rep`

### Block A3: Nominal behavior

Goal:

- multi-way splits,
- binary grouping under `-B`,
- declared domains vs observed values,
- rare or empty branches,
- unseen values in test.

Cases:

13. `nominal_multiway_basic`
14. `nominal_unseen_test_value`
15. `nominal_binary_grouping`
16. `nominal_binary_tie`
17. `nominal_high_cardinality`
18. `nominal_empty_branch_probability`

Primary configs:

- `base`
- `binary_nominal`

Seeds:

- `1`

### Block A4: Pruning and REP

Goal:

- pessimistic pruning,
- subtree retention vs collapse,
- reduced-error pruning semantics,
- branches with no holdout support,
- leaf-vs-subtree decisions.

Cases:

19. `rep_single_prune`
20. `rep_holdout_empty_branch`
21. `rep_leaf_vs_subtree`
22. `unpruned_retains_small_split`
23. `no_subtree_raising_delta`
24. `collapse_vs_prune`

Primary configs:

- `base`
- `unpruned`
- `no_subtree_raising`
- `rep`

Seeds:

- `1` for deterministic configs
- `1..50` for `rep`

### Block A5: Probabilities and rare classes

Goal:

- leaf probability fidelity,
- Laplace effects,
- tiny classes,
- residual max-delta explanations.

Cases:

25. `laplace_rare_leaf`
26. `laplace_empty_nominal_branch`
27. `tiny_class_three_way`
28. `rare_label_under_rep`
29. `minnumobj_tiny_leaf`

Primary configs:

- `base`
- `laplace`
- `rep`

Seeds:

- `1`
- `1..50` for `rep`

## Coverage summary

Expanded case count:

- `29` synthetic cases

Recommended first selective execution matrix:

- `base`: `22`
- `no_split_actual`: `2`
- `no_mdl`: `2`
- `binary_nominal`: `7`
- `unpruned`: `1`
- `no_subtree_raising`: `1`
- `no_collapse`: `1`
- `laplace`: `5`
- `rep`: `400`

Total recommended first expanded run:

- `441` synthetic experiments

This matrix is intentionally selective, not exhaustive:

- deterministic configurations are only attached to cases where they are
  semantically informative;
- `REP` receives the bulk of the computational budget because that is where the
  seed-sensitive behavior actually lives.

## Acceptance interpretation

For the synthetic battery, results should be classified as:

- `Exact expected`
  - same predictions,
  - same or equivalent tree structure,
  - tiny probability deltas.
- `Near-exact acceptable`
  - same predictions,
  - small probability deltas,
  - small structural drift allowed when the case is intentionally unstable.
- `Diagnostic-only`
  - intentionally sensitive case used to reveal behavior, not necessarily to
    assert exact parity.

The initial batch already suggests that two families deserve special attention:

- `binary_nominal`
- `rep` on tiny datasets

Those should receive the highest priority in the expanded synthetic suite.

## Suggested execution order

1. Finish the numeric block.
2. Expand the nominal and `-B` block.
3. Expand `REP`-sensitive tiny cases.
4. Add the probability-specific cases.
5. Only then rerun the full expanded synthetic matrix.

## Statistical reporting recommendation

For the `REP` cases with `50` seeds, report:

- mean,
- median,
- standard deviation,
- interquartile range,
- min/max,
- `95%` confidence interval,
- exact-match frequency,
- and structural-divergence frequency.

This turns the synthetic battery into a true sensitivity analysis rather than a
single-seed regression suite.

## Paper role

This expanded battery supports the paper in three ways:

1. it shows that the local implementation was audited on controlled semantic
   mechanisms, not only on large benchmarks;
2. it documents which differences are general and which are narrow option- or
   dataset-specific residuals;
3. it strengthens the discussion section by linking remaining discrepancies to
   explicit controlled cases.
