# J48 Synthetic Expanded Results

This note records the first completed `Layer A` synthetic campaign executed
after freezing the current `strict` baseline.

Primary full run:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_112530/summary.csv)

Focused follow-up run after the `binary_nominal` routing fix:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_114212/summary.csv)

Final full rerun after the `binary_nominal` routing fix:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_114644/summary.csv)

## Scope

- `29` synthetic edge cases
- `441` paired runs against `WEKA J48`
- deterministic configurations evaluated with `1` fixed seed
- `REP` evaluated with the shared paper seed set of `50` fixed six-digit seeds

## Aggregate outcome

The evidence is best read in three passes:

1. the first full campaign established broad exactness and isolated the only
   remaining controlled residuals;
2. a focused rerun of the affected blocks (`A3` and `A5`) confirmed that the
   `binary_nominal` routing fix closes those residuals without reopening the
   previously exact blocks;
3. a final full rerun confirmed that the complete `Layer A` campaign is exact
   after the fix.

### Pass 1: full 441-run campaign

- `441/441 OK`
- `438/441` runs achieved exact prediction agreement with WEKA
- the only non-exact rows belong to `binary_nominal`

### By block

| Block | Purpose | Runs | Exact-rate | Mean match | Total mismatches | Mean prob. delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `A1` | Numeric split behavior | 10 | `1.000` | `1.000000` | 0 | `0.000000` |
| `A2` | Missing values | 206 | `1.000` | `1.000000` | 0 | `0.000000` |
| `A3` | Nominal behavior | 10 | `0.800` | `0.900000` | 3 | `0.077778` |
| `A4` | Pruning and `REP` | 156 | `1.000` | `1.000000` | 0 | `0.000000` |
| `A5` | Probabilities and rare classes | 59 | `0.983` | `0.988701` | 2 | `0.005650` |

### By configuration

| Config | Runs | Exact-rate | Mean match | Total mismatches | Mean prob. delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 22 | `1.000` | `1.000000` | 0 | `0.000000` |
| `rep` | 400 | `1.000` | `1.000000` | 0 | `0.000000` |
| `laplace` | 5 | `1.000` | `1.000000` | 0 | `0.000000` |
| `no_mdl` | 2 | `1.000` | `1.000000` | 0 | `0.000000` |
| `no_split_actual` | 2 | `1.000` | `1.000000` | 0 | `0.000000` |
| `unpruned` | 1 | `1.000` | `1.000000` | 0 | `0.000000` |
| `no_subtree_raising` | 1 | `1.000` | `1.000000` | 0 | `0.000000` |
| `no_collapse` | 1 | `1.000` | `1.000000` | 0 | `0.000000` |
| `binary_nominal` | 7 | `0.571` | `0.761905` | 5 | `0.158730` |

## REP result

`REP` is exact in all controlled cases in this campaign:

- `400/400` exact paired runs
- `8` case families
- `50` fixed seeds per family

Case families covered:

- `missing_fractional_train_test`
- `missing_high_mass_fractional`
- `missing_numeric_one_sided`
- `missing_with_pruning`
- `rare_label_under_rep`
- `rep_holdout_empty_branch`
- `rep_leaf_vs_subtree`
- `rep_single_prune`

This result is important because it shows that the remaining stochastic path in
scope is stable under the fixed paper seed set and does not introduce residual
behavioral drift in the controlled suite.

## Residual cases isolated in the full campaign

Only three runs remain non-exact, and all of them belong to `-B` edge cases.

| Block | Case | Config | Match | Mismatches | Mean prob. delta | Local nodes | WEKA nodes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `A3` | `nominal_unseen_test_value` | `binary_nominal` | `0.666667` | 1 | `0.222222` | 5 | 5 |
| `A3` | `nominal_binary_tie` | `binary_nominal` | `0.333333` | 2 | `0.111111` | 3 | 3 |
| `A5` | `laplace_empty_nominal_branch` | `binary_nominal` | `0.333333` | 2 | `0.333333` | 3 | 3 |

Shared pattern:

- the first structural divergence is at the root,
- the divergence is limited to `split_type_mismatch` under `-B`,
- and no residual controlled case outside `binary_nominal` remains open.

### Pass 2: focused rerun of the previously non-exact blocks

The follow-up run covers:

- `A3` nominal behavior
- `A5` probabilities and rare classes
- `69` paired runs in total

Outcome:

- `69/69 OK`
- `69/69` exact prediction agreement
- `binary_nominal` exact in `6/6` focused runs
- no remaining controlled residuals in the rerun scope

### Pass 3: final full rerun after the fix

Outcome:

- `441/441 OK`
- `441/441` exact prediction agreement
- all five blocks (`A1` to `A5`) exact
- all configurations exact, including `binary_nominal`
- `400/400` exact `REP` runs preserved under the same fixed paper seed set

Relative to the first full campaign, only five rows change, and all changes are
consistent with the intended fix:

- `nominal_unseen_test_value + binary_nominal`
- `nominal_binary_tie + binary_nominal`
- `laplace_empty_nominal_branch + binary_nominal`
- two already exact `binary_nominal` rows keep exact prediction agreement while
  updating internal probability routing behavior

## Interpretation

Taken together, the expanded synthetic campaign supports the following reading
of the frozen `strict` line:

- numeric behavior is exact in the controlled suite,
- missing-value handling is exact in the controlled suite,
- pessimistic pruning and `REP` are exact in the controlled suite,
- probability behavior is exact in the controlled suite,
- the initial `-B` residuals were narrow, explicitly diagnosed, closed in the
  focused rerun, and confirmed closed in the final full rerun.

This is strong enough to support a paper discussion that distinguishes:

- a broad near-faithful `strict` baseline for the J48 behaviors covered by
  `Layer A`,
- and a nominal binary grouping block that was sensitive enough to require a
  focused follow-up audit before the controlled suite could be fully closed.
