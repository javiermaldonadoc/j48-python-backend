# J48 Extended Validation Battery

This document defines the next validation layer after the current `matrix200`.

The goal is not only to show that `EvLib.j48` is close to `WEKA J48` on the
current IDS-oriented benchmark, but to validate that similarity across:

- controlled synthetic cases,
- heterogeneous real tabular datasets,
- IDS wrapper scenarios,
- and a wider set of J48 option combinations.

It complements:

- [J48_ACCEPTANCE_CRITERIA.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_ACCEPTANCE_CRITERIA.md)
- [PHASE1_STATUS.md](/home/javier/VisualStudio/IDEA/EvLib/j48/PHASE1_STATUS.md)

## Current baseline

Latest full refresh:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_phase1_matrix200_20260311_172114/summary.csv)
- [comparison_vs_20260311_153847.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_phase1_matrix200_20260311_172114/comparison_vs_20260311_153847.csv)

Current reading:

- `base`, `unpruned`, `no_subtree_raising`, `no_collapse`, `laplace`, and `no_mdl` are exact on the current 5-dataset x 5-seed matrix.
- `REP` is exact on `UNSW-NB15`, `CIRA2020-AttNorm`, and `CIRA2020-DoHNDoH`.
- `REP` still has tiny residual drift on `CIC-IDS2017` and `NSL-KDD`.
- `no_split_actual` only keeps a tiny residual on `NSL-KDD`.

## Validation layers

### Layer A: Controlled differential cases

Purpose:

- isolate semantics,
- catch regressions fast,
- explain remaining residuals,
- and validate option-specific behavior before touching larger datasets.

Cases to cover:

1. Numeric threshold ties
2. Numeric threshold relocation to actual value
3. Missing values with fractional routing
4. Nominal multi-way splits
5. Binary nominal splits (`-B`)
6. Pessimistic pruning (`-C`, `-O`, `-S`)
7. Reduced-error pruning (`-R`, `-N`)
8. Empty nominal branches and unseen categorical values
9. Low-support branches with no holdout support under `REP`
10. Laplace probability output (`-A`)

Execution shape:

- `20` to `30` handcrafted datasets
- `1` seed for deterministic cases
- `5` seeds only for fold-sensitive `REP` cases

Primary criterion:

- exact prediction match,
- exact or near-exact probability output,
- exact tree structure when the case is intentionally tiny.

### Layer B: General real tabular benchmark

Purpose:

- test external validity outside IDS,
- cover mixed numeric/categorical/missing-value regimes,
- and support the claim of practical substitution for `WEKA J48`.

Recommended dataset families:

1. Predominantly categorical
   - Adult
     Source: UCI Adult dataset
     https://archive.ics.uci.edu/ml/datasets/Adult
   - Car Evaluation
     Source: UCI Car Evaluation dataset
     https://archive.ics.uci.edu/dataset/19/car
   - Mushroom
     Source: UCI Mushroom dataset
     https://archive.ics.uci.edu/dataset/73/mushroom
   - Nursery
     Source: UCI Nursery dataset
     https://archive.ics.uci.edu/ml/datasets/nursery

2. Mixed / missing values
   - Credit Approval
     Source: UCI Credit Approval dataset
     https://archive.ics.uci.edu/dataset/27
   - Breast Cancer Wisconsin (Original)
     Source: UCI Breast Cancer Wisconsin (Original)
     https://archive.ics.uci.edu/dataset/15

3. Predominantly numeric
   - Ionosphere
     Source: UCI Ionosphere dataset
     https://archive.ics.uci.edu/ml/datasets/Ionosphere
   - Breast Cancer Wisconsin (Diagnostic)
     Source: UCI Breast Cancer Wisconsin (Diagnostic)
     https://archive.ics.uci.edu/dataset/17/breast%2Bcancer

Recommended benchmark size:

- `8` real datasets
- `5` seeds
- default + focused option grid

Primary criterion:

- competitive substitution:
  prediction metrics equivalent or near-equivalent to `WEKA J48`
- strict audit:
  high prediction agreement and low probability delta

### Layer C: Applied IDS benchmark

Purpose:

- preserve the problem setting that motivates the project,
- measure effect inside the wrapper pipeline,
- and quantify practical runtime benefits later.

Recommended datasets:

- `NSL-KDD`
- `CIC-IDS2017`
- `UNSW-NB15`
- `CIRA2020-AttNorm`
- `CIRA2020-DoHNDoH`

Recommended experiments:

1. Plain classifier comparison
2. Wrapper evaluation with the EA
3. Stability of selected feature subsets
4. End-to-end runtime

Primary criterion:

- maintain or improve wrapper utility,
- keep classifier behavior equivalent enough to WEKA,
- then compare runtime.

## Option grid

The extended battery should not brute-force every combination.

Recommended core grid:

- `base`
- `unpruned`
- `no_subtree_raising`
- `no_collapse`
- `laplace`
- `rep`
- `no_mdl`
- `no_split_actual`
- `binary_splits`

Recommended interaction grid:

- `rep + laplace`
- `rep + no_subtree_raising`
- `binary_splits + missing`
- `no_mdl + no_split_actual`

## Metrics

Per run:

- prediction agreement with WEKA
- mismatch count
- probability mean absolute delta
- probability max absolute delta
- local node count
- local leaf count
- local max depth
- WEKA node count
- fit time
- inference time

Per dataset/config aggregate:

- mean and median agreement
- mean probability delta
- mean node gap
- mean runtime ratio

Against true labels:

- balanced accuracy
- macro-F1
- log loss

## Statistical protocol

Use the same logic defined in:

- [J48_ACCEPTANCE_CRITERIA.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_ACCEPTANCE_CRITERIA.md)

Recommended statistical tests:

- paired bootstrap for metric deltas
- McNemar for disagreement direction
- Holm correction for multiple pairwise claims

Suggested execution order:

1. Layer A full
2. Layer B default-only
3. Layer B focused option grid
4. Layer C plain classifier
5. Layer C wrapper runs

## Suggested workload

Pragmatic next campaign:

- Layer A:
  `~30` handcrafted cases
- Layer B:
  `8 datasets x 5 seeds x 9 configs = 360 runs`
- Layer C plain classifier:
  `5 datasets x 5 seeds x 9 configs = 225 runs`
- Layer C wrapper:
  selected configs only, likely `base`, `rep`, `unpruned`

That is large enough to support a stronger paper claim without becoming
unmanageable.

## Recommended next action

Before moving to performance work:

1. finish the tiny residual threshold drift in `REP` on `CIC-IDS2017` and `NSL-KDD`;
2. freeze `strict`;
3. execute Layer A and Layer B default-only;
4. only then open the `fast` line.
