# J48 Acceptance Criteria

This document defines the acceptance criteria for the `j48.strict` line.

Its role is practical:

- decide when a semantic gap is considered closed,
- decide when the strict line is "good enough" to freeze,
- and separate `behavioral equivalence` from `structural identity`.

## Core principle

The primary acceptance criterion is **behavioral equivalence under matched settings**.

That means:

- same or equivalent predictions,
- very similar probabilities,
- equivalent predictive metrics,
- stable behavior across seeds and option combinations.

Exact tree identity is **not** the primary acceptance criterion.

Tree structure is used mainly as:

- a debugging tool,
- a localization tool for semantic divergence,
- and a secondary quality indicator.

## Acceptance levels

### Level 0: API / Option Coverage

An option is considered covered when:

- it is exposed in the local API,
- it is passed correctly through the harness,
- it is not known to be semantically broken,
- and it has at least one focused regression test.

This level is necessary, but not sufficient.

### Level 1: Handcrafted Semantic Validity

This level covers synthetic and handcrafted cases.

Acceptance:

- all deterministic unit tests pass,
- all handcrafted WEKA differential cases pass,
- split choice matches WEKA on the intended microcase,
- and no known semantic bug remains in the touched area.

This level is binary:

- pass / fail,
- no statistical analysis is needed here.

### Level 2: Behavioral Equivalence

This is the main acceptance level for `j48.strict`.

Acceptance is based on:

- prediction agreement,
- probability agreement,
- predictive metric deltas,
- and seed stability.

Structural equality is not required if behavioral equivalence is demonstrated.

### Level 3: Structural Consistency

This is a secondary acceptance level.

Acceptance means:

- no obviously wrong early divergence,
- first divergence occurs deeper in the tree,
- and tree size gaps are reduced on the hard datasets.

This level supports debugging and confidence, but does not override Level 2.

## Item-level acceptance

Each `J48-xxx` item is considered closed only if all of the following hold:

1. `py_compile` passes.
2. `python -m unittest tests.test_c45_j48_strict -q` passes.
3. At least one handcrafted differential case confirms the intended behavior.
4. At least one real differential run shows improvement or no regression on the targeted metric.
5. No regression appears on the previously good datasets.

For semantic changes that affect tree construction:

- rerun `CIC-IDS2017 base`,
- rerun `NSL-KDD base`,
- rerun `UNSW-NB15 base`.

For semantic changes that affect `REP`:

- rerun the full `REP` slice used for validation.

## Release-level acceptance for `j48.strict`

The strict line is ready to freeze when all conditions below hold.

### A. Scope coverage

- all `P0` and `P1` backlog items are closed,
- remaining `P2` items are either closed or documented as low impact,
- and no known semantic bug remains in option handling.

### B. Block A: controlled validity

- `100%` of handcrafted semantic cases pass,
- and all focused WEKA differential microcases pass.

### C. Block B: general real-tabular equivalence

Experimental unit:

- one `dataset x config x seed` triplet.

Recommended default:

- `5` seeds,
- matched configurations,
- paired comparison against WEKA.

Primary acceptance metrics:

- `prediction_match_fraction`
- `probability_mean_abs_delta`
- `balanced_accuracy`
- `macro_f1`
- `log_loss`

Primary acceptance thresholds:

- median `prediction_match_fraction >= 0.99`
- no Block B dataset below `0.97`
- median `probability_mean_abs_delta <= 0.01`
- no Block B dataset above `0.03`

Metric equivalence thresholds:

- `|delta balanced_accuracy| <= 0.005`
- `|delta macro_f1| <= 0.01`
- `|delta log_loss| <= 0.02`

### D. Block C: IDS / wrapper-oriented equivalence

Primary acceptance thresholds:

- median `prediction_match_fraction >= 0.98`
- no Block C dataset below `0.95`
- median `probability_mean_abs_delta <= 0.02`
- no Block C dataset above `0.05`

Metric equivalence thresholds:

- `|delta balanced_accuracy| <= 0.01`
- `|delta macro_f1| <= 0.015`
- `|delta log_loss| <= 0.03`

These bounds are slightly wider than Block B because IDS datasets are harder and larger.

### E. Structural sanity

Structural sanity is accepted when:

- the first divergence is not consistently at the root or immediately below it,
- tree-size gaps shrink on the difficult datasets,
- and repeated seeds show the same divergence pattern when the model is deterministic.

This is a diagnostic gate, not the main equivalence gate.

## Statistical validation protocol

The validation should be based on **equivalence**, not on "absence of significant difference".

In particular:

- "not significant" is not enough,
- and should not be used as the main acceptance argument.

### Per-dataset paired validation

For each `dataset x config x seed` result:

- compute `balanced_accuracy`, `macro_f1`, `log_loss`,
- compute local minus WEKA deltas,
- and keep instance-level predictions for paired analysis.

For each `dataset x config` pair:

1. Use paired bootstrap on test instances to obtain a `95% CI` for:
   - `delta balanced_accuracy`
   - `delta macro_f1`
   - `delta log_loss`
2. Use McNemar on the paired predictions to detect directional disagreement.

Acceptance:

- the `95% CI` must lie entirely inside the predefined equivalence margin,
- and McNemar should not show a strong directional shift after correction.

McNemar is secondary here.
The main argument is the equivalence CI.

### Across-dataset validation

For each configuration:

- use the dataset-level deltas as paired observations,
- and run equivalence testing across datasets.

Recommended:

- `TOST` for paired deltas if assumptions are acceptable,
- otherwise paired bootstrap CI over dataset-level deltas,
- and `Wilcoxon signed-rank` as a secondary nonparametric check.

### Multiple-comparison correction

When testing multiple configurations or multiple metrics:

- correct p-values with `Holm-Bonferroni`.

This applies especially to:

- McNemar,
- Wilcoxon,
- or any repeated per-config inference.

## What is not required

The following are **not** required to accept the strict line:

- exact node-by-node tree identity,
- identical leaf counts on all datasets,
- identical runtime,
- or identical tie resolution in cases that do not affect behavior.

They become important only when they materially affect:

- predictions,
- probabilities,
- or stability.

## Practical decision rule

Use the following order:

1. Does the change preserve or improve behavioral equivalence?
2. Does it reduce probability delta?
3. Does it reduce early structural divergence?
4. Does it avoid regressions on already-good datasets?
5. Only then ask whether the exact tree matches WEKA more closely.

## Freeze criterion before performance work

The strict line can move to `j48.fast` work when:

- all `P0/P1` gaps are closed,
- `REP` is no longer clearly broken,
- Block A passes completely,
- Block B meets the equivalence criteria,
- Block C is within the looser IDS bounds,
- and remaining differences are documented.
