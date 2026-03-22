# J48 WEKA Closure Backlog

This document is the active engineering backlog for closing the behavior gap
between the local `EvLib.j48` implementation and `WEKA J48`.

It is intentionally practical. Each item states:

- what still differs,
- where the code lives,
- why that gap likely matters,
- how to validate the change,
- and what "done" means.

## Goal

Primary target:

- matched configuration,
- behavior as close as possible to `WEKA J48`,
- same or equivalent predictions,
- similar tree structure when feasible,
- then better runtime as a second step.

Reference baseline:

- WEKA J48 public options and defaults
- `C45PruneableClassifierTree`
- `BinC45Split`
- `Distribution`

Freeze note:

- [J48_STRICT_FREEZE.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_STRICT_FREEZE.md)
- [J48_SYNTHETIC_EXPANDED_RESULTS.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_SYNTHETIC_EXPANDED_RESULTS.md)

## Current baseline

The current differential matrix is:

- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_phase1_matrix200_20260311_153847/summary.csv)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_rep_rerun_20260311_161614/summary.csv) (`REP` refresh after aligning `PruneableClassifierTree` fold construction)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_phase1_nondefault_refresh_20260311_170104/summary.csv) (`unpruned`, `no_subtree_raising`, and `REP` refresh after the recent pruning fixes)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_rep_refresh_20260311_170955/summary.csv) (`REP` refresh after pruning unsupported holdout branches)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_112530/summary.csv) (`Layer A` synthetic expanded campaign)
- [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_synthetic_expanded/j48_synth_expanded_20260312_114212/summary.csv) (`A3` + `A5` focused rerun after the `binary_nominal` routing fix)

Representative rows from that matrix:

- `CIC-IDS2017 base`: match `1.000`, local nodes `87`, WEKA nodes `87`
- `NSL-KDD base`: match `1.000`, local nodes `196`, WEKA nodes `196`
- `UNSW-NB15 base`: match `1.000`, local nodes `886`, WEKA nodes `886`
- `CIC-IDS2017 unpruned` (focused rerun): match `1.000`, local nodes `87`, WEKA nodes `87`
- `NSL-KDD unpruned` (focused rerun): match `1.000`, local nodes `266`, WEKA nodes `266`
- `UNSW-NB15 unpruned` (focused rerun): match `1.000`, local nodes `1077`, WEKA nodes `1077`
- `UNSW-NB15 no_subtree_raising` (focused rerun): match `1.000`, local nodes `1037`, WEKA nodes `1037`
- `UNSW-NB15 no_subtree_raising` (5-seed refresh): match `1.000`, local nodes `1037`, WEKA nodes `1037`
- `CIC-IDS2017 unpruned` (5-seed refresh): match `1.000`, local nodes `87`, WEKA nodes `87`
- `CIC-IDS2017 rep`: match `0.997`, local nodes `67`, WEKA nodes `65`
- `NSL-KDD rep`: match `0.999`, local nodes `240`, WEKA nodes `240`
- `UNSW-NB15 rep`: match `1.000`, local nodes `473`, WEKA nodes `473`
- synthetic expanded campaign: `441/441 OK`, with `438/441` exact in the full sweep and `69/69` exact in the focused rerun of the previously non-exact `A3` and `A5` blocks

Important note:

- the `base` configuration is now aligned across all five datasets and five seeds;
- the focused `unpruned` reruns are now aligned on the three hardest datasets (`NSL-KDD`, `CIC-IDS2017`, `UNSW-NB15`) for seeds `1`, `7`, and `13`;
- the focused `no_subtree_raising` reruns are also aligned on `NSL-KDD`, `CIC-IDS2017`, and `UNSW-NB15` after fixing pessimistic error estimation for singleton leaves;
- the 5-seed non-default refresh now aligns `unpruned` and `no_subtree_raising` across all five datasets;
- `REP` is now exact on `UNSW-NB15` and close on the remaining datasets after aligning fold construction and pruning branches with no holdout support;
- `J48-009` already removed the largest probability-fidelity bug on empty nominal branches;
- the remaining gaps are concentrated in tiny `REP` threshold differences on `CIC-IDS2017` and `NSL-KDD`, plus residual probability fidelity on the harder option combinations;
- the next debugging effort should target those blocks, not `base`.

## Priority legend

- `P0`: blocks strict comparison or is clearly semantically wrong
- `P1`: major source of remaining structural/predictive gap
- `P2`: important edge-case or option-combination coverage
- `P3`: cleanup, instrumentation, or support work

## Backlog

| ID | Priority | Element | Files | Current gap | Hypothesis | Validation | Done when |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `J48-001` | `P0` | Refresh stale `no_collapse` data | [run_j48_phase1_matrix200.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_phase1_matrix200.sh), [summary_reconciled.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_phase1_matrix200_20260310_135930/rerun_20260310_141433/summary_reconciled.csv) | The matrix still contains pre-fix `no_collapse` rows. | The latest pruning fix materially changes that configuration and invalidates old measurements. | Rerun only `no_collapse` for the full matrix and regenerate a reconciled summary. | The matrix has no stale `no_collapse` rows. |
| `J48-002` | `P1` | Numeric split scoring / MDL exactness | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L618) | MDL is still explicitly documented as approximate. | Numeric scoring is a main source of structural divergence on `NSL-KDD` and `UNSW-NB15`. | Add microtests with handcrafted numeric datasets and rerun `base` / `no_mdl`. | `base` and `no_mdl` move closer to WEKA on tree size and match, especially on `NSL-KDD` and `UNSW-NB15`. |
| `J48-003` | `P1` | Confidence-based pruning semantics | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1308), [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1808) | Pruning exists, but we do not yet claim exact J48 equivalence. | Small differences in pessimistic error estimation can cascade into very different tree shapes. | Add focused tests for `-C`, `-O`, `-U` and compare local vs WEKA tree sizes on small crafted datasets. | `base`, `unpruned`, and `no_collapse` show consistent ordering and reduced tree-size gap. |
| `J48-004` | `P1` | Subtree raising exactness | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1764) | The code still describes subtree raising as an approximation. | Remaining structural gaps are likely driven by subtree raising choices after split selection. | Compare `base` vs `no_subtree_raising` before/after changes; inspect node-count deltas and handcrafted trees. | `base` gets closer to WEKA without worsening `no_subtree_raising`. |
| `J48-005` | `P1` | Global tie-breaking discipline | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L729), [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1098) | Tie-breaking improved, but it is not yet proven to match WEKA in all split competitions. | Stable deterministic ordering can reduce dataset-dependent drift when gains are very close. | Add synthetic tie cases for numeric and nominal features; compare against WEKA tree text and chosen feature/threshold. | Handcrafted tie cases select the same split as WEKA. |
| `J48-006` | `P3` | Binary nominal split validation (`-B`) | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L854), [sklearn_api.py](/home/javier/VisualStudio/IDEA/EvLib/j48/sklearn_api.py#L30) | `-B` was the last controlled residual family in the first synthetic expanded sweep. A focused rerun closes the `nominal_unseen_test_value`, `nominal_binary_tie`, and `laplace_empty_nominal_branch` cases after fixing routing for the `__WEKA_OTHER__` branch. | Binary grouping under one-vs-rest was correct in training semantics but still routed prediction-time values incorrectly when the grouped branch had to absorb non-selected categories. | Keep the fixed `A3` and `A5` synthetic cases in `Layer A` and confirm the closure in the next full synthetic rerun. | The next full `Layer A` rerun remains exact on the `-B` cases. |
| `J48-007` | `P2` | Missing values under nominal + pruning | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1360), [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1815) | Missing routing exists, but mixed nominal/pruning cases remain under-tested. | Missing-value weighting may still distort pruning or child probabilities in complex trees. | Add targeted tests with nominal + missing + `base` / `-B` / `REP`. | Probability deltas shrink and no branch-routing anomalies remain in microtests. |
| `J48-008` | `P2` | Reduced-error pruning parity | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1910) | The main fold-construction gap is closed, but small residual structural differences remain in some `REP` trees. | `PruneableClassifierTree` semantics are now much closer; the remaining gap likely lives in the last pruning choices rather than in the holdout split itself. | Keep a focused `REP` rerun, add handcrafted fold-sensitive datasets, and inspect the remaining first divergences on `UNSW-NB15` and `NSL-KDD`. | `REP` stays above `0.99` on the focused rerun and the remaining structural diffs are limited to low-impact threshold or leaf-vs-split cases. |
| `J48-009` | `P2` | Probability fidelity | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L2217), [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_rep_rerun_probfix_20260311_163516/summary.csv) | The main empty-branch bug is fixed, but some high `probability_max_abs_delta` cases remain where local and WEKA still predict different labels. | Residual probability gaps are now dominated by the last structural mismatches rather than by one-hot fallback on empty branches. | Keep tracking `probability_mean_abs_delta` and inspect `probability_max_abs_delta` only on runs that still have prediction mismatches. | Mean probability deltas stay low and the remaining max deltas are explained by the few residual structural mismatches. |
| `J48-010` | `P2` | Structural differential analysis tooling | [run_j48_strict_differential_harness.py](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_strict_differential_harness.py) | We compare tree size and conditions, but not a deeper node-by-node diff. | Better structural diagnostics will localize the first divergence faster than aggregate metrics alone. | Extend exports with per-node rank/order metadata and first-divergence reporting. | Debugging a mismatch no longer requires manual inspection of full JSON trees. |
| `J48-011` | `P3` | Refresh Phase 1 matrix after semantic fixes | [run_j48_phase1_matrix200.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_phase1_matrix200.sh) | The big matrix predates `-B`, tie-breaking changes, and the `no_collapse` fix. | The benchmark picture should improve once the new semantics are exercised. | Rerun a reduced matrix first, then the full matrix if trends are good. | We have a fresh baseline that reflects the current code. |
| `J48-012` | `P3` | Speed baseline after strict parity work | [summary_reconciled.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_phase1/j48_phase1_matrix200_20260310_135930/rerun_20260310_141433/summary_reconciled.csv#L43) | The local implementation is still slower than WEKA in fit time. | Strict work should stop once semantics are stable enough; speed belongs to the next layer. | Freeze `strict`, then benchmark and profile hot paths. | We have a stable `strict` baseline to optimize without moving semantics. |

## Recommended execution order

1. synthetic edge-case validation on the frozen `strict` line
2. external tabular validation once `layer_b` datasets are available
3. `J48-012` performance work
4. only revisit residual `REP` / `no_split_actual` gaps if synthetic evidence points to a general semantic issue

## Coverage snapshot

Closed enough for the current strict line:

- `J48-001` refresh `no_collapse`
- `J48-002` numeric scoring / MDL
- `J48-003` confidence pruning
- `J48-004` subtree raising
- `J48-005` tie-breaking
- `J48-006` `-B` implementation and broad validation
- `J48-010` structural differential tooling
- `J48-011` fresh full matrix
- focused `unpruned` alignment on `NSL-KDD`, `CIC-IDS2017`, and `UNSW-NB15`
- focused `no_subtree_raising` alignment on `NSL-KDD`, `CIC-IDS2017`, and `UNSW-NB15`
- refreshed 5-seed non-default matrix for `unpruned`, `no_subtree_raising`, and `REP`

Still open or partial:

- `J48-007` missing + nominal + pruning edge cases
- `J48-008` reduced-error pruning residual cleanup
- `J48-009` residual probability fidelity in the harder configs
- `J48-012` speed work after semantics freeze

Practical interpretation:

- `J48-007` now moves from "fix before freeze" to "validate with synthetic edge cases";
- `J48-006` is no longer a broad implementation gap and its last controlled residuals are closed on the focused synthetic rerun;
- `J48-008` and `J48-009` are currently documented residuals, not blockers for freezing the local `strict` baseline;
- `J48-012` is the main engineering step after synthetic validation is in place.

## Minimal validation protocol for each semantic change

Every non-trivial change should trigger:

1. `py_compile`
2. `python -m unittest tests.test_c45_j48_strict -q`
3. one handcrafted differential run against WEKA
4. one real dataset differential run on:
   - `CIC-IDS2017` if the change is expected to be "easy"
   - `NSL-KDD` if the change touches pruning/structure
   - `UNSW-NB15` if the change targets hard numeric cases

## Definition of "strict enough"

The strict line is good enough to hand off to performance work when:

- all exposed J48 options in scope are implemented and validated,
- no known semantic bug remains in option handling,
- `base` stays aligned on the current matrix,
- `unpruned` stays aligned on the focused rerun and is refreshed into the next matrix pass,
- `no_subtree_raising` stays aligned on the focused rerun and is refreshed into the next matrix pass,
- `REP` stays in the low-delta regime reached by the focused rerun,
- the remaining difficult configs are reduced to residual `REP`, residual probability deltas, or documented edge cases,
- probability deltas are acceptable for the competitive-equivalent claim,
- remaining differences are documented as deliberate or low impact.
