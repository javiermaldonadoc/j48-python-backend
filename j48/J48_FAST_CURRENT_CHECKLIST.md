# J48 Fast Current Checklist

This note captures the current engineering checkpoint for `PYJ48_FAST` after
the latest accepted core optimizations.

It is meant to answer two practical questions before launching new campaigns:

1. what is already done and trusted,
2. what still needs evidence refresh before we treat the current code as the
   new validated baseline against `strict` and `WEKA`.

## Current checkpoint

Recent accepted commits on top of the previously validated `j48.fast` line:

- `e54f25c` `Avoid object coercion in J48 fast numeric path`
- `9a6104b` `Optimize nominal handling in J48 fast core`
- `a170e50` `Preserve J48 fast engine state across prepared fits`

These changes were accepted because they produced a strong local runtime gain
without intentionally changing tree semantics.

Representative local EA-style benchmark on `NSL-KDD` (`standard`, `acc/acc`,
`50` generations, `8` cores, fixed seed):

| Version | `run_elapsed_sec` | `avg_model_fit_sec` | Comment |
| --- | ---: | ---: | --- |
| pre-opt baseline | `59.05` | `0.1754` | original checkpoint |
| after `e54f25c` | `31.63` | `0.0935` | numeric fast-path cleanup |
| after `9a6104b` | `12.94` | `0.0372` | nominal-path cleanup |
| after `a170e50` | `11.89` | `0.0335` | wrapper-state fix preserving optimized behavior |

This is a large enough improvement to justify a fresh validation pass before
the next large campaign.

## What is already closed

These points are considered solid at the current checkpoint:

- [x] `strict` semantic baseline frozen
- [x] synthetic expanded suite exact against `WEKA`
- [x] `fast-vs-strict Layer A` exact on the last accepted validation baseline
- [x] `fast-vs-strict Layer B/C` exact on the last accepted validation baseline
- [x] `fast-vs-WEKA Layer B/C` near-faithful on the last accepted validation baseline
- [x] `FAST-012 / attempt C` accepted as the current numeric training baseline
- [x] `FAST-013 / attempt A` accepted as the current pruning-cache baseline
- [x] `FAST-013 / attempt B` accepted as a conservative pruning/raising cleanup
- [x] `FAST-011 / attempt C` accepted as the current metadata-aware inference-preparation baseline
- [x] `FAST-018 / attempt D` accepted as the current object-heavy fit-preparation baseline
- [x] `FAST-016 / attempt C` repeated-evaluation cache available for EA evaluation
- [x] `FAST-016 / attempt B` identified `p3off_leaf1` as the best general EA preset candidate
- [x] `FAST-016 / attempt C` accepted as the current repeated-evaluation cache baseline
- [x] `FAST-017` code audit completed and still useful as a hotspot map
- [x] `FAST-019` now has dedicated node-size/depth audit tooling

## What changed after the last formal evidence refresh

These changes are already in code and have now been refreshed at the
validation-artifact level:

- [x] avoid global promotion to `dtype=object` in the fast numeric fit path
- [x] keep nominal arrays numeric where possible during domain collection,
  split scoring, routing, `predict`, and `predict_proba`
- [x] verify locally that the safe optimization path reproduces the same
  benchmark trajectory after discarding the unsafe variant

## Required evidence refresh before the next paper-facing or campaign-facing baseline

These were the minimum checks required to validate the new checkpoint:

- [x] unit tests
  - command: `python -m unittest tests.test_c45_j48_strict -q`
- [x] `fast-vs-strict Layer A`
- [x] `fast-vs-strict Layer B/C`
- [x] `fast-vs-WEKA Layer B/C`
- [x] serial timing rerun
- [x] artifact regeneration
  - script:
    [generate_j48_fast_validation_artifacts.py](/home/javier/VisualStudio/IDEA/scripts/analysis/generate_j48_fast_validation_artifacts.py)

All six checks stayed clean, so the current optimized code can now be treated
as the new validated `j48.fast` checkpoint.

Headline refreshed evidence:

- `fast-vs-strict Layer A`: `441/441` exact
- `fast-vs-strict Layer B/C`: `741/741` exact
- `fast-vs-WEKA Layer B/C`: mean match `0.999428`, minimum `0.956522`
- timing medians: `fit 4.303x`, `predict 1.158x`, `predict_only 0.627x`,
  `predict_proba 2.921x`

## Optional but recommended evidence

These are not blockers for a short validation battery, but they are useful
before a very large EA campaign:

- [ ] repeat the local EA-style mirror benchmark on:
  - `NSL-KDD`
  - `UNSW-NB15`
  - `CIC-IDS2017`
  - `CIRA2020-AttNorm`
  - `CIRA2020-DoHNDoH`
- [ ] compare `p3off_leaf1` vs the strict J48-oriented default after the new
  core optimizations
- [ ] decide whether `subtree_raising` remains on in the EA default profile
  or becomes an explicit high-fidelity option

## Current bottleneck view

Based on the current code audit, the refreshed validation campaign, and the
latest rejected micro-optimization attempt:

1. `subtree raising` and pruning remain a real residual
2. structural numeric split search is still important, but less urgent than
   before the recent accepted changes
3. nominal-heavy training work remains worth watching, but not every
   inference-side nominal optimization attempt pays off in practice

Practical reading:

- the project is no longer blocked on low-level object coercion problems,
- the next optimization wave should be deliberate,
- and the immediate priority is evidence refresh, not more speculative edits.

Most recent rejected attempt:

- [x] nominal-heavy inference micro-optimization in `engine.py` tested and
  reverted
  - preserved fidelity (`52/52` tests still clean)
  - did not produce a significant win in `predict` / `predict_only`
  - should not be revived unless a deeper hotspot study changes the diagnosis
- [x] `FAST-019 / attempt C` local-order partitioning tested and reverted
  - preserved fidelity on focused datasets
  - regressed `fit` on `Adult`, `NSL-KDD`, and `UNSW-NB15`
- [x] `FAST-019 / attempt D` top-of-tree-only order reuse tested and reverted
  - preserved fidelity on focused datasets
  - still failed to beat the accepted baseline
- [x] `FAST-007 / attempt A` minimal `Cython` numeric-kernel prototype tested
  and reverted
  - preserved fidelity on `Adult`, `NSL-KDD`, and `UNSW-NB15`
  - did not produce a meaningful `fit` win against the accepted baseline
  - should not be revived as a thin wrapper around the current numeric kernel
- [x] `FAST-014 / attempt B` metadata-aware dense nominal training path tested
  and reverted
  - preserved core unit-test fidelity (`52/52` tests still clean)
  - used only structure-based gates, not dataset-specific conditions
  - regressed metadata-aware `fit` on all focused nominal-heavy datasets
  - should not be revived without a materially different nominal-core design
- [x] `FAST-016 / fixture path` removed from the runtime baseline
  - global fixture-on/off audit showed only `14/30` exact result rows and
    `15/30` equal metric rows across representative datasets
  - direct `mainMP.py` comparisons showed the cache was transparent but the
    fixture path changed EA trajectories
  - should not be revived unless a future design proves full semantic
    equivalence first

## Readiness decision

Current readiness level:

- `code readiness`: **high**
- `validation readiness`: **high**
- `campaign readiness for very large runs`: **high, after host/runtime sizing**

## Immediate next step

Recommended next action:

1. treat this checkpoint as the frozen paper-facing milestone for
   `paper/springer_latex_c45_j48/`,
2. keep the newly accepted object-heavy fit-preparation path as the current
   single-run `fit` baseline for object-heavy datasets,
3. then move the next optimization wave to `FAST-016` repeated-evaluation
   throughput or another clearly evidenced core opportunity rather than more
   dense nominal-training heuristics.

## Refreshed opportunity read

Based on the refreshed code audit (`fast017_code_audit_refresh_20260320`) and
direct wall-clock decomposition on the focal datasets:

- `prepare_fit_bundle(...)` is still a meaningful single-run residual on
  object-heavy datasets:
  - `Adult`: about `24%` of total `fit`
  - `Mushroom`: about `93%` of total `fit`
  - `CreditApproval`: about `26%` of total `fit`
- the tree core still dominates the numeric IDS-style cases:
  - `NSL-KDD`: preparation about `11%` of total `fit`
  - `UNSW-NB15`: preparation about `3%` of total `fit`
- `prepare_predict_data(...)` still dominates inference wall time across all
  focal datasets, so any future inference work should target preparation, not
  compiled traversal

Practical reading:

1. `FAST-018` now has a slightly stronger object-heavy baseline after a
   pandas/object-heavy fit path cleanup that:
   - avoids materializing the whole DataFrame as an `object` ndarray on the
     accepted path,
   - bulk-extracts already-typed numeric columns when the frame is dominated
     by nominal metadata,
   - and preserves exact equality of prepared arrays against the previous
     baseline
2. `FAST-011` now has a stronger metadata-aware baseline after a lane split
   that:
   - keeps categorical predict-time columns on a dedicated path,
   - uses value-factorization for non-categorical nominal pandas inputs,
   - and preserves the accepted `strict`/`WEKA` fidelity checkpoints on the
     focused smoke manifest
3. `FAST-016` remains the highest-confidence wall-clock lever for repeated
   campaign loops
4. `FAST-019` is still open, but only through a lower-level kernel/materialization
   redesign, not another order-reuse experiment
5. `FAST-014` should now be treated as blocked pending a qualitatively
   different nominal-core idea, not a denser variant of the current branch
   statistics path
