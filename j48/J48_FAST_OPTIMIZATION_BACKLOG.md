# J48 Fast Optimization Backlog

This document is the active engineering backlog for the `j48.fast` line.

It starts from a fixed semantic baseline:

- [J48_STRICT_FREEZE.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_STRICT_FREEZE.md)
- [J48_FAST_VALIDATION_RESULTS.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_VALIDATION_RESULTS.md)

The goal is not only to make the implementation faster, but to do so without
breaking the exact preservation checks already achieved against `strict`.

## Current optimization baseline

Validated results at the current checkpoint:

- `fast-vs-strict Layer A`: `441/441` exact
- `fast-vs-strict Layer B/C`: `741/741` exact
- `fast-vs-WEKA Layer B/C`: mean match `0.999428`
- timing medians:
  - `fit 4.303x`
  - `predict 1.158x`
  - `predict_only 0.627x`
  - `predict_proba 2.921x`

Representative timing picture:

- strongest `fit` wins:
  - `Adult 11.913x`
  - `NSL-KDD 7.846x`
  - `Nursery 7.298x`
  - `CarEvaluation 6.414x`
  - `UNSW-NB15 6.143x`
- strongest end-to-end `predict` wins:
  - `UNSW-NB15 14.373x`
  - `NSL-KDD 9.061x`
  - `Adult 2.207x`
  - `Nursery 1.551x`
  - `CIC-IDS2017 1.409x`
- remaining weak spots:
  - `predict_only` is still below `1.0` in several nominal-heavy datasets
  - `predict` / `predict_only` remain weakest on `BreastCancerOriginal`,
    `Mushroom`, `CarEvaluation`, and `CreditApproval`
  - the latest nominal-heavy inference micro-optimization attempt preserved
    fidelity but did not move these weak spots enough to justify adoption

## Immediate planning recommendation

Given the refreshed paper-facing milestone and the latest rejected
micro-optimization attempt, the next optimization wave should be:

1. `FAST-013` second pass
   - pruning / subtree-raising cost reduction
   - best balance between likely runtime gain and semantic containment
2. `FAST-019`
   - structural numeric split-search redesign
   - higher upside, but also a larger implementation change
3. keep `FAST-015` / artifact refresh discipline active
   - every accepted change should rerun the current `weka_fast` refresh script

Deprioritized for now:

- additional nominal-inference micro-tweaks in `engine.py`
- nominal branch lookup rewrites unless a new hotspot study changes the signal

## Priority legend

- `P0`: correctness or validation blocker
- `P1`: highest-return performance work with low semantic risk
- `P2`: important structural optimization with moderate risk
- `P3`: exploratory or follow-up work

## Backlog

| ID | Priority | Focus | Files | Why it matters | Expected effect | Validation gate | Done when |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `FAST-001` | `P1` | Nominal split induction counts | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py) | Nominal training still builds repeated masks per category and class. This is likely the next largest general training hotspot after the numeric kernels. | Faster `fit`, especially on `Adult`, `Nursery`, `CarEvaluation`, `Mushroom`, `CreditApproval`. | Unit tests, `Layer A`, `Layer B base`, `Layer C base`, then timing rerun on `Adult`, `Nursery`, `CarEvaluation`, `CreditApproval`. | Nominal-heavy datasets improve in `fit` without losing `fast-vs-strict` exactness. |
| `FAST-002` | `P1` | Nominal branch lookup in inference | [engine.py](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py) | The compiled traversal still scans nominal edges linearly. This is a likely reason why `predict_only` remains mixed. | Faster `predict_only` and end-to-end `predict`, especially where nominal branching dominates. | Unit tests, `fast-vs-strict Layer A/B/C`, then timing rerun on `Adult`, `Nursery`, `CarEvaluation`, `Mushroom`. | `predict_only` median improves materially on the nominal-heavy datasets without semantic drift. |
| `FAST-003` | `P1` | Reusable encoded inputs for repeated evaluation | [engine.py](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py), [sklearn_api.py](/home/javier/VisualStudio/IDEA/EvLib/j48/sklearn_api.py) | Wrapper search and repeated benchmarks pay the encoding cost many times. Today encoding is still a visible portion of inference cost. | Lower repeated `fit`/`predict` overhead in evaluation loops and large campaign runs. | Targeted benchmark of repeated calls, plus `fast-vs-strict` smoke. | Repeated-evaluation scenarios show a measurable drop in preprocessing overhead without changing outputs. |
| `FAST-004` | `P2` | Compact nominal node representation | [engine.py](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py) | The current tree compilation is generic. A denser nominal representation could reduce branch dispatch cost and improve cache behavior. | Better `predict_only`, potentially better `predict_proba`. | `fast-vs-strict` full rerun, then timing on `Adult`, `Nursery`, `CarEvaluation`, `UNSW-NB15`. | Inference improves in median and no semantic regression appears. |
| `FAST-005` | `P2` | Numba kernels for nominal split scoring | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py) | Numeric kernels already paid off. Nominal split scoring remains comparatively Python-heavy. | Better `fit` on mixed and categorical datasets. | Unit tests, `Layer A`, `Layer B base`, `Layer C base`, timing rerun on nominal-heavy datasets. | The nominal path shows clear `fit` gains and keeps exact preservation. |
| `FAST-006` | `P2` | Profile-guided predict/proba separation | [engine.py](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py), [benchmark_j48_fast_baseline.py](/home/javier/VisualStudio/IDEA/scripts/analysis/benchmark_j48_fast_baseline.py) | `predict_only` and `predict_proba` do not move together. We need clearer profiling to avoid optimizing the wrong path. | Better prioritization and safer optimization choices. | Add targeted profiling outputs and rerun the timing suite. | We can attribute remaining inference cost to specific internal stages. |
| `FAST-007` | `P3` | Cython backend feasibility | [README.md](/home/javier/VisualStudio/IDEA/EvLib/j48/README.md), [J48_FAST_ROADMAP.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_ROADMAP.md) | If `numba` plateaus, the next step is a compiled backend for the confirmed hot paths. | Longer-term speed ceiling increase. | Prototype only after `FAST-001` to `FAST-005` settle. | A focused prototype shows whether `cython` is worth the integration cost. |
| `FAST-008` | `P3` | Performance results artifact refresh | [generate_j48_fast_validation_artifacts.py](/home/javier/VisualStudio/IDEA/scripts/analysis/generate_j48_fast_validation_artifacts.py), [J48_FAST_VALIDATION_RESULTS.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_VALIDATION_RESULTS.md) | Every meaningful optimization should update the technical evidence used by the paper. | Keeps paper evidence synchronized with the code. | Regenerate artifacts after each accepted optimization milestone. | The technical artifact and paper tables reflect the latest accepted checkpoint. |
| `FAST-009` | `P1` | DataFrame-first inference preparation | [engine.py](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py) | `FAST-006` showed that several weak spots are dominated by `prepare_predict_data`, especially when mixed-type pandas inputs are materialized too early as `object` arrays. | Lower `predict` and `predict_proba` overhead without touching tree semantics. | Unit tests, focused serial timing on `Adult`, `Nursery`, `CreditApproval`, `CarEvaluation`, `CIC-IDS2017`. | Preparation time drops materially on the focal datasets and no semantic regression appears. |
| `FAST-010` | `P1` | Missing-aware fast inference path | [engine.py](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py), [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py) | `FAST-006` showed that several weak inference cases are dominated by semantic-preserving fallback when `fractional_missing` meets non-finite inputs. | Lower fallback incidence and better `predict` / `predict_proba` on datasets with missing values. | Unit tests, `fast-vs-strict Layer A`, focal timing rerun on `Adult`, `UNSW-NB15`, `CarEvaluation`, `CreditApproval`. | Fallback incidence drops materially and timing improves with no semantic drift. |
| `FAST-011` | `P1` | Predict-input preparation cost reduction | [engine.py](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py) | After `FAST-009` and `FAST-010`, inference profiles still show `prepare_predict_data` dominating total time on several datasets even when traversal is already compiled and exact. | Lower `predict` / `predict_proba` wall time by reducing coercion, encoding, and schema-handling overhead before traversal. | Unit tests, focused serial timing on `Adult`, `CreditApproval`, `UNSW-NB15`, `CarEvaluation`, `CIC-IDS2017`, then `fast-vs-strict` smoke. | `prepare_time_sec` drops materially on the focal datasets and end-to-end inference improves without semantic drift. |
| `FAST-012` | `P1` | Numeric split-search cost reduction in training | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py) | The main residual `fit` hotspot still lives in numeric split search, especially repeated sorting and rescoring in `_find_best_numeric_split_candidate(...)`. | Better `fit` on datasets where training is still slower than WEKA, especially `Adult`, `NSL-KDD`, and `UNSW-NB15`. | Unit tests, focused timing/profiling on `Adult`, `NSL-KDD`, `UNSW-NB15`, followed by `fast-vs-strict Layer A/B/C` reruns. | Training time improves clearly on the focal datasets while preserving exact behavior against `strict` and stable alignment with WEKA. |
| `FAST-013` | `P1` | Pruning and subtree-raising cost reduction | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py) | After the accepted `FAST-012` split-search improvements, fit profiles still show non-trivial residual time in pruning and raising (`_prune_tree(...)`, `_estimate_raise_cost(...)`, routing and subtree error estimation). | Better end-to-end `fit`, especially on larger trees where split search is no longer the only dominant block. | Unit tests, focused profiling/timing on `Adult`, `NSL-KDD`, `UNSW-NB15`, then `fast-vs-strict Layer B/C` rerun. | Pruning/raising time drops materially on the focal datasets without changing the exact `strict`-preserving behavior of `j48.fast`. |
| `FAST-014` | `P1` | Dense nominal training path | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py) | The remaining `fit` gap against WEKA is now concentrated in nominal-heavy datasets (`Adult`, `Nursery`, `Mushroom`, `CarEvaluation`, `CreditApproval`). The current nominal split path still builds repeated masks and branch counts in Python. | Better `fit` on the datasets where `j48.fast` is still slower than WEKA, without touching the semantic baseline. | Unit tests, focused serial timing on `Adult`, `Nursery`, `Mushroom`, `CarEvaluation`, `CreditApproval`, then `fast-vs-strict Layer B/C` and `fast-vs-weka` spot checks. | Nominal-heavy datasets show a clear `fit` improvement and at least one of the current WEKA deficits materially narrows without semantic drift. |
| `FAST-015` | `P2` | Clean WEKA-vs-fast timing protocol | [benchmark_j48_fast_baseline.py](/home/javier/VisualStudio/IDEA/scripts/analysis/benchmark_j48_fast_baseline.py), [run_j48_fast_timing_matrix.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_fast_timing_matrix.sh), [generate_j48_fast_validation_artifacts.py](/home/javier/VisualStudio/IDEA/scripts/analysis/generate_j48_fast_validation_artifacts.py) | Timing claims are now the limiting factor for a stronger comparison with WEKA. We need a tighter, explicitly comparable timing protocol to separate algorithmic wins from wrapper overhead. | More defensible timing evidence and clearer visibility into the remaining gap to WEKA. | Serial timing rerun on the current manifest, artifact refresh, and consistency checks against the existing timing summary. | The project has a stable timing protocol that can be cited directly in the paper and used to judge future performance milestones. |
| `FAST-016` | `P2` | Repeated-evaluation throughput inside the EA | [Classificators.py](/home/javier/VisualStudio/IDEA/EvLib/Classificators.py), [verify_j48_fast_recent_optimizations.py](/home/javier/VisualStudio/IDEA/scripts/analysis/verify_j48_fast_recent_optimizations.py), [benchmark_j48_input_protocols.py](/home/javier/VisualStudio/IDEA/scripts/analysis/benchmark_j48_input_protocols.py) | Repeated EA loops still benefit from reusing work inside a worker process, but the earlier fixture-backed preprocessing path proved semantically unsafe and was removed from the runtime baseline. | Lower wall-clock time for repeated chromosome evaluation through transparent in-process caching, without reintroducing alternate prepared-data representations. | Cache-on/off repeated-evaluation checks, direct `mainMP.py` smoke, and explicit confirmation that results stay identical. | Repeated-evaluation runs show a measurable throughput gain from cache reuse alone, with no semantic drift and no fixture-only code in the runtime path. |
| `FAST-017` | `P2` | Code optimization audit | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py), [engine.py](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py), [benchmark_j48_fast_baseline.py](/home/javier/VisualStudio/IDEA/scripts/analysis/benchmark_j48_fast_baseline.py) | We have improved several structural hotspots, but we have not yet run a systematic audit focused on CPU time, allocation churn, copies, and Python/NumPy boundary costs in the current accepted baseline. | A prioritized map of code-level optimization opportunities, separate from algorithmic redesigns and experiment infrastructure. | Allocation-aware profiling on `Adult`, `NSL-KDD`, `UNSW-NB15`, `Mushroom`, and `CreditApproval`, followed by a ranked hotspot report and at least one low-risk cleanup candidate. | The project has a documented hotspot inventory with concrete next actions, including which bottlenecks are algorithmic, which are code-level, and which are experiment-protocol overhead. |
| `FAST-018` | `P1` | Fit-input preparation cost reduction | [engine.py](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py), [profile_j48_fast_code_audit.py](/home/javier/VisualStudio/IDEA/scripts/analysis/profile_j48_fast_code_audit.py) | `FAST-017` showed that `prepare_fit_bundle(...)` and `_encode_numeric_series(...)` still dominate `fit` on several datasets before the tree core even starts. | Lower end-to-end `fit` time on datasets where input preparation now dominates, especially `Adult`, `Mushroom`, and `CreditApproval`, without changing the tree algorithm itself. | Unit tests, focused audit rerun on `Adult`, `Mushroom`, `CreditApproval`, `NSL-KDD`, then `fast-vs-strict` smoke and serial timing rerun. | `prepare_fit_bundle` time drops materially on the focal datasets, `fit` improves, and semantic alignment stays exact. |
| `FAST-019` | `P1` | Structural numeric split-search redesign | [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py), [J48_FAST_CODE_AUDIT.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_CODE_AUDIT.md), [benchmark_j48_fast_baseline.py](/home/javier/VisualStudio/IDEA/scripts/analysis/benchmark_j48_fast_baseline.py) | `FAST-017` and the accepted `FAST-012` work show that the largest remaining single-run opportunity is no longer a local wrapper cleanup but repeated numeric split-search work inside `_build_tree(...)` and `_find_best_numeric_split_candidate(...)`. | Materially lower `fit` time on `Adult`, `NSL-KDD`, and `UNSW-NB15` by reducing repeated sorting/rescoring and moving more node-local numeric preparation into a data-oriented compiled flow. | Unit tests, focused profiling/timing on `Adult`, `NSL-KDD`, `UNSW-NB15`, then `fast-vs-strict Layer A/B/C`, followed by `fast-vs-weka` base and timing spot checks. | The focal datasets show a clear `fit` gain against the current accepted baseline, while `fast-vs-strict` stays exact and WEKA alignment does not regress. |

## Attempt notes

- `FAST-001 / attempt A`
  - change tested: dense nominal branch counts in `core.py` using `bincount`
    over encoded categories
  - outcome: exact against `strict`, but rejected
  - reason: when measured with the metadata-aware `fast-vs-strict` harness, it
    reduced `fit_speedup_vs_reference` on the focal nominal-heavy datasets
    (`Adult`, `Nursery`, `CarEvaluation`, `CreditApproval`)
  - decision: keep the current baseline and revisit `FAST-001` only with a
    different design or stronger profiling evidence

- `FAST-002 / attempt A`
  - change tested: direct nominal child lookup tables in `engine.py` for the
    compiled traversal path
  - outcome: exact against `strict`, but rejected
  - reason: focused timing on the same nominal-heavy datasets showed only mixed
    and small effects; `predict_only` improved slightly in some cases but
    regressed in others, and the net signal was not strong enough to justify
    the extra complexity
  - decision: keep the current traversal baseline and revisit `FAST-002` only
    if later profiling isolates a clearer nominal-lookup bottleneck

- `FAST-003 / attempt A`
  - change tested: reusable encoded input caches in `engine.py`, keyed by the
    source object identity and encoding signature, shared across `j48.fast`
    engine instances in the same process
  - outcome: accepted
  - reason: it preserves exact behavior and gives a clear benefit in repeated
    evaluation scenarios, which is the intended target of `FAST-003`
  - observed signal in same-process repeated runs:
    - `Adult`: second `fit` `1.28x` faster, second `predict` `1.02x` faster
    - `CreditApproval`: second `fit` `1.18x` faster
    - `Nursery`: second `fit` `4.61x` faster, second `predict` `1.13x` faster

- `FAST-004 / attempt A`
  - change tested: node-local dense nominal lookup tables in `engine.py` for
    compiled multiway nominal nodes, used only when branch count and domain
    size suggested a compact dispatch table
  - outcome: exact in focused `fast-vs-strict` harness reruns, but rejected
  - reason: process-isolated timing on nominal-heavy datasets showed only tiny
    and mixed effects relative to the current baseline, not enough to justify
    the extra compiled-tree complexity
  - observed signal vs the accepted timing baseline:
    - `Adult`: `predict_only 1.309x -> 1.312x`
    - `CreditApproval`: `0.577x -> 0.580x`
    - `Nursery`: `1.188x -> 1.141x`
    - `CarEvaluation`: `0.502x -> 0.492x`
    - `Mushroom`: `0.418x -> 0.416x`
  - validation:
    - metadata-aware `fast-vs-strict` harness remained exact on `Adult`,
      `Nursery`, `CarEvaluation`, `CreditApproval`, and `Mushroom`
    - benchmark-only mismatch on `Nursery` was traced to the simple timing
      harness operating without the richer nominal metadata used by the
      differential harness, so it was not treated as a semantic regression
  - decision: keep the simpler compiled tree representation and revisit this
    line only if later profiling isolates nominal exact-match dispatch as a
    dominant residual again

- `FAST-005 / attempt A`
  - change tested: a `numba`-accelerated nominal split scoring path in
    `core.py`, enabled only for `j48.fast`
  - outcome: exact against the current baseline, but rejected
  - reason: focused timing on `Adult`, `Nursery`, `CarEvaluation`, and
    `CreditApproval` showed changes that were too small and too inconsistent to
    justify the extra implementation complexity; the net `fit` signal stayed in
    the range of experimental noise
  - decision: keep the existing nominal scoring baseline and revisit this only
    if later profiling isolates a stronger nominal-training hotspot

- `FAST-006 / attempt A`
  - change tested: targeted inference profiling in `engine.py`,
    `benchmark_j48_fast_baseline.py`, and the timing runner, separating
    `prepare_predict_data`, dispatch/traversal time, chosen path, fallback
    reason, and cache reuse for both `predict` and `predict_proba`
  - outcome: accepted
  - reason: the new profiles clearly distinguish between traversal cost and
    preprocessing/fallback cost, which changes the next optimization target
  - observed signal in focused reruns:
    - `Adult`: `fast` fell back to the core path because of
      `fractional_missing_with_nonfinite_input`; the dominant `predict` cost
      was still `prepare_predict_data`, not traversal
    - `UNSW-NB15`: same fallback reason as `Adult`, with `fast` still winning
      because encoded preparation plus fallback dispatch remained cheaper than
      the `strict` path
    - `CarEvaluation`: also fell back to the core path for the same reason,
      which explains why `predict_only` remained weak despite earlier traversal
      optimizations
    - `CIC-IDS2017`: the compiled path was available; there the main residual
      cost was preparation, while compiled dispatch itself was already very low
  - decision: use this profiling baseline before revisiting inference work; the
    most promising next targets are preparation cost and fallback frequency,
    not nominal branch lookup by itself

- `FAST-009 / attempt A`
  - change tested: `prepare_predict_data` now uses a DataFrame-first path and
    avoids materializing the full input as an `object` array before checking
    whether pandas-native vectorized conversion can be used
  - outcome: accepted
  - reason: focused serial timing showed a clear reduction in preparation cost
    on the main mixed-type datasets identified by `FAST-006`
  - observed signal in focused reruns:
    - `Adult`: `predict_only 1.154x -> 1.194x`, `predict 1.059x -> 2.017x`,
      `predict_proba 0.967x -> 12.076x`; `fast` predict preparation median
      dropped to about `0.0214s`
    - `Nursery`: `predict_only 1.166x -> 1.186x`, `predict 0.824x -> 1.524x`,
      `predict_proba 0.473x -> 6.175x`
    - `CreditApproval`: `predict 0.436x -> 0.692x`,
      `predict_proba 0.419x -> 1.765x`
    - `CIC-IDS2017`: `predict 1.171x -> 1.427x`,
      `predict_proba 2.498x -> 4.162x`, while `predict_only` remained below
      `1.0` and still points to preparation as the dominant residual
    - `CarEvaluation`: mixed result; `predict_only` stayed below `1.0`, but
      end-to-end `predict` and `predict_proba` improved over the previous
      timing baseline
  - decision: keep this change and use it as the new preparation baseline
    before revisiting traversal or fallback-specific optimizations

- `FAST-009 / attempt B`
  - change tested: precomputed pandas label indices plus typed fast paths in
    `_encode_nominal_series_predict(...)` for string-like and scalar-like
    nominal `Series`
  - outcome: exact against the current baseline, but rejected
  - reason: focused serial timing on `Adult`, `Nursery`, `CreditApproval`,
    `CarEvaluation`, and `CIC-IDS2017` showed only mixed and mostly marginal
    changes; small wins on some datasets were offset by regressions on others
  - decision: keep the simpler nominal-series path and revisit this idea only
    if a later profile isolates `_encode_nominal_series_predict(...)` as the
    dominant residual after larger preparation and fallback costs are reduced

- `FAST-010 / attempt A`
  - change tested: missing-aware compiled inference in `engine.py`, replacing
    the blanket fallback for `fractional_missing` plus non-finite or unseen
    nominal inputs with a numba path that reproduces the `core` routing rules
    for:
    - numeric missing values via `left_prob`
    - nominal missing values via `nominal_child_probs`
    - nominal unseen values, including deterministic routing to the explicit
      `__WEKA_OTHER__` branch when present
  - outcome: accepted
  - reason: the new path preserved exactness in unit tests, removed the
    fallback in the missing-heavy focal datasets that motivated the item, and
    stayed exact in the broader `fast-vs-strict` reruns
  - validation:
    - unit tests: `47 tests OK`
    - focused artifact:
      [fast010_focus_20260314.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast010_focus_20260314.json)
    - `Adult`: `match=1.0`, `predict_path=fast_compiled_missing_aware`,
      `predict_speedup=3.15x`
    - `CreditApproval`: `match=1.0`,
      `predict_path=fast_compiled_missing_aware`
    - `UNSW-NB15`: `match=1.0`,
      `predict_path=fast_compiled_missing_aware`,
      `predict_speedup=10.47x`
    - `CarEvaluation`: stayed on the plain compiled path
      (`predict_path=fast_compiled`), which is consistent with the absence of
      the missing/unseen pattern that previously forced fallback
    - `fast-vs-strict Layer A` rerun:
      [j48_fast_vs_strict_synth_20260315_001749](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_vs_strict_synth/j48_fast_vs_strict_synth_20260315_001749/summary.csv)
      with `441/441` exact
    - `fast-vs-strict Layer B/C` remains exact in
      [j48_fast_vs_strict_20260314_232218](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_vs_strict/j48_fast_vs_strict_20260314_232218/summary.csv)
      with `741/741`
    - technical artifacts refreshed in
      [J48_FAST_VALIDATION_RESULTS.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_VALIDATION_RESULTS.md)

- `FAST-011 / attempt A`
  - change tested: lower-overhead `prepare_predict_data(...)` paths in
    `engine.py`, including:
    - cached fast feature metadata derived from the fit bundle
    - direct per-column numeric conversion for pandas inputs instead of
      rebuilding numeric feature sets and using `DataFrame.apply(...)`
    - lightweight caching for the last path-description query so
      `describe_hard_predict_path(...)` / `describe_predict_proba_path(...)`
      do not rescan the same prepared batch immediately before dispatch
  - outcome: accepted
  - reason: focused repeated timing against `HEAD` showed a consistent drop in
    preparation cost on the key inference datasets, with neutral-to-positive
    end-to-end impact and no semantic drift in the existing unit suite
  - validation:
    - unit tests: `47 tests OK`
    - focused repeated benchmark: `3` serial repetitions against `HEAD` on
      `Adult`, `Nursery`, `CreditApproval`, `CarEvaluation`, and `UNSW-NB15`
    - representative median deltas vs `HEAD`:
      - `Adult`: `predict_speedup 2.089x -> 2.262x`,
        `predict_only_speedup 1.200x -> 1.306x`,
        `fast.predict.prepare_time_sec 0.022333s -> 0.020651s`
      - `CreditApproval`: `predict_speedup 0.807x -> 0.967x`,
        `predict_only_speedup 0.466x -> 0.560x`,
        `fast.predict.prepare_time_sec 0.001192s -> 0.000966s`
      - `CarEvaluation`: `predict_speedup 0.465x -> 0.524x`,
        `predict_only_speedup 0.469x -> 0.532x`
      - `UNSW-NB15`: `predict_speedup 9.305x -> 14.678x`,
        `predict_only_speedup 7.883x -> 13.089x`,
        `fast.predict.prepare_time_sec 0.002662s -> 0.001440s`
      - `Nursery`: mixed but acceptable; `predict_only_speedup` stayed nearly
        flat (`1.170x -> 1.156x`) while `predict_speedup` and
        `predict_proba_speedup` remained slightly positive
  - decision: keep this as the new local inference-preparation baseline and
    refresh broader artifacts after the next accepted milestone or dedicated
    timing rerun

- `FAST-011 / attempt B`
  - change tested: a second metadata-aware cleanup of
    `prepare_predict_data(...)` in `engine.py`, specifically:
    - bulk extraction of already-numeric pandas columns when nominal metadata
      is explicit, instead of converting those columns one by one
    - categorical-based encoding for nominal predict-time pandas series when
      the fitted label domains are already available
  - outcome: accepted
  - reason: the generic optimization-target benchmark without explicit
    nominal metadata understated the real inference gain. When rerun in the
    same metadata-aware configuration used by the differential harness, the
    new path reduced `prepare_predict_data(...)` materially across all focal
    datasets while preserving exact prepared arrays and exact `fast-vs-strict`
    smoke behavior
  - validation:
    - unit tests: `52 tests OK`
    - metadata-aware repeated comparison vs the previous accepted baseline:
      [fast011_attempt_b_20260320_161145](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast011_attempt_b_20260320_161145/comparison.csv)
      - `Adult`: `1.606x`
      - `Mushroom`: `1.585x`
      - `CreditApproval`: `1.437x`
      - `NSL-KDD`: `1.623x`
      - `UNSW-NB15`: `1.602x`
      - all prepared arrays stayed exact vs the previous path
    - `fast-vs-strict` smoke on `layer_b,layer_c / base`:
      [j48_fast_vs_strict_20260320_161008](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_vs_strict_fast011_smoke/j48_fast_vs_strict_20260320_161008/summary.csv)
      with `13/13` `OK` and `13/13` exact prediction agreement
  - decision: keep this as the new metadata-aware inference-preparation
    baseline; future `FAST-011` work should beat this checkpoint, not the
    pre-`attempt B` path

- `FAST-011 / attempt C`
  - change tested: a lane split for nominal predict-time preparation in
    `engine.py`, specifically:
    - dedicated categorical-dtype handling that remaps category codes without
      rebuilding a `pd.Categorical(...)` batch when the fit-time label order is
      already available
    - a factorization-based path for non-categorical nominal pandas inputs that
      decides missing/unseen handling over unique values instead of rebuilding a
      `StringDtype` representation for the whole column
    - reuse of the same factorization-based path for ndarray/object predict
      inputs to keep DataFrame-vs-array equivalence exact
  - outcome: accepted
  - reason: focused repeated timing against `HEAD` on the post-`fixture`
    inference focal set showed a clear reduction in predict preparation cost
    without introducing new semantic drift in the canonical differential
    harnesses
  - validation:
    - unit tests: `53 tests OK`
    - focused repeated comparison vs `HEAD`:
      [fast011_attempt_c_20260321](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast011_attempt_c_20260321/comparison.json)
      - `Adult`: `prepare 4.745x`, `predict_total 2.736x`
      - `Mushroom`: `prepare 8.589x`, `predict_total 6.807x`
      - `CarEvaluation`: `prepare 7.241x`, `predict_total 6.129x`
      - `CreditApproval`: `prepare 5.799x`, `predict_total 5.171x`
      - `CIRA2020-AttNorm`: `prepare 1.127x`, `predict_total 1.094x`
    - `fast-vs-strict` smoke on a 5-dataset manifest with
      `base,no_subtree_raising`:
      [j48_fast_vs_strict_20260321_092331](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_vs_strict_fast011c_smoke/j48_fast_vs_strict_20260321_092331/summary.csv)
      with `10/10` `OK` and `10/10` exact prediction agreement
    - `fast-vs-WEKA` smoke on the same manifest/config set:
      [j48_fast_vs_weka_20260321_092354](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_vs_weka_fast011c_smoke/j48_fast_vs_weka_20260321_092354/summary.csv)
      with no new residuals beyond the previously documented
      `CreditApproval` / `Adult(base)` differences
  - decision: keep this as the new inference-preparation baseline and treat it
    as the successor to `FAST-011 / attempt B`

- `FAST-012 / attempt A`
  - change tested: keep the training matrix numeric during `fit` when the
    backend already provides encoded numeric inputs plus explicit
    `nominal_features`, instead of eagerly promoting the full matrix to
    `dtype=object`
  - outcome: rejected
  - reason: repeated timing against `HEAD` on `Adult`, `NSL-KDD`, and
    `UNSW-NB15` did not show a clear `fast.fit` win. The median fit deltas were
    effectively flat on `Adult` and `NSL-KDD`, and slightly worse on
    `UNSW-NB15`, even though end-to-end inference moved for unrelated reasons
    already attributable to `FAST-011`
  - validation:
    - unit tests: `48 tests OK`
    - focused repeated benchmark: `3` serial repetitions against `HEAD` on
      `Adult`, `NSL-KDD`, and `UNSW-NB15`
    - representative median `fast.fit_time_sec` deltas vs `HEAD`:
      - `Adult`: `0.206410s -> 0.205444s`
      - `NSL-KDD`: `0.037599s -> 0.037710s`
      - `UNSW-NB15`: `0.164402s -> 0.171984s`
    - short `cProfile` runs on `Adult` and `UNSW-NB15` still showed the main
      fit hotspot concentrated in:
      - `_find_best_numeric_split_candidate(...)`
      - `_evaluate_numeric_split_candidate_sorted(...)`
      - the numba numeric split kernels
  - decision: keep the existing fit representation and focus the next
    `FAST-012` iteration directly on numeric split search, not on the initial
    matrix conversion step

- `FAST-012 / attempt B`
  - change tested: avoid rebuilding sorted numeric arrays inside
    `_evaluate_numeric_split_candidate_sorted(...)` by preparing `x_sorted`,
    `y_sorted`, and `w_sorted` once in `_find_best_numeric_split_candidate(...)`
    and passing them down directly
  - outcome: rejected
  - reason: repeated timing against `HEAD` on `Adult`, `NSL-KDD`, and
    `UNSW-NB15` again showed effectively flat `fast.fit` behavior. The tiny
    movement in `Adult` and `NSL-KDD` was within noise, and `UNSW-NB15`
    regressed slightly
  - validation:
    - unit tests: `47 tests OK`
    - focused repeated benchmark: `3` serial repetitions against `HEAD` on
      `Adult`, `NSL-KDD`, and `UNSW-NB15`
    - representative median `fast.fit_time_sec` deltas vs `HEAD`:
      - `Adult`: `0.209487s -> 0.209486s`
      - `NSL-KDD`: `0.038104s -> 0.037958s`
      - `UNSW-NB15`: `0.165573s -> 0.166492s`
  - decision: stop pursuing local data-shuffling changes inside the current
    numeric split wrapper and move the next `FAST-012` iteration toward a more
    structural reduction of repeated sorting/rescoring across nodes

- `FAST-012 / attempt C`
  - change tested: move valid-value extraction, local index collection, and
    numeric feature sorting into numba so the `fast` path evaluates each
    numeric feature with a more data-oriented compiled flow instead of doing
    the wrapper work in Python before calling the split-scoring kernels
  - outcome: accepted
  - reason: focused repeated timing against `HEAD` showed clear `fit`
    improvements on the training-heavy targets, with exact prediction matches
    against `strict` in the benchmark harness
  - validation:
    - unit tests: `47 tests OK`
    - focused repeated benchmark: `3` serial repetitions against `HEAD` on
      `Adult`, `NSL-KDD`, and `UNSW-NB15`
    - representative median deltas vs `HEAD`:
      - `Adult`: `fit_speedup 10.919x -> 11.153x`,
        `fast.fit_time_sec 0.209250s -> 0.207536s`
      - `NSL-KDD`: `fit_speedup 4.989x -> 7.610x`,
        `fast.fit_time_sec 0.040028s -> 0.027830s`
      - `UNSW-NB15`: `fit_speedup 3.603x -> 5.582x`,
        `fast.fit_time_sec 0.169299s -> 0.111583s`
    - all focal runs stayed at `prediction_match_fraction = 1.0`
  - decision: keep this as the new `FAST-012` baseline and only revisit the
    numeric split kernels from here with broader validation or a larger
    structural change

- `FAST-013 / attempt A`
  - change tested: cache node-local pruning metrics in `core.py` during
    pessimistic pruning and subtree raising:
    - sample count
    - leaf training error
    - leaf estimated error
    - subtree estimated error
    with explicit invalidation when a split is cleared or a promoted subtree is
    augmented with incoming sibling mass
  - outcome: accepted
  - reason: focused timing on the larger-tree datasets showed a clear `fit`
    improvement, and focused `fast-vs-strict` reruns stayed exactly aligned
    with the frozen semantic baseline
  - validation:
    - unit tests: `48 tests OK`
    - focused repeated benchmark: `3` serial repetitions on `Adult`,
      `NSL-KDD`, and `UNSW-NB15`
    - representative medians:
      - `Adult`: `fit_speedup 11.151x -> 11.777x`
      - `NSL-KDD`: `7.293x -> 7.794x`
      - `UNSW-NB15`: `5.540x -> 6.096x`
    - focused `fast-vs-strict` harness reruns:
      - `Adult`: `match=1.0`, `710 vs 710` nodes
      - `NSL-KDD`: `match=1.0`, `196 vs 196` nodes
      - `UNSW-NB15`: `match=1.0`, `886 vs 886` nodes
  - decision: keep this as the working `FAST-013` baseline and schedule the
    wider rerun / artifact refresh before using it in paper-facing summaries

- `FAST-013 / attempt B`
  - change tested: reduce unnecessary work in
    `_subtree_estimated_errors_with_incoming(...)` by:
    - returning directly to `_subtree_estimated_errors(...)` when there is no
      incoming mass
    - skipping `bincount` / `total_counts` construction for internal nodes
      where those counts are not used
  - outcome: accepted
  - reason: the change is semantically conservative, kept focused
    `fast-vs-strict` spot checks exact, and produced a measurable `fit`
    improvement on the pruning/raising focal datasets
  - validation:
    - unit tests: `52 tests OK`
    - benchmark spot check on `Adult`, `NSL-KDD`, and `UNSW-NB15`
    - representative `fast.fit_time_sec` deltas vs the previous accepted
      baseline:
      - `Adult`: `0.1928s -> 0.1873s` (`-2.8%`)
      - `NSL-KDD`: `0.02528s -> 0.02447s` (`-3.2%`)
      - `UNSW-NB15`: `0.09604s -> 0.08951s` (`-6.8%`)
    - focused `fast-vs-strict` harness reruns:
      - `Adult`: `match=1.0`, `710 vs 710` nodes
      - `NSL-KDD`: `match=1.0`, `196 vs 196` nodes
      - `UNSW-NB15`: `match=1.0`, `886 vs 886` nodes
  - decision: keep this as part of the accepted `FAST-013` line; it is not a
    large enough change to trigger a paper-facing refresh by itself, but it is
    worth preserving for the next broader optimization checkpoint

- `FAST-014 / attempt A`
  - change tested: dense nominal branch counting in `core.py` for encoded
    nominal domains, first as a general fast path and then as a narrower
    variant restricted to large nominal domains only
  - outcome: exact against `strict`, but rejected
  - reason: metadata-aware `fast-vs-strict` focus reruns did not show a
    consistent net fit win across the nominal-heavy datasets that motivated
    this item. Both variants improved `Adult`, but the broader focal set still
    regressed overall.
  - validation:
    - unit tests: `49 tests OK`
    - focused `fast-vs-strict` harness reruns on `Adult`, `Nursery`,
      `Mushroom`, `CarEvaluation`, and `CreditApproval`
    - all focal reruns kept `prediction_match_fraction = 1.0`
    - representative `fit_speedup_vs_reference` movement vs the accepted
      baseline:
      - `Adult`: `0.603x -> 0.843x`
      - `Nursery`: `0.744x -> 0.685x`
      - `Mushroom`: `0.704x -> 0.595x`
      - `CarEvaluation`: `0.738x -> 0.714x`
      - `CreditApproval`: `0.970x -> 0.942x`
  - decision: keep the current nominal-training baseline and revisit this only
    if later per-feature profiling suggests a more targeted design than
    dataset-level dense counting heuristics

- `FAST-014 / attempt B`
  - change tested: metadata-aware dense nominal branch statistics in `core.py`
    for the encoded `j48.fast` training path, with a shared helper reused by
    both multiway and binary nominal candidate scoring and a structural gate
    based on:
    - dense zero-based nominal codes
    - bounded domain size
    - sufficiently large node size
  - outcome: unit-test exact, but rejected
  - reason: even as a general structure-driven gate rather than a dataset
    special case, the added setup cost regressed `fit` on every focused
    metadata-aware nominal-heavy dataset we measured
  - validation:
    - unit tests: `52 tests OK`
    - focused metadata-aware fit medians on `Adult`, `Nursery`, `Mushroom`,
      `CarEvaluation`, and `CreditApproval`
    - representative movement vs the accepted baseline:
      - `Adult`: `2.1049s -> 2.4810s`
      - `Nursery`: `0.1475s -> 0.1681s`
      - `Mushroom`: `0.0499s -> 0.0547s`
      - `CarEvaluation`: `0.0261s -> 0.0305s`
      - `CreditApproval`: `0.0295s -> 0.0350s`
    - a narrower gate (`node_size >= 256`, `8 <= domain_size <= 32`) still
      regressed further and was also discarded:
      - `Adult`: `2.5071s`
      - `Nursery`: `0.1902s`
      - `Mushroom`: `0.0666s`
      - `CarEvaluation`: `0.0324s`
      - `CreditApproval`: `0.0368s`
  - decision: treat broad dense nominal branch statistics as exhausted for the
    current codebase. Do not reopen `FAST-014` without stronger per-feature
    evidence or a materially different nominal-core design.

- `FAST-015 / attempt A`
  - change tested: protocol benchmark comparing three single-run input paths:
    - `CSV -> pandas/DataFrame -> j48.fast`
    - preprocessed binary fixture loaded into RAM
    - preprocessed binary fixture loaded with `mmap`
  - outcome: accepted as diagnostic evidence
  - reason: the result was clear enough to guide implementation strategy:
    binary fixtures are useful for repeated evaluation, but `mmap` alone adds
    very little beyond the binary-format win
  - validation:
    - benchmark script:
      [benchmark_j48_input_protocols.py](/home/javier/VisualStudio/IDEA/scripts/analysis/benchmark_j48_input_protocols.py)
    - focal outputs:
      - [Adult](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/input_protocols_adult.json)
      - [Mushroom](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/input_protocols_mushroom.json)
      - [Nursery](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/input_protocols_nursery.json)
      - [CreditApproval](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/input_protocols_credit_approval.json)
    - all focal runs kept identical prediction digests and tree statistics
  - observed signal on `fit_pipeline_time_sec` (`load + prepare + fit`):
    - binary RAM vs CSV:
      - `Adult`: `1.055x`
      - `Mushroom`: `1.607x`
      - `Nursery`: `1.312x`
      - `CreditApproval`: `1.332x`
    - binary `mmap` vs binary RAM:
      - `Adult`: `0.994x`
      - `Mushroom`: `1.004x`
      - `Nursery`: `1.002x`
      - `CreditApproval`: `1.013x`
  - decision: do not treat `mmap` as a primary optimization target; instead,
    carry the binary-fixture idea forward as EA/campaign preprocessing under a
    dedicated item

- `FAST-016 / attempt A`
  - change tested: explicit preprocessed fixture support for repeated
    `PYJ48_FAST` evaluation inside the EA flow
  - outcome: rejected and removed from the runtime baseline
  - reason: although the fixture path produced speedups, it failed the
    transparency requirement for scientific runs
  - validation:
    - global fixture-on/off audit:
      - `fixture_equal_rows = 14/30`
      - `metrics_equal_rows = 15/30`
    - direct `mainMP.py` runs confirmed that the fixture path, not the cache,
      was what changed EA trajectories
  - interpretation:
    - fixture-backed preprocessing added runtime complexity but did not remain
      semantically equivalent across representative datasets
  - decision:
    - remove fixture support from the runtime path
    - keep only the audit evidence as a record of what not to revive without a
      materially different design and a full equivalence proof

- `FAST-016 / attempt B`
  - change tested: EA-facing profile tuning around the strongest first-round
    preset (`p3off_leaf1`), comparing:
    - `p3off_leaf1`
    - `p3off_leaf1_cf015`
    - `p3off_leaf1_cf025`
    - `p3off_leaf1_cf035`
    - `p3off_leaf1_nocollapse`
    - `p3off_leaf1_unpruned`
  - implementation:
    - tuning runner in
      [run_j48_fast_ea_tuning_round2.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_fast_ea_tuning_round2.sh)
    - summary note in
      [J48_FAST_EA_PROFILE_TUNING.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_EA_PROFILE_TUNING.md)
  - outcome: accepted as EA configuration guidance
  - reason: the run isolates a stable default preset for EA usage without
    changing tree-core semantics
  - validation:
    - aggregate artifact:
      [aggregate.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_ea_profiles/j48_fast_ea_profiles_20260316_095809/aggregate.csv)
    - `60/60` runs `OK`
    - focal result:
      - `p3off_leaf1` remains the best general-purpose profile
      - `cf025` and `nocollapse` are redundant and slower
      - `cf015` regresses quality and should be rejected
      - `unpruned` and `cf035` are only worth keeping as dataset-specific
        alternatives, mainly for `CIC-IDS2017`
  - decision:
    - treat `p3off_leaf1` as the default EA preset candidate for `PYJ48_FAST`
    - keep `unpruned` and `cf035` only as optional targeted presets

- `FAST-016 / attempt C`
  - change tested: per-process repeated-evaluation cache for `PYJ48_FAST`
    results keyed by chromosome inside `evaluateModel(...)`
  - implementation:
    - cache storage and lifecycle in
      [Classificators.py](/home/javier/VisualStudio/IDEA/EvLib/Classificators.py)
    - cache invalidation tied to `_clear_split_cache()` so it resets between
      generations and configuration changes
  - design tested:
    - add a second cache only for `clID == 9`, storing the already computed
      scalar metrics plus structured confusion outputs for identical
      chromosomes inside the same worker process
    - reuse cached results before any data prep or fit work on repeated
      chromosomes
  - target scenarios:
    - EA populations with duplicate chromosomes inside a generation
    - repeated evaluation loops using `PYJ48_FAST` inside the same process
  - outcome: accepted
  - reason: this directly improves the intended `FAST-016` workload
    (repeated evaluation) and produced a large throughput gain in a controlled
    repeated-subset benchmark without changing model semantics
  - validation:
    - unit tests: `tests.test_c45_j48_strict`
    - focused repeated-evaluation verification in
      [verify_j48_fast_recent_optimizations.py](/home/javier/VisualStudio/IDEA/scripts/analysis/verify_j48_fast_recent_optimizations.py)
    - repeated-evaluation benchmark artifact:
      [comparison.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast016_attempt_c_20260320/comparison.json)
  - observed signal on a 20-evaluation loop with 4 unique masks repeated 5
    times:
    - `CreditApproval`: `5.74x`
    - `Mushroom`: `4.95x`
    - `NSL-KDD`: `5.02x`
  - semantic checks:
    - repeated-loop metric digests stayed identical with and without the cache
    - direct `mainMP.py` cache on/off runs kept the same final chromosome and
      the same final metrics in the focused smoke
  - decision:
    - keep this as the current repeated-evaluation throughput baseline for
      `PYJ48_FAST`
    - treat the cache as a per-process EA optimization, not as a single-run
      speed claim

- `FAST-017 / attempt A`
  - change tested: formal in-process code audit with `cProfile` and
    `tracemalloc`, implemented in
    [profile_j48_fast_code_audit.py](/home/javier/VisualStudio/IDEA/scripts/analysis/profile_j48_fast_code_audit.py)
    and summarized in
    [J48_FAST_CODE_AUDIT.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_CODE_AUDIT.md)
  - outcome: accepted as the first formal optimization audit baseline
  - reason: the audit cleanly separates code-level hotspots from runner or
    protocol overhead and leaves a ranked hotspot inventory that can drive the
    next optimization wave
  - validation:
    - focal artifact:
      [fast017_code_audit_20260315.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast017_code_audit_20260315.json)
    - focal datasets:
      - `Adult`
      - `NSL-KDD`
      - `UNSW-NB15`
      - `Mushroom`
      - `CreditApproval`
  - main findings:
    - fit is split between:
      - code-level input preparation in `prepare_fit_bundle(...)` and
        `_encode_numeric_series(...)`
      - algorithmic work in `_build_tree(...)`, numeric split search, and the
        numba numeric kernels
    - pruning and subtree raising remain a real fit residual
    - inference is now dominated by `prepare_predict_data(...)`, not compiled
      traversal
    - the largest net allocations are array materialization in fit/predict
      preparation, not tree traversal
    - campaign-level process sleep states do not explain the core runtime
      picture; the same hotspots appear in-process
  - decision: use this audit as the baseline for future code-level
    optimization work and prefer the next wave on fit preparation, numeric
    split search, and pruning/raising before revisiting traversal micro-tweaks

- `FAST-018 / attempt A`
  - change tested: a DataFrame-first fit-preparation path in `engine.py`
    intended to reduce the cost of `prepare_fit_bundle(...)` by:
    - avoiding earlier full-matrix conversion for pandas inputs
    - replacing repeated `_encode_numeric_series(...)` calls with broader
      numeric-frame conversion
  - outcome: exact in the unit suite, but rejected
  - reason: the focused code-audit rerun showed only mixed results and did not
    materially reduce `prepare_fit_bundle(...)` on the datasets that motivated
    the item
  - validation:
    - unit tests: `50 tests OK`
    - audit rerun:
      [fast018_code_audit_20260315_b.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast018_code_audit_20260315_b.json)
  - representative movement vs the accepted `FAST-017` audit baseline:
    - `Adult`: `fit 0.9188s -> 0.9296s`,
      `prepare_fit_bundle 0.5403s -> 0.5571s`
    - `NSL-KDD`: `fit 0.0982s -> 0.0891s`,
      `prepare_fit_bundle 0.0141s -> 0.0079s`
    - `UNSW-NB15`: `fit 0.4521s -> 0.4185s`
    - `Mushroom`: `fit 0.1919s -> 0.1925s`,
      `prepare_fit_bundle 0.1899s -> 0.1905s`
    - `CreditApproval`: `fit 0.0267s -> 0.0264s`
  - decision: keep the baseline fit-preparation path for now and revisit
    `FAST-018` only with a more targeted design, likely distinguishing between
    mostly-numeric frames and object-heavy frames rather than applying a
    broader pandas conversion path everywhere

- `FAST-018 / attempt B`
  - change tested: numeric-major fit preparation path
  - design tested:
    - detect DataFrames where numeric columns dominate
    - extract the numeric block in one vectorized operation
    - keep nominal-column handling separate and explicit
    - avoid broad per-column conversion for already-typed numeric features
  - target datasets:
    - `NSL-KDD`
    - `UNSW-NB15`
    - mixed numeric-heavy cases from `Layer B`
  - outcome: exact in unit tests, but rejected
  - reason: a controlled same-process benchmark with metadata-aware harness
    setup showed a nearly null net effect on `fit`
  - validation:
    - unit tests: `51 tests OK`
    - same-process A/B benchmark, toggling only the numeric-major path while
      keeping the same metadata, backend warmup, and dataset objects
  - observed signal:
    - `Adult`: `1.00029x`
    - `CreditApproval`: `0.99792x`
    - `NSL-KDD`: `1.00064x`
    - `UNSW-NB15`: `1.00045x`
  - decision: keep the baseline fit-preparation path and do not continue with
    this line; the effect is too small to justify extra branching complexity

- `FAST-018 / attempt C`
  - change tested: categorical-major fit preparation path with fixed category
    codes for pandas nominal `Series`
  - design tested:
    - keep the baseline array path unchanged
    - for object-heavy pandas nominal columns, use explicit
      category-domain-driven coding
    - prefer fixed category codes over repeated value-to-dict mapping when the
      domain is already known
    - keep the output as dense numeric arrays for the tree core
  - target datasets:
    - `Adult`
    - `Mushroom`
    - `Nursery`
    - `CreditApproval`
  - outcome: accepted
  - reason: the gain is modest, but it is consistent across the object-heavy
    focal datasets and preserves exact alignment against `strict`
  - validation:
    - unit tests: `52 tests OK`
    - same-process A/B benchmark artifact:
      [fast018c_same_process_benchmark_20260315.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast018c_same_process_benchmark_20260315.json)
    - focused `fast-vs-strict` harness reruns:
      - [Adult](/tmp/fast018c_adult/comparison.json)
      - [Mushroom](/tmp/fast018c_mushroom/comparison.json)
      - [Nursery](/tmp/fast018c_nursery/comparison.json)
      - [CreditApproval](/tmp/fast018c_credit/comparison.json)
  - observed signal in same-process median `fit` time:
    - `Adult`: `1.0136x`
    - `Mushroom`: `1.0587x`
    - `Nursery`: `1.0232x`
    - `CreditApproval`: `1.0185x`
  - semantic checks:
    - all four focal `fast-vs-strict` reruns stayed at `match = 1.0`
    - node counts matched exactly
  - decision: keep this as the current object-heavy fit-preparation baseline,
    but treat it as a modest improvement rather than a major milestone

- `FAST-018 / attempt D`
  - change tested: pandas/object-heavy fit preparation path that bypasses the
    eager full-frame `object` materialization and bulk-extracts already-typed
    numeric columns
  - design tested:
    - activate only when metadata already marks at least half of the features
      as nominal and `auto_detect_nominal=False`
    - keep the accepted categorical-major nominal encoding path from
      `attempt C`
    - avoid `np.asarray(X)` on the accepted DataFrame path
    - bulk-load numeric pandas columns in one `to_numpy(dtype=float64)` call
      while keeping non-numeric fallbacks unchanged
    - keep the output as the same dense `float64` matrix expected by the tree
      core
  - target datasets:
    - `Adult`
    - `Mushroom`
    - `Nursery`
    - `CreditApproval`
    - `NSL-KDD` as a numeric-heavy control
  - outcome: accepted
  - reason: the gain is small, but it is consistently non-negative on the
    object-heavy focal datasets, stays neutral on the control dataset, and
    preserves exact alignment against the previous `fast` baseline and
    `strict`
  - validation:
    - unit tests: `52 tests OK`
    - same-process A/B benchmark artifact:
      [comparison.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast018d_attempt_20260320/comparison.json)
    - focused `fast-vs-strict` smoke embedded in the same artifact
  - observed signal in same-process median wall-clock:
    - `Adult`: `prepare 1.0353x`, `fit 1.0060x`
    - `Mushroom`: `prepare 0.9983x`, `fit 1.0002x`
    - `Nursery`: `prepare 1.0008x`, `fit 1.0128x`
    - `CreditApproval`: `prepare 1.0129x`, `fit 1.0177x`
    - `NSL-KDD`: `prepare 0.9705x`, `fit 1.0055x`
  - semantic checks:
    - `equal_prepared_arrays = true` on all five focal datasets
    - `prediction_match_fraction = 1.0` against `strict`
    - tree stats matched exactly
  - decision: keep this as the current object-heavy fit-preparation baseline,
    but treat it as an incremental cleanup rather than a large milestone

- `FAST-019 / planned design`
  - focus: numeric split search in [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py), not another round of local pandas or wrapper cleanups
  - design intent:
    - reduce repeated feature-local sorting and rescoring across nodes
    - carry more node-local numeric partition state in compiled form
    - keep threshold relocation and split semantics unchanged so the work stays in the performance layer
  - initial implementation direction:
    - build a fit-time numeric-order cache or partition-friendly index structure
    - let node growth consume that structure instead of rebuilding local sorted views feature by feature
    - push as much as possible of the “valid values + local order + scan candidates” path into `numba`
  - focal datasets:
    - `Adult`
    - `NSL-KDD`
    - `UNSW-NB15`
  - note:
    - any later EA-facing flag or helper should live under `FAST-016`
      (preprocessed repeated evaluation), not inside the numeric training core

- `FAST-019 / attempt A`
  - change tested: precomputed global per-feature numeric rank maps in
    [core.py](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py), then node-local sorting by rank instead of by raw feature value
  - outcome: rejected
  - reason: the design was structurally cleaner than earlier local wrapper
    tweaks, but focused timing still regressed the two most important numeric
    targets
  - validation:
    - dedicated workload audit:
      [fast019_numeric_split_audit_20260315.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_numeric_split_audit_20260315.json)
    - unit tests: `52 tests OK`
    - focused benchmark on `Adult`, `NSL-KDD`, and `UNSW-NB15`
  - audit signal from the accepted baseline:
    - numeric split search calls across the three focal datasets: `25,451`
    - wrapper-level split-search time share of `fit`: about `42%` to `46%`
  - observed benchmark signal vs the accepted baseline:
    - `Adult`: `fit_speedup 11.95x -> 12.25x`
    - `NSL-KDD`: `7.85x -> 5.17x`
    - `UNSW-NB15`: `6.17x -> 3.48x`
  - decision:
    - keep the audit tooling and the `FAST-019` structural focus
    - reject this specific rank-map design
    - next attempts should avoid replacing one per-node sort with another sort
      over local ranks and instead target partition-preserving order reuse
      across nodes

- `FAST-019 / attempt B`
  - change tested: partition-preserving sorted numeric blocks for the `fast`
    path on numeric datasets without missing values
  - design tested:
    - build one sorted global index list per numeric feature at fit start
    - reuse those ordered lists during node growth instead of re-sorting local
      feature values
    - partition child feature lists stably after a numeric split so child
      nodes inherit already ordered numeric views
    - keep threshold relocation and split semantics unchanged
  - outcome: mixed after broad validation; not yet accepted as a milestone
  - reason: unlike `attempt A`, this design improved the focused numeric `fit`
    benchmarks, but the broad validation signal was too small and uneven to
    justify the added complexity in `core.py`
  - validation:
    - unit tests: `52 tests OK`
    - numeric split workload audit artifact:
      [fast019_numeric_split_audit_attempt_b_20260315.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_numeric_split_audit_attempt_b_20260315.json)
    - focused fit benchmark artifact:
      [fast019_attempt_b_focus_20260315.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_attempt_b_focus_20260315.json)
    - focused exactness check on `Adult`, `NSL-KDD`, and `UNSW-NB15`:
      `prediction_match_fraction = 1.0`, matching tree stats, zero probability checksum drift
  - observed signal vs the accepted baseline:
    - `Adult`: `fit_speedup 11.95x -> 14.73x`
    - `NSL-KDD`: `7.85x -> 8.49x`
    - `UNSW-NB15`: `6.17x -> 6.23x`
  - broad validation:
    - `fast-vs-strict Layer A`:
      [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_vs_strict_synth/j48_fast_vs_strict_synth_20260315_165047/summary.csv)
      -> `441/441` exactos
    - `fast-vs-strict Layer B/C`:
      [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_vs_strict/j48_fast_vs_strict_20260315_165129/summary.csv)
      -> `741/741` exactos
    - `fast-vs-weka Layer B/C`:
      [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_vs_weka/j48_fast_vs_weka_20260315_165440/summary.csv)
      -> `match_mean = 0.999428`, unchanged from the previous baseline
    - timing:
      [summary.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_timing/j48_fast_timing_20260315_165923/summary.csv)
      -> medians `fit = 4.3049x`, `predict = 1.1295x`,
      `predict_only = 0.6225x`, `predict_proba = 2.7968x`
  - comparison against the last committed timing baseline:
    - `fit`: `4.2740x -> 4.3049x`
    - `predict`: `1.1335x -> 1.1295x`
    - `predict_only`: `0.6208x -> 0.6225x`
    - `predict_proba`: `2.7433x -> 2.7968x`
  - decision:
    - keep the audit tooling and the design notes
    - do not promote this attempt as an accepted optimization milestone yet
    - if revisited later, it should come back with a broader structural gain
      than the current narrow numeric-only/no-missing path

- `FAST-019 / attempt C`
  - change tested: conservative local-order reuse for numeric/no-missing nodes,
    carrying per-feature local sorted orders through child partitioning
  - outcome: rejected
  - reason: fidelity stayed exact on the focused datasets, but the added
    partition/remap overhead regressed `fit` noticeably relative to the
    accepted baseline
  - validation:
    - unit tests: `52 tests OK`
    - focused benchmark artifacts:
      - [adult.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_attempt_c_20260320/adult.json)
      - [nsl.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_attempt_c_20260320/nsl.json)
      - [unsw.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_attempt_c_20260320/unsw.json)
    - focused `fast-vs-strict` spot checks:
      - [adult/comparison.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_strict_spot/adult/comparison.json)
      - [nsl/comparison.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_strict_spot/nsl/comparison.json)
      - [unsw/comparison.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_strict_spot/unsw/comparison.json)
  - observed signal vs the accepted baseline:
    - `Adult`: `fast.fit_time_sec 0.1873s -> 0.2363s` (`+26.1%`)
    - `NSL-KDD`: `0.02447s -> 0.02646s` (`+8.1%`)
    - `UNSW-NB15`: `0.08951s -> 0.09464s` (`+5.7%`)
  - decision:
    - reject this local-order partitioning design
    - keep the fidelity evidence; do not revive this family without a more
      compelling cost model

- `FAST-019 / attempt D`
  - change tested: top-of-tree-only reuse of global sorted numeric orders with
    strict gating (`numeric-only`, `no-missing`, shallow depth, large nodes)
  - outcome: rejected
  - reason: this narrowed version preserved fidelity but still failed to beat
    the accepted baseline on any focused dataset
  - validation:
    - unit tests: `52 tests OK`
    - focused benchmark artifacts:
      - [adult.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_attempt_d_20260320/adult.json)
      - [nsl.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_attempt_d_20260320/nsl.json)
      - [unsw.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_attempt_d_20260320/unsw.json)
  - observed signal vs the accepted baseline:
    - `Adult`: `0.1873s -> 0.1927s` (`+2.9%`)
    - `NSL-KDD`: `0.02447s -> 0.02574s` (`+5.2%`)
    - `UNSW-NB15`: `0.08951s -> 0.09423s` (`+5.3%`)
  - decision:
    - reject this top-of-tree order-reuse design
    - do not spend more time on order-caching variants without stronger
      evidence from a lower-level kernel audit

- `FAST-019 / node-size audit`
  - change tested: new dedicated audit tooling to measure numeric split-search
    cost by node depth and node-size bucket
  - outcome: accepted as permanent diagnostic tooling
  - artifact:
    - script:
      [profile_j48_fast_numeric_split_node_audit.py](/home/javier/VisualStudio/IDEA/scripts/analysis/profile_j48_fast_numeric_split_node_audit.py)
    - focused audit output:
      [fast019_node_audit_20260320.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast019_node_audit_20260320.json)
  - signal:
    - `Adult`: large top-level nodes dominate split-search time
    - `NSL-KDD`: medium-sized nodes carry a large share of the residual
    - `UNSW-NB15`: cost is fragmented across many small nodes
  - decision:
    - keep this auditor for future `FAST-019` work
    - next attempts should target lower-level kernel/materialization cost, not
      another tree-level order-caching scheme

- `FAST-007 / attempt A`
  - change tested: minimal `Cython` prototype around the current binary
    numeric `unsorted` kernel with an explicit experimental gate
  - outcome: rejected
  - reason: fidelity stayed exact on the focused datasets, but the prototype
    did not produce a meaningful `fit` gain against the accepted baseline
  - validation:
    - unit tests: `52 tests OK`
    - focused fidelity spot on `Adult`, `NSL-KDD`, and `UNSW-NB15`:
      `prediction_match_fraction = 1.0` with identical tree stats
    - focused repeated `fit` benchmark against the accepted baseline:
      - `Adult`: `0.15106s -> 0.17478s` (`-13.6%`)
      - `NSL-KDD`: `0.02194s -> 0.02177s` (`+0.8%`)
      - `UNSW-NB15`: `0.08828s -> 0.08882s` (`-0.6%`)
  - decision:
    - reject this thin compiled-wrapper design
    - if `FAST-007` is revisited later, it should target a redesigned
      lower-level kernel, not a direct Cython port of the current one

## Recommended execution order

1. `FAST-018` fit-input preparation cost reduction on object-heavy datasets
2. `FAST-011` predict-input preparation cost reduction
3. `FAST-016` repeated-evaluation cache throughput inside the EA
4. `FAST-015` clean WEKA-vs-fast timing protocol
5. `FAST-019` only through a lower-level kernel/materialization redesign
6. revisit `FAST-013` with a second pass only if pruning/raising remains the
   limiting `fit` residual after the previous steps
7. revisit `FAST-014` only with stronger per-feature evidence than the first
   rejected attempt
8. revisit `FAST-004` only if later profiling points back to nominal
   dispatch as a dominant residual
9. `FAST-008` artifact refresh after each accepted milestone
10. `FAST-007` only if the previous steps plateau again with a genuinely new
    compiled-kernel design

Historical note:
- `FAST-001`, `FAST-002`, and `FAST-005` were already explored and rejected at
  the current checkpoint because they did not show a strong enough performance
  return for the added complexity.

## Minimal validation protocol per optimization

Every accepted `j48.fast` optimization should trigger:

1. `python -m unittest tests.test_c45_j48_strict -q`
2. `fast-vs-strict Layer A`
3. `fast-vs-strict Layer B/C`
4. `fast-vs-WEKA Layer B/C`
5. serial timing rerun
6. artifact regeneration:
   - [generate_j48_fast_validation_artifacts.py](/home/javier/VisualStudio/IDEA/scripts/analysis/generate_j48_fast_validation_artifacts.py)

## Paper relevance

This backlog is directly useful for the manuscript because it maps each future
optimization to:

- the expected speed signal,
- the semantic-risk envelope,
- and the validation evidence required before reporting a stronger claim.

That makes it possible to discuss `j48.fast` as a controlled engineering line
rather than as a sequence of ad hoc speed hacks.
