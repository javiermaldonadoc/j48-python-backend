# J48 Fast Code Audit

This document records the first formal code-optimization audit for
`j48.fast`.

Artifact:

- [fast017_code_audit_20260315.json](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_diagnostics/fast017_code_audit_20260315.json)

Scope:

- estimator under study: `J48FastClassifier`
- protocol: in-process profiling with `cProfile` and `tracemalloc`
- sections:
  - `fit`
  - `prepare_predict_data`
  - `predict`
  - `predict_proba`
- focal datasets:
  - `Adult`
  - `NSL-KDD`
  - `UNSW-NB15`
  - `Mushroom`
  - `CreditApproval`

The audit intentionally bypasses the large shell runners so the results reflect
code-level cost inside `EvLib/j48`, not process orchestration or logging
overhead.

## Dataset Picture

| Dataset | Fit (s) | Prepare Predict (s) | Predict (s) | Predict Proba (s) | Path |
| --- | ---: | ---: | ---: | ---: | --- |
| `Adult` | `0.9188` | `0.2661` | `0.0042` | `0.0036` | `fast_compiled_missing_aware` |
| `NSL-KDD` | `0.0982` | `0.0102` | `0.0011` | `0.0008` | `fast_compiled_missing_aware` |
| `UNSW-NB15` | `0.4521` | `0.0095` | `0.0011` | `0.0010` | `fast_compiled_missing_aware` |
| `Mushroom` | `0.1919` | `0.0868` | `0.0010` | `0.0008` | `fast_compiled_missing_aware` |
| `CreditApproval` | `0.0267` | `0.0058` | `0.0006` | `0.0005` | `fast_compiled_missing_aware` |

## Main Findings

### 1. Fit still splits into two different costs

The main `fit` picture is not a single bottleneck:

- input preparation cost:
  - [engine.py:703](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py#L703) `prepare_fit_bundle`
  - [engine.py:605](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py#L605) `_encode_numeric_series`
- algorithmic tree-building cost:
  - [core.py:1806](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1806) `_build_tree`
  - [core.py:1346](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1346) `_find_best_numeric_split_candidate`
  - [core.py:421](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L421) `_find_best_binary_numeric_split_unsorted_numba`
  - [core.py:615](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L615) `_find_best_multiclass_numeric_split_unsorted_numba`

On `Adult`, input preparation is especially visible:

- `prepare_fit_bundle`: `0.540s`
- `_encode_numeric_series`: `0.526s`
- `core.fit`: `0.376s`

On `NSL-KDD` and `UNSW-NB15`, the bottleneck shifts toward the tree core:

- `_build_tree`
- numeric split search
- numba numeric kernels

### 2. Pruning and subtree raising are still a real residual

Even after the accepted pruning cache work, the audit still shows:

- [core.py:2900](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L2900) `_prune_tree`
- [core.py:2835](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L2835) `_estimate_raise_cost`
- [core.py:2729](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L2729) `_subtree_estimated_errors_with_incoming`

These are not the top cost everywhere, but they remain a meaningful residual
in `Adult`, `NSL-KDD`, `UNSW-NB15`, and `CreditApproval`.

### 3. Inference is dominated by preparation, not traversal

The traversal path is now consistently:

- `fast_compiled_missing_aware`

for all five focal datasets.

That matters because it means the old fallback concern is no longer the main
story inside the code.

The dominant inference function is now:

- [engine.py:847](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py#L847) `prepare_predict_data`

and not compiled traversal itself.

Examples:

- `Adult`: `prepare_predict_data = 0.266s`, while `predict = 0.004s`
- `Mushroom`: `prepare_predict_data = 0.087s`, while `predict = 0.001s`

### 4. The largest allocations are array materialization, not traversal

The top net allocations repeatedly point to:

- [engine.py:766](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py#L766)
  `X_fast = np.empty(X_arr.shape, dtype=np.float64)` during fit preparation
- [engine.py:887](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py#L887)
  `out = np.empty((n_rows, n_cols), dtype=np.float64)` during predict
  preparation
- [core.py:1865](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1865)
  weighted class counts per node

This points to copy/materialization pressure, not a compiled traversal problem.

### 5. The campaign-level `S` states do not explain the core bottleneck

This audit ran in-process and still found the same dominant costs:

- fit preparation
- numeric split search
- pruning/raising
- predict preparation

So the shell runner, process creation, or occasional sleeping state in the OS
may affect wall-clock campaign throughput, but they do not explain the core
runtime picture inside `j48.fast`.

## Recommended Next Targets

### Highest-value code targets

1. Reduce fit input preparation cost

- focus:
  - [engine.py:703](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py#L703)
  - [engine.py:605](/home/javier/VisualStudio/IDEA/EvLib/j48/engine.py#L605)
- likely direction:
  - faster numeric `DataFrame` ingestion
  - fewer per-column conversions
  - fewer full-matrix materializations

2. Keep attacking numeric split search in training

- focus:
  - [core.py:1346](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L1346)
  - [core.py:421](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L421)
  - [core.py:615](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L615)

3. Revisit pruning and subtree raising as a second-order fit residual

- focus:
  - [core.py:2900](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L2900)
  - [core.py:2835](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L2835)
  - [core.py:2729](/home/javier/VisualStudio/IDEA/EvLib/j48/core.py#L2729)

### Lower-value targets right now

- compiled traversal micro-optimizations
- nominal dispatch tweaks in inference
- `mmap` as a primary runtime optimization

The audit does not support prioritizing those ahead of fit preparation and
numeric training cost.
