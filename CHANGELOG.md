# Changelog

All notable changes to this project are documented in this file.
The project follows [Semantic Versioning](https://semver.org/); while the
version is `0.x`, a minor bump may change public behavior.

## [0.4.0] - 2026-09-26

Performance release. Trees, split features, thresholds and predictions are
identical to 0.3.0 (192 randomized strict/fast configurations, and the
WEKA parity suite gives the same results); stored split statistics and
probabilities differ only by floating-point rounding (< 3e-14), because
tied values are now summed in a different order.

### Benchmarks (vs 0.3.0, same machine, single runs)

| Workload | Line | Fit 0.3.0 | Fit 0.4.0 | Peak memory 0.3.0 → 0.4.0 |
|---|---|---|---|---|
| 100k x 20 numeric, 5% missing | fast | 16.9 s | **3.1 s** (5.5x) | 49.2 → 49.9 MiB |
| 100k x 20 numeric, 5% missing | strict | 11.2 s | **7.8 s** (1.4x) | 38.7 → 35.1 MiB |
| 60k x 41 IDS-like (3 nominal, 70-value service) | fast | 28.9 s | **20.0 s** (-31%) | |
| 60k x 41 IDS-like | strict | 197 s | 207 s (+5%, within run-to-run noise at 20k rows) | |
| 30k x 15, 5 nominal (8 values) | fast / strict | 4.3 / 15.6 s | 4.4 / 15.2 s | |

`J48FastClassifier.predict` on 100k x 20 numeric rows: 0.11 s → 0.014 s.

### Changed

- Numeric features are sorted once per fit and each child inherits a
  narrowed order in O(n), instead of re-sorting every feature at every
  node (classic C4.5 presorting). For many-way nominal splits, children
  re-sort when that is cheaper than narrowing.
- Fast line: all numeric features of a node are evaluated in a single
  numba call; nominal split search uses one `bincount` per feature on the
  encoded codes instead of one mask per value.
- Fast line: numeric columns are encoded vectorized (was a per-value
  Python loop, ~65% of fit time on numeric data), and all-numeric
  prediction input is used without copying.
- Order buffers are released along the recursion path, keeping peak
  memory at or below 0.3.0 levels.
- Mask-free binary entropy in the strict line (bitwise identical).

### Removed

- Internal numba kernels that sorted inside every call
  (`_find_best_*_numeric_split_unsorted_numba`) and
  `C45TreeClassifier._find_best_numeric_split_candidate`, superseded by
  the presorted path. They were private.

### Added

- Property test for order narrowing (NumPy and numba implementations).

## [0.3.0] - 2026-09-26

scikit-learn conformance and WEKA parity release. Trees are unchanged
versus 0.2.0 except with `unpruned=True, collapse_tree=False` (see Fixed):
the 192-configuration comparison against 0.2.0 gives identical trees,
predictions and probabilities.

### Fixed

- **`collapse_tree=False` is honored for unpruned trees** (J48 `-U -O`).
  Subtrees were always collapsed in unpruned mode. As in WEKA's
  `C45PruneableClassifierTree`, collapsing now only happens when
  `collapse_tree=True`. Against `weka.jar` 3.8.6, `-U -O` trees went from
  4/24 to 24/24 identical.
- Nominal matching no longer relies on NumPy comparing numeric arrays with
  strings, which returns a scalar (and a `FutureWarning`) on NumPy < 2.

### Changed (input validation, scikit-learn conventions)

- `predict()` / `predict_proba()` raise `ValueError` for 1D input instead
  of treating it as a single row; use `X.reshape(1, -1)`.
  (`predict_prepared()` / `predict_proba_prepared()` keep accepting it.)
- Feature-count mismatches raise the scikit-learn message
  `X has N features, but <Estimator> is expecting M features as input.`
- `fit()` now rejects, with explicit errors: `y=None`; continuous
  (regression) targets; `y` containing NaN or infinity (drop those rows
  first; WEKA also discards instances with a missing class); sparse `X`
  (`TypeError`); empty `X` (0 samples or 0 features); complex data,
  including complex scalars inside `object` arrays or DataFrame columns;
  and `sample_weight` that is all zeros.
- A column-vector `y` is raveled with a `DataConversionWarning`.
- Missing values (NaN, None, `"?"`, and infinity, which is treated as
  missing) and non-numeric nominal columns are still accepted, as before.
- Predicting on 0 rows still returns an empty array.
- Added `__sklearn_tags__` (scikit-learn >= 1.6) alongside `_more_tags`.

`check_estimator` now passes except for one documented, by-design failure:
`check_sample_weight_equivalence_on_dense_data`. WEKA's J48 itself does not
treat integer weights as repeated rows (verified with `weka.jar`: 1/20
random cases equivalent with zero weights, 11/20 without), so the package
keeps WEKA's semantics.

### Added

- `tests/test_weka_parity.py`: differential tests against WEKA's J48 over 9
  option sets x 24 datasets (numeric/nominal, with/without missing values),
  comparing tree size, leaves and test predictions. Enabled by setting
  `J48_WEKA_CLASSPATH`; 9 known divergences, all with nominal attributes,
  are listed as strict expected failures.
- `tests/test_sklearn_compat.py`: `parametrize_with_checks` for both
  estimators plus tests for each validation rule.
- GitHub Actions workflow: Python 3.10-3.13, a run without numba, a run on
  the oldest supported dependencies (NumPy 1.23, scikit-learn 1.2, SciPy
  1.10, pandas 1.5, numba 0.58), and the WEKA parity suite on WEKA 3.8.6.

### Known limitations

- With a nominal attribute, J48 can still differ from WEKA in a few cases
  (listed in `tests/test_weka_parity.py`), mostly with `-R` and `-B`.
- With instance weights that include zeros, trees can differ from WEKA,
  which keeps zero-weight rows as candidate split boundaries.

## [0.2.0] - 2026-09-26

Correctness release. On ordinary inputs the trees are unchanged: fitting
192 randomized strict/fast configurations (numeric, integer, nominal and
missing values; pruning, REP, `-B`, `-A`, `-U` variants) with 0.1.1 and
0.2.0 gives identical exported trees and identical predictions. The changes
below only affect the cases described.

### Fixed

- **Predictions are usable with `sklearn.metrics`.** `classes_` was always
  stored as an `object` array, so `predict()` returned `object` values and
  `clf.score()`, `accuracy_score`, `cross_val_score` and scored
  `GridSearchCV` raised `ValueError: Classification metrics can't handle a
  mix of ... and unknown targets`. `classes_` now keeps the label dtype, as
  scikit-learn estimators do.
- **`J48FastClassifier` no longer reuses stale inputs.** Encoded training
  and prediction matrices were cached in module-level caches keyed on
  `id(X)`, shape and dtype. Changing `X` in place (or a new object reusing a
  freed `id`) made `fit()` train on the old data and `predict()` return the
  old predictions. Both caches were removed. The same applied to the
  per-call prediction-path memo, which was also keyed on `id(X)`.
- **`fit_prepared_bundle()` refits are used for prediction.** Re-fitting an
  estimator through `fit_prepared_bundle()` reused the engine's previously
  compiled tree, so the fast engine kept predicting with the old tree. The
  compiled tree is now rebuilt whenever the fitted tree changes.
- **Integer features keep full precision in the strict line.**
  `J48Classifier` cast non-float inputs to `float32`, merging distinct
  integer values above 2^24 (large byte counts, timestamps, flow
  identifiers). Inputs are now cast to `float64`. This also removes a
  divergence between the strict and fast lines on such data.

### Changed

- `predict_proba()` returns `float64` instead of `float32` in both lines.
  Probabilities differ from 0.1.1 only by float32 rounding (< 2e-7), and
  strict/fast probabilities now agree to machine precision.
- Empty predictions (`predict(X[:0])`) keep the dtype of `classes_`.
- `NumpyJ48Engine.get_last_prepare_predict_cache_hit()` is kept for
  compatibility but always returns `False`.

### Added

- Test suite under `tests/` (`pytest`): regression tests for each fix above
  and strict-vs-fast agreement checks across 8 configurations. Install with
  `pip install -e ".[test]"`.
- `test` optional-dependency group.

### Documentation

- Fixed stale docstrings in `j48.core` (module docstring placement, import
  path in the `C45TreeClassifier` example, leftover untranslated text).

## [0.1.1]

- Artifact hardening: packaging metadata (`pyproject.toml`), license,
  paper reproduction guide and README updates.
- Fewer array allocations when routing weighted instances in `j48.core`.

## [0.1.0]

- Paper-facing artifact snapshot.

[0.4.0]: https://github.com/javiermaldonadoc/j48-python-backend/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/javiermaldonadoc/j48-python-backend/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/javiermaldonadoc/j48-python-backend/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/javiermaldonadoc/j48-python-backend/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/javiermaldonadoc/j48-python-backend/releases/tag/v0.1.0
