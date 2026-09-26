# Changelog

All notable changes to this project are documented in this file.
The project follows [Semantic Versioning](https://semver.org/); while the
version is `0.x`, a minor bump may change public behavior.

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

[0.2.0]: https://github.com/javiermaldonadoc/j48-python-backend/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/javiermaldonadoc/j48-python-backend/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/javiermaldonadoc/j48-python-backend/releases/tag/v0.1.0
