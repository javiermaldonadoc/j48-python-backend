"""scikit-learn API conformance."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import sklearn
from scipy import sparse
from sklearn.exceptions import DataConversionWarning

from j48 import J48Classifier, J48FastClassifier

ESTIMATORS = [J48Classifier, J48FastClassifier]
SKLEARN_VERSION = tuple(int(part) for part in sklearn.__version__.split(".")[:2])

# Checks that J48 fails by design, with the reason.
EXPECTED_FAILED_CHECKS = {
    "check_sample_weight_equivalence_on_dense_data": (
        "J48 follows WEKA, where integer instance weights are not equivalent "
        "to repeated rows: numeric split points are relocated to observed "
        "values and zero-weight rows still count as candidate boundaries."
    ),
}


if SKLEARN_VERSION >= (1, 6):
    from sklearn.utils.estimator_checks import parametrize_with_checks

    @parametrize_with_checks(
        [J48Classifier(), J48FastClassifier()],
        expected_failed_checks=lambda estimator: EXPECTED_FAILED_CHECKS,
    )
    def test_sklearn_estimator_checks(estimator, check):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check(estimator)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_predict_rejects_1d_input(cls, data):
    X, y = data
    clf = cls().fit(X, y)
    with pytest.raises(ValueError, match="Expected a 2D array"):
        clf.predict(X[0])


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_predict_rejects_wrong_feature_count(cls, data):
    X, y = data
    clf = cls().fit(X, y)
    with pytest.raises(ValueError, match="X has 2 features, but .* is expecting 3 features"):
        clf.predict(X[:, :2])
    with pytest.raises(ValueError, match="X has 2 features"):
        clf.predict_proba(X[:, :2])


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_sparse_input_is_rejected(cls, data):
    X, y = data
    with pytest.raises(TypeError, match="sparse"):
        cls().fit(sparse.csr_matrix(X), y)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_empty_training_data_is_rejected(cls, data):
    X, y = data
    with pytest.raises(ValueError, match="0 sample"):
        cls().fit(X[:0], y[:0])
    with pytest.raises(ValueError, match="0 feature"):
        cls().fit(X[:, :0], y)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize(
    "bad_y, message",
    [
        (None, "requires y"),
        ("continuous", "Unknown label type"),
        ("nan", "NaN"),
    ],
)
def test_invalid_targets_are_rejected(cls, data, bad_y, message):
    X, y = data
    if bad_y == "continuous":
        bad_y = X[:, 0]
    elif bad_y == "nan":
        bad_y = y.astype(float)
        bad_y[0] = np.nan
    with pytest.raises(ValueError, match=message):
        cls().fit(X, bad_y)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_column_vector_target_warns_and_fits(cls, data):
    X, y = data
    with pytest.warns(DataConversionWarning):
        clf = cls().fit(X, y.reshape(-1, 1))
    assert clf.predict(X).shape == (len(y),)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_all_zero_sample_weights_are_rejected(cls, data):
    X, y = data
    with pytest.raises(ValueError, match="non-zero"):
        cls().fit(X, y, sample_weight=np.zeros(len(y)))


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_missing_values_and_strings_are_still_accepted(cls):
    X = np.array([[1.0, "a"], [np.nan, "b"], [3.0, None], [4.0, "?"], [5.0, "a"], [6.0, "b"]] * 5, dtype=object)
    y = np.array([0, 1, 0, 1, 1, 0] * 5)
    clf = cls(nominal_features=[1], min_num_obj=1).fit(X, y)
    assert clf.predict(X).shape == (len(y),)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_infinite_values_are_treated_as_missing(cls, data):
    X, y = data
    X_inf = X.copy()
    X_inf[:5, 0] = np.inf
    X_nan = X.copy()
    X_nan[:5, 0] = np.nan
    np.testing.assert_array_equal(
        cls().fit(X_inf, y).predict_proba(X_inf),
        cls().fit(X_nan, y).predict_proba(X_nan),
    )


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_collapse_tree_flag_applies_to_unpruned_trees(cls):
    # With -U, J48 still collapses subtrees that do not reduce training error
    # unless -O is given. Disabling collapse must never shrink the tree.
    rng = np.random.default_rng(3)
    changed = 0
    for _ in range(10):
        X = rng.normal(size=(300, 4))
        y = (X[:, 0] + rng.normal(scale=1.5, size=300) > 0).astype(int)
        collapsed = cls(unpruned=True).fit(X, y).get_tree_stats()["node_count"]
        full = cls(unpruned=True, collapse_tree=False).fit(X, y).get_tree_stats()["node_count"]
        assert full >= collapsed
        changed += full > collapsed
    assert changed > 0
