"""Regression tests for the defects fixed in 0.2.0."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from j48 import J48Classifier, J48FastClassifier

ESTIMATORS = [J48Classifier, J48FastClassifier]


@pytest.fixture
def iris():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_predictions_work_with_sklearn_metrics(cls, iris):
    X, y = iris
    clf = cls().fit(X, y)
    pred = clf.predict(X)

    assert clf.classes_.dtype == y.dtype
    assert pred.dtype == y.dtype
    assert clf.score(X, y) == pytest.approx(accuracy_score(y, pred))
    assert clf.score(X, y) > 0.9
    scores = cross_val_score(cls(), X, y, cv=3)
    assert scores.shape == (3,)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_string_labels_are_preserved(cls, iris):
    X, y = iris
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    clf = cls().fit(X, names)
    assert set(clf.predict(X)) <= set(names)
    assert clf.score(X, names) > 0.9


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_predict_proba_is_float64_and_normalized(cls, iris):
    X, y = iris
    proba = cls().fit(X, y).predict_proba(X)
    assert proba.dtype == np.float64
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_empty_prediction_keeps_class_dtype(cls, iris):
    X, y = iris
    clf = cls().fit(X, y)
    pred = clf.predict(X[:0])
    assert pred.shape == (0,)
    assert pred.dtype == clf.classes_.dtype


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_large_integer_features_keep_precision(cls):
    # 2**24 and 2**24 + 1 collapse to the same float32 value.
    X = np.array([[16777216], [16777217]] * 4, dtype=np.int64)
    y = np.array([0, 1] * 4)
    clf = cls(min_num_obj=1).fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), y)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_predict_sees_in_place_changes_to_ndarray(cls):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = cls().fit(X, y)

    X_test = X[:20].copy()
    before = clf.predict(X_test)
    X_test[:, 0] *= -1
    after = clf.predict(X_test)

    np.testing.assert_array_equal(after, clf.predict(X_test.copy()))
    np.testing.assert_array_equal(after, 1 - before)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_predict_sees_in_place_changes_to_dataframe(cls):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = (x > 0).astype(int)
    df = pd.DataFrame({"c": np.where(x > 0, "pos", "neg"), "x": rng.normal(size=200)})
    clf = cls(nominal_features=[0]).fit(df, y)

    df_test = df.iloc[:20].copy()
    before = clf.predict(df_test)
    df_test.loc[:, "c"] = np.where(df_test["c"] == "pos", "neg", "pos")
    after = clf.predict(df_test)

    np.testing.assert_array_equal(after, clf.predict(df_test.copy()))
    np.testing.assert_array_equal(after, 1 - before)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_refit_sees_in_place_changes_to_training_data(cls):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"c": rng.choice(["a", "b"], 300), "x": rng.normal(size=300)})
    y = (df["c"] == "a").astype(int).to_numpy()
    cls(nominal_features=[0]).fit(df, y)

    # Make "c" constant and "x" perfectly predictive, without copying.
    df.loc[:, "x"] = np.where(df["c"] == "a", 5.0, -5.0)
    df.loc[:, "c"] = "a"
    clf = cls(nominal_features=[0]).fit(df, y)

    assert clf.export_tree()["root"]["feature_name"] == "x"


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_fit_prepared_bundle_refit_uses_new_tree(cls):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y_first = (X[:, 0] > 0).astype(int)
    y_second = (X[:, 1] > 0).astype(int)

    clf = cls()
    clf._ensure_engine()
    bundle = clf.engine_.prepare_fit_bundle(X, y_first)
    clf.fit_prepared_bundle(bundle)
    clf.predict(X)

    clf.fit_prepared_bundle({**bundle, "y": y_second})
    np.testing.assert_array_equal(clf.predict(X), y_second)
    np.testing.assert_array_equal(clf.predict_proba(X).argmax(axis=1), y_second)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_pickle_and_clone(cls, iris):
    X, y = iris
    clf = cls(confidence_factor=0.1).fit(X, y)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(restored.predict(X), clf.predict(X))
    assert clone(clf).get_params() == clf.get_params()
