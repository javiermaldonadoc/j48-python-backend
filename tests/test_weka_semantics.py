"""Regression tests for the WEKA comparison rules adopted in 0.5.0.

They run without WEKA; `test_weka_parity.py` checks the same behavior
against `weka.jar`.
"""

from __future__ import annotations

import numpy as np
import pytest

from j48 import J48Classifier, J48FastClassifier
from j48.core import C45TreeClassifier
from weka_datasets import FAMILIES

ESTIMATORS = [J48Classifier, J48FastClassifier]


def _split(family: str, seed: int, nominal: bool, missing: bool):
    X, y, _, domains = FAMILIES[family](seed, nominal, missing)
    n_train = len(y) * 2 // 3
    extra = {"nominal_features": sorted(domains), "nominal_value_domains": domains} if domains else {}
    return X[:n_train], y[:n_train], X[n_train:], extra


def test_first_best_gain_follows_the_sequential_rule():
    def reference(gains, known_weight):
        best, best_idx = 0.0, -1
        for i, gain in enumerate(gains):
            gain = 0.0 if abs(gain * known_weight) < 1e-6 else gain
            if gain > best + 1e-6:
                best, best_idx = gain, i
        return best_idx

    rng = np.random.default_rng(0)
    for _ in range(5000):
        gains = rng.choice([0.0, 0.3, 1e-7]) + rng.integers(-3, 4, int(rng.integers(0, 30))) * rng.choice(
            [1e-7, 4e-7, 1e-6, 1e-3]
        )
        if rng.random() < 0.3:
            gains = np.sort(gains)
        known_weight = float(rng.choice([0.5, 1.0, 50.0]))
        assert C45TreeClassifier._weka_first_best_gain(gains, known_weight) == reference(gains, known_weight)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_numeric_threshold_ties_keep_the_first_threshold(cls):
    # A 5-instance node where "n2 <= 0" and "n2 <= 1" have the same gain up
    # to rounding. WEKA keeps the first threshold (Utils.gr); 0.4.0 took the
    # second one because its gain was larger in the last bit.
    X_train, y_train, _, extra = _split("B", 8, nominal=True, missing=False)
    clf = cls(unpruned=True, collapse_tree=False, **extra).fit(X_train, y_train)
    thresholds = [
        node["threshold"]
        for node in clf.iter_tree_nodes()
        if not node["is_leaf"] and node["feature_index"] == 2 and node["n_samples"] == pytest.approx(5.0)
    ]
    assert thresholds == [0.0]


def test_probability_ties_go_to_the_first_class_in_both_lines():
    # Rows with a missing value reach two leaves whose mixture is 0.5/0.5 up
    # to rounding. Like WEKA's classifyInstance, both lines predict the
    # first class; in 0.4.0 the strict line followed the rounding and
    # disagreed with the fast line.
    X_train, y_train, X_test, extra = _split("A", 19, nominal=False, missing=True)
    predictions = []
    for cls in ESTIMATORS:
        clf = cls(**extra).fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        ties = np.abs(proba[:, 1] - proba[:, 0]) < 1e-9
        assert ties.sum() > 0
        assert np.all(clf.predict(X_test)[ties] == clf.classes_[0])
        predictions.append(clf.predict(X_test))
    np.testing.assert_array_equal(predictions[0], predictions[1])


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("family,seed", [("A", 0), ("B", 2), ("B", 13)])
def test_laplace_only_changes_probabilities(cls, family, seed):
    # WEKA's -A smooths distributionForInstance; classifyInstance, and so
    # the predicted classes, ignore it.
    X_train, y_train, X_test, extra = _split(family, seed, nominal=True, missing=True)
    plain = cls(**extra).fit(X_train, y_train)
    smoothed = cls(use_laplace=True, **extra).fit(X_train, y_train)
    np.testing.assert_array_equal(smoothed.predict(X_test), plain.predict(X_test))
    assert not np.allclose(smoothed.predict_proba(X_test), plain.predict_proba(X_test))


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_no_split_when_only_many_valued_nominals_have_a_model(cls):
    # The numeric column is constant (no split model) and the nominal one has
    # 4 values for 10 rows (>= 30%), so it is left out of the gain average.
    # With no model in the average, WEKA 3.8.6 returns a single leaf.
    X = np.array([[1.0, value] for value in "aaabbbccdd"], dtype=object)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    clf = cls(nominal_features=[1], nominal_value_domains={1: list("abcd")}).fit(X, y)
    assert clf.get_tree_stats()["node_count"] == 1
