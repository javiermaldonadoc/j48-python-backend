"""The fast backend must reproduce the strict reference line."""

from __future__ import annotations

import numpy as np
import pytest

from j48 import J48Classifier, J48FastClassifier

CONFIGS = [
    {},
    {"unpruned": True},
    {"reduced_error_pruning": True, "random_state": 1},
    {"binary_splits": True},
    {"use_laplace": True},
    {"subtree_raising": False},
    {"confidence_factor": 0.1, "min_num_obj": 5},
    {"fractional_missing": False},
]


def _mixed_dataset(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Numeric + nominal columns with ~10% missing values in each."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(80, 300))
    numeric = rng.normal(size=(n, 3))
    nominal = rng.choice(list("abcd"), size=n)
    signal = numeric[:, 0] + (nominal == "a") + rng.normal(scale=0.7, size=n)
    y = np.digitize(signal, [0.0, 1.0])
    numeric[rng.random((n, 3)) < 0.1] = np.nan
    nominal = nominal.astype(object)
    nominal[rng.random(n) < 0.1] = None
    return np.column_stack([numeric, nominal]).astype(object), y


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: ",".join(f"{k}={v}" for k, v in c.items()) or "default")
@pytest.mark.parametrize("seed", range(4))
def test_fast_matches_strict(config, seed):
    X, y = _mixed_dataset(seed)
    strict = J48Classifier(nominal_features=[3], **config).fit(X, y)
    fast = J48FastClassifier(nominal_features=[3], **config).fit(X, y)

    assert fast.get_tree_stats() == strict.get_tree_stats()
    np.testing.assert_array_equal(fast.predict(X), strict.predict(X))
    np.testing.assert_allclose(fast.predict_proba(X), strict.predict_proba(X), atol=1e-9)
