"""Order narrowing used by the presorted numeric split search."""

from __future__ import annotations

import numpy as np
import pytest

from j48.core import NUMBA_AVAILABLE, C45TreeClassifier, _SortedOrders


@pytest.mark.parametrize(
    "use_numba",
    [False, pytest.param(True, marks=pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed"))],
)
def test_subset_sorted_orders_matches_reference(use_numba):
    rng = np.random.default_rng(0)
    tree = C45TreeClassifier(use_numba_numeric_kernel=use_numba)
    for _ in range(500):
        n_positions = int(rng.integers(1, 60))
        n_features = int(rng.integers(1, 6))
        # Includes empty per-feature orders (features with no known values).
        arrays = [rng.permutation(n_positions)[: int(rng.integers(0, n_positions + 1))] for _ in range(n_features)]
        orders = _SortedOrders.from_arrays(list(range(n_features)), arrays)
        positions = rng.permutation(n_positions)[: int(rng.integers(0, n_positions + 1))]

        narrowed = tree._subset_sorted_orders(orders, positions, n_positions)

        child_index = {int(p): i for i, p in enumerate(positions)}
        for feat, order in enumerate(arrays):
            expected = [child_index[int(v)] for v in order if int(v) in child_index]
            assert narrowed[feat].tolist() == expected
