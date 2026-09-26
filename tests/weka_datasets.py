"""Random datasets shared by the WEKA parity tests and the WEKA-semantics
regression tests. Each generator returns ``(X, y, n_classes, domains)``,
where ``domains`` maps nominal column indices to their values."""

from __future__ import annotations

import numpy as np


def _family_a(seed: int, nominal: bool, missing: bool):
    """Continuous features and one 4-valued nominal feature."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(80, 400))
    k = int(rng.integers(2, 4))
    numeric = np.round(rng.normal(size=(n, 3)), 3)
    cat = rng.choice(list("abcd"), size=n)
    signal = numeric[:, 0] + 0.8 * numeric[:, 1] * (cat == "a") + rng.normal(scale=0.8, size=n)
    y = np.digitize(signal, np.quantile(signal, np.linspace(0, 1, k + 1)[1:-1]))

    columns = [numeric[:, 0], numeric[:, 1], numeric[:, 2]] + ([cat] if nominal else [])
    X = np.column_stack(columns).astype(object)
    for j in range(3):
        X[:, j] = [float(v) for v in X[:, j]]
    if missing:
        X[rng.random(X.shape) < 0.08] = None
    domains = {3: list("abcd")} if nominal else {}
    return X, y, k, domains


def _family_b(seed: int, nominal: bool, missing: bool):
    """IDS-like: small-integer features (many ties), a skewed 4-valued and a
    10-valued nominal feature, three classes."""
    rng = np.random.default_rng(10_000 + seed)
    n = int(rng.integers(100, 500))
    ints = rng.integers(0, 6, size=(n, 3)).astype(float)
    proto = rng.choice(list("tuic"), size=n, p=[0.5, 0.3, 0.15, 0.05])
    services = [f"s{i}" for i in range(10)]
    svc = rng.choice(services, size=n)
    signal = ints[:, 0] + (proto == "t") * 1.5 + (svc == "s1") * 2 + (svc == "s2") + rng.normal(scale=1.0, size=n)
    y = np.digitize(signal, np.quantile(signal, [1 / 3, 2 / 3]))

    columns = [ints[:, 0], ints[:, 1], ints[:, 2]] + ([proto, svc] if nominal else [])
    X = np.column_stack(columns).astype(object)
    for j in range(3):
        X[:, j] = [float(v) for v in X[:, j]]
    if missing:
        X[rng.random(X.shape) < 0.10] = None
    domains = {3: list("tuic"), 4: services} if nominal else {}
    return X, y, 3, domains


FAMILIES = {"A": _family_a, "B": _family_b}
