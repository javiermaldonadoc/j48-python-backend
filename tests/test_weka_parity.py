"""Differential tests against WEKA's J48 (`weka.jar`).

Skipped unless `J48_WEKA_CLASSPATH` points at a WEKA 3.8 classpath, e.g.::

    J48_WEKA_CLASSPATH=weka-stable-3.8.6.jar:bounce-0.18.jar pytest tests/test_weka_parity.py

Each case fits WEKA and `J48Classifier` on the same random ARFF split with
equivalent options and requires the same tree size, number of leaves and
test-set predictions.
"""

from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from j48 import J48Classifier

WEKA_CLASSPATH = os.environ.get("J48_WEKA_CLASSPATH")

pytestmark = [
    pytest.mark.weka,
    pytest.mark.skipif(
        not WEKA_CLASSPATH or shutil.which("java") is None,
        reason="set J48_WEKA_CLASSPATH (and install Java) to run WEKA parity tests",
    ),
]

# name: (J48 command-line flags, equivalent J48Classifier parameters)
MODES = {
    "default": ([], {}),
    "-U": (["-U"], {"unpruned": True}),
    "-U -O": (["-U", "-O"], {"unpruned": True, "collapse_tree": False}),
    "-O": (["-O"], {"collapse_tree": False}),
    "-S": (["-S"], {"subtree_raising": False}),
    "-B": (["-B"], {"binary_splits": True}),
    "-A": (["-A"], {"use_laplace": True}),
    "-C 0.1 -M 5": (["-C", "0.1", "-M", "5"], {"confidence_factor": 0.1, "min_num_obj": 5}),
    "-R": (["-R", "-N", "3", "-Q", "1"], {"reduced_error_pruning": True, "num_folds": 3, "random_state": 1}),
}

# Known divergences from WEKA 3.8.6, all on datasets with a nominal attribute.
# Keyed by (mode, seed, nominal, missing). Strict xfail: fixing one of them
# makes the test fail until it is removed from this list.
KNOWN_DIVERGENCES = {
    ("default", 1, True, False),
    ("-O", 1, True, False),
    ("-A", 1, True, False),
    ("-B", 3, True, False),
    ("-B", 4, True, False),
    ("-R", 0, True, False),
    ("-R", 1, True, True),
    ("-R", 2, True, True),
    ("-R", 4, True, False),
}

SEEDS = range(6)
NOMINAL_DOMAIN = ["a", "b", "c", "d"]


def _dataset(seed: int, nominal: bool, missing: bool):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(80, 400))
    k = int(rng.integers(2, 4))
    numeric = np.round(rng.normal(size=(n, 3)), 3)
    cat = rng.choice(NOMINAL_DOMAIN, size=n)
    signal = numeric[:, 0] + 0.8 * numeric[:, 1] * (cat == "a") + rng.normal(scale=0.8, size=n)
    y = np.digitize(signal, np.quantile(signal, np.linspace(0, 1, k + 1)[1:-1]))

    columns = [numeric[:, 0], numeric[:, 1], numeric[:, 2]] + ([cat] if nominal else [])
    X = np.column_stack(columns).astype(object)
    for j in range(3):
        X[:, j] = [float(v) for v in X[:, j]]
    if missing:
        X[rng.random(X.shape) < 0.08] = None
    return X, y, k


def _write_arff(path: Path, X, y, n_classes: int, nominal: bool) -> None:
    lines = ["@relation parity"]
    lines += [f"@attribute n{j} numeric" for j in range(3)]
    if nominal:
        lines.append("@attribute c {" + ",".join(NOMINAL_DOMAIN) + "}")
    lines.append("@attribute class {" + ",".join(str(i) for i in range(n_classes)) + "}")
    lines.append("@data")
    for row, label in zip(X, y):
        cells = ["?" if v is None else (repr(v) if isinstance(v, float) else str(v)) for v in row]
        lines.append(",".join(cells) + f",{label}")
    path.write_text("\n".join(lines) + "\n")


def _run_weka(train: Path, test: Path, flags: list[str]):
    base = ["java", "-cp", WEKA_CLASSPATH, "weka.classifiers.trees.J48", "-t", str(train), "-T", str(test), *flags]
    model = subprocess.run(base, capture_output=True, text=True, check=True).stdout
    leaves = int(re.search(r"Number of Leaves\s*:\s*(\d+)", model).group(1))
    size = int(re.search(r"Size of the tree\s*:\s*(\d+)", model).group(1))
    output = subprocess.run(
        base + ["-classifications", "weka.classifiers.evaluation.output.prediction.CSV"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    block = output[output.index("inst#"):].strip().split("\n\n")[0]
    predictions = np.array([int(row[2].split(":")[1]) for row in list(csv.reader(io.StringIO(block)))[1:]])
    return leaves, size, predictions


def _cases():
    for mode in MODES:
        for seed in SEEDS:
            for nominal in (False, True):
                for missing in (False, True):
                    marks = []
                    if (mode, seed, nominal, missing) in KNOWN_DIVERGENCES:
                        marks.append(pytest.mark.xfail(strict=True, reason="known divergence from WEKA"))
                    case_id = f"{mode}|seed={seed}|{'nominal' if nominal else 'numeric'}|{'missing' if missing else 'complete'}"
                    yield pytest.param(mode, seed, nominal, missing, id=case_id, marks=marks)


@pytest.mark.parametrize("mode,seed,nominal,missing", list(_cases()))
def test_matches_weka(tmp_path, mode, seed, nominal, missing):
    flags, params = MODES[mode]
    X, y, n_classes = _dataset(seed, nominal, missing)
    n_train = len(y) * 2 // 3
    train, test = tmp_path / "train.arff", tmp_path / "test.arff"
    _write_arff(train, X[:n_train], y[:n_train], n_classes, nominal)
    _write_arff(test, X[n_train:], y[n_train:], n_classes, nominal)

    weka_leaves, weka_size, weka_pred = _run_weka(train, test, flags)

    extra = {"nominal_features": [3], "nominal_value_domains": {3: NOMINAL_DOMAIN}} if nominal else {}
    clf = J48Classifier(**params, **extra).fit(X[:n_train], y[:n_train])
    stats = clf.get_tree_stats()

    assert (stats["leaf_count"], stats["node_count"]) == (weka_leaves, weka_size)
    np.testing.assert_array_equal(clf.predict(X[n_train:]), weka_pred)
