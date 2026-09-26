"""Differential tests against WEKA's J48 (`weka.jar`).

Skipped unless `J48_WEKA_CLASSPATH` points at a WEKA 3.8 classpath, e.g.::

    J48_WEKA_CLASSPATH=weka-stable-3.8.6.jar:bounce-0.18.jar pytest tests/test_weka_parity.py

Each case fits WEKA, `J48Classifier` and `J48FastClassifier` on the same
random ARFF split with equivalent options and requires:

* the same tree size and number of leaves;
* the same class probabilities on the test set as WEKA's
  `distributionForInstance` (up to rounding);
* the same predicted classes as WEKA's `J48.classifyInstance`, which picks
  the first class whose probability exceeds the previous best by more than
  `Utils.SMALL` (1e-6) and ignores Laplace smoothing (`-A` only smooths the
  probabilities).
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

from j48 import J48Classifier, J48FastClassifier
from weka_datasets import FAMILIES

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

SEEDS = range(6)
WEKA_SMALL = 1e-6


def _write_arff(path: Path, X, y, n_classes: int, domains: dict) -> None:
    lines = ["@relation parity"]
    lines += [f"@attribute n{j} numeric" for j in range(3)]
    for j in sorted(domains):
        lines.append(f"@attribute c{j} {{" + ",".join(domains[j]) + "}")
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
        base + ["-classifications", "weka.classifiers.evaluation.output.prediction.CSV -distribution -decimals 17"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    block = output[output.index("inst#"):].strip().split("\n\n")[0]
    rows = list(csv.reader(io.StringIO(block)))[1:]
    # Columns: inst#, actual, predicted, error, then one probability per
    # class (the predicted one prefixed with "*").
    proba = np.array([[float(cell.lstrip("*")) for cell in row[4:]] for row in rows])
    return leaves, size, proba


def _classify_instance(proba: np.ndarray) -> np.ndarray:
    """WEKA's ClassifierTree.classifyInstance class choice (Utils.gr)."""
    best = np.full(proba.shape[0], -1.0)
    idx = np.zeros(proba.shape[0], dtype=int)
    for j in range(proba.shape[1]):
        take = proba[:, j] > best + WEKA_SMALL
        idx[take] = j
        best[take] = proba[take, j]
    return idx


def _cases():
    for mode in MODES:
        for family in FAMILIES:
            for seed in SEEDS:
                for nominal in (False, True):
                    for missing in (False, True):
                        case_id = (
                            f"{mode}|{family}{seed}|{'nominal' if nominal else 'numeric'}"
                            f"|{'missing' if missing else 'complete'}"
                        )
                        yield pytest.param(mode, family, seed, nominal, missing, id=case_id)


@pytest.mark.parametrize("mode,family,seed,nominal,missing", list(_cases()))
def test_matches_weka(tmp_path, mode, family, seed, nominal, missing):
    flags, params = MODES[mode]
    X, y, n_classes, domains = FAMILIES[family](seed, nominal, missing)
    n_train = len(y) * 2 // 3
    train, test = tmp_path / "train.arff", tmp_path / "test.arff"
    _write_arff(train, X[:n_train], y[:n_train], n_classes, domains)
    _write_arff(test, X[n_train:], y[n_train:], n_classes, domains)

    weka_leaves, weka_size, weka_proba = _run_weka(train, test, flags)

    if "-A" in flags:
        # -A does not change J48.classifyInstance: its classes come from the
        # unsmoothed distribution, i.e. from the same model without -A.
        unsmoothed = [flag for flag in flags if flag != "-A"]
        weka_pred = _classify_instance(_run_weka(train, test, unsmoothed)[2])
    else:
        weka_pred = _classify_instance(weka_proba)

    extra = {"nominal_features": sorted(domains), "nominal_value_domains": domains} if domains else {}
    for estimator in (J48Classifier, J48FastClassifier):
        clf = estimator(**params, **extra).fit(X[:n_train], y[:n_train])
        stats = clf.get_tree_stats()
        assert (stats["leaf_count"], stats["node_count"]) == (weka_leaves, weka_size), estimator.__name__
        np.testing.assert_allclose(clf.predict_proba(X[n_train:]), weka_proba, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(clf.predict(X[n_train:]), weka_pred)
