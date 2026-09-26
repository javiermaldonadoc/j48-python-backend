# j48-python-backend

`j48-python-backend` is a Python implementation of a J48-targeting decision-tree backend extracted from the IDEA research workspace.

The repository is intentionally focused on the software artifact that supports the paper. It exposes a scikit-learn-compatible API, keeps the strict semantics-oriented implementation available, and includes a faster encoded backend used for engineering evaluation.

## Scope

This repository contains:

- a J48-oriented core classifier,
- scikit-learn-compatible wrappers for training and inference,
- backend-selection utilities for strict and fast execution modes,
- a small analysis helper used in validation post-processing.

This repository does not contain:

- redistributed IDS datasets,
- experiment dumps or local run artifacts,
- internal planning notes, checklists, or engineering backlogs.

## Public API

The main public entry points are:

- `j48.J48Classifier`: strict, semantics-oriented estimator,
- `j48.J48FastClassifier`: faster estimator using an encoded internal backend,
- `j48.C45TreeClassifier`: lower-level tree implementation,
- `j48.build_engine`: backend-construction utility.

## Requirements

Core runtime dependencies:

- Python 3.10+
- `numpy`
- `scikit-learn`

Additional analysis utilities in `j48.acceptance_analysis` also use:

- `pandas`
- `scipy`

Optional acceleration support:

- `numba`

## Installation

Install a tagged release from the repository once it has been published:

```bash
python -m pip install "git+https://github.com/javiermaldonadoc/j48-python-backend.git@<tag>"
```

For local editable use:

```bash
git clone https://github.com/javiermaldonadoc/j48-python-backend.git
cd j48-python-backend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

Minimal runtime installation only requires the core dependencies:

```bash
python -m pip install -e .
```

Optional extras:

- `analysis`: installs `pandas` and `scipy`
- `fast`: installs `numba`
- `test`: installs `pytest` (and `pandas`) to run the test suite
- `all`: installs `pandas`, `scipy` and `numba`

Quick smoke test:

```bash
python -c "from j48 import J48Classifier, J48FastClassifier; print(J48Classifier, J48FastClassifier)"
```

## Minimal usage

```python
import numpy as np

from j48 import J48Classifier

X = np.array([
    [0.0, 1.0],
    [0.0, 0.0],
    [1.0, 1.0],
    [1.0, 0.0],
], dtype=float)
y = np.array([0, 0, 1, 1])

clf = J48Classifier()
clf.fit(X, y)

pred = clf.predict(X)
proba = clf.predict_proba(X)
tree_stats = clf.get_tree_stats()
```

## Running the tests

```bash
python -m pip install -e ".[all,test]"
python -m pytest
```

The suite covers the regressions fixed in each release, scikit-learn API conformance (`check_estimator`), and checks that `J48FastClassifier` reproduces `J48Classifier` (trees, predictions and probabilities) across the main J48 configurations.

### WEKA parity tests

`tests/test_weka_parity.py` fits WEKA's J48 and `J48Classifier` on the same random ARFF splits for 9 option sets (`-U`, `-O`, `-S`, `-B`, `-A`, `-R`, `-C/-M`, ...) and compares tree size, number of leaves and test predictions. It needs Java and a WEKA 3.8 classpath:

```bash
base=https://repo1.maven.org/maven2/nz/ac/waikato/cms/weka
curl -sSfLO $base/weka-stable/3.8.6/weka-stable-3.8.6.jar
curl -sSfLO $base/thirdparty/bounce/0.18/bounce-0.18.jar
J48_WEKA_CLASSPATH=weka-stable-3.8.6.jar:bounce-0.18.jar python -m pytest -m weka
```

Known divergences are listed in the test file as expected failures. CI runs all of the above on every pull request.

## Choosing an estimator

- `J48Classifier` is the reference line used for WEKA comparisons. It is pure NumPy.
- `J48FastClassifier` builds the same trees (verified by the test suite) from an encoded representation and uses numba kernels when `numba` is installed. On 100k x 20 numeric rows it trains about 2.5x faster than `J48Classifier` and 5x faster than in 0.3.0; with nominal columns it is 3.5x (5 nominal columns) to 10x (IDS-like data with a 70-value `service` attribute) faster than the strict line. Call `warmup_backend()` once to exclude numba compilation from timings.

See [CHANGELOG.md](CHANGELOG.md) for benchmark details.

## Input conventions

The estimators follow scikit-learn conventions (2D `X`, 1D `y`, explicit errors for sparse input, continuous or missing targets, and all-zero weights), with two J48-specific allowances: missing feature values (`NaN`, `None`, `"?"`; infinity is treated as missing) and non-numeric nominal columns are accepted directly.

## Validation and paper context

The implementation was developed in the context of a paper comparing the Python backend against WEKA-aligned behavior and against a faster internal execution mode.

Public repository documentation is intentionally limited to the information needed to understand and use the software artifact. Detailed internal engineering notes remain outside this repository.

The public documentation set is intentionally minimal and is limited to material relevant for technical reviewers and for users of the module.

See [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md) for the technical structure and validation summary retained in the public version.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the reviewer-facing reproduction guidance.
See [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) for a precise statement of what the public artifact can and cannot reproduce from the paper.

## Data policy

IDS datasets used in the broader research workflow are not redistributed in this repository.

When a validation or reproduction workflow depends on restricted or third-party datasets, users are expected to obtain them from their original sources and place them locally.

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the repository-level data availability statement.

## Citation

Until the associated paper has final bibliographic metadata, cite the software artifact directly and include the exact release tag or commit used in your workflow.

Suggested software citation format:

Javier Maldonado. *j48-python-backend*. GitHub repository. Versioned software artifact release. 2026. Available at: https://github.com/javiermaldonadoc/j48-python-backend.

## Artifact version

The public artifact is intended to be consumed through tagged releases.

For manuscript alignment, cite the exact tag and commit used by the paper instead of the moving default branch.

Releases after the paper snapshot (`0.2.0` and later) contain correctness fixes, stricter scikit-learn input validation and WEKA parity tests; on ordinary inputs they build the same trees as `0.1.x`. See [CHANGELOG.md](CHANGELOG.md) for the exact scope of each change.

## Status

This repository serves as the release-oriented software artifact associated with the paper. Its public-facing documentation is intentionally compact and focused on technical review and practical reuse.