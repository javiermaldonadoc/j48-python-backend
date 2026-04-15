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
- `all`: installs all optional dependencies

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

## Status

This repository serves as the release-oriented software artifact associated with the paper. Its public-facing documentation is intentionally compact and focused on technical review and practical reuse.