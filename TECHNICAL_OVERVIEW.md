# Technical Overview

## Components

The public repository keeps the software surface small.

- `j48/core.py`: core J48-oriented tree implementation.
- `j48/engine.py`: backend selection and input-preparation utilities.
- `j48/sklearn_api.py`: scikit-learn-compatible estimator wrappers.
- `j48/acceptance_analysis.py`: auxiliary analysis helpers used in validation post-processing.

## Public estimators

- `J48Classifier`: strict estimator intended to preserve the semantics-oriented baseline.
- `J48FastClassifier`: faster estimator that keeps the same public interface while using an encoded backend.

Both estimators expose a scikit-learn-style API with `fit`, `predict`, `predict_proba`, and tree-inspection helpers.

## Design intent

The repository separates two concerns:

- a semantics-oriented implementation used as the reference software baseline,
- a faster backend used to evaluate engineering improvements without changing the public estimator interface.

This split keeps the public API stable while allowing backend-specific work behind the interface boundary.

## Validation summary

The software was validated in the research workspace through differential comparisons against WEKA-oriented behavior and through comparisons between the strict and fast variants.

For the public repository, the important takeaway is the validation shape rather than the full internal campaign history:

- the strict line serves as the semantic reference implementation,
- the fast line is evaluated against that reference implementation,
- detailed intermediate notes, backlogs, and local run logs are intentionally excluded from this repository.

## Repository policy

This public-facing repository keeps only documentation that is directly useful for software users and reviewers.

Excluded from the public tree:

- internal backlogs,
- engineering checklists,
- freeze notes,
- local-path experiment summaries,
- planning documents that do not improve external use of the software.