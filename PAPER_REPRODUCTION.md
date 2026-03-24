# Paper Reproduction Guide

This document explains what can and cannot be reproduced from the public software artifact associated with the paper.

Artifact version covered here: `0.1.0`.

## Scope

The paper combines two layers:

- a released software artifact: this repository,
- a larger research workspace used to run paired WEKA-vs-Python campaigns, generate intermediate summaries, and rebuild manuscript-facing tables and figures.

This guide is intentionally explicit about that separation.

## What can be reproduced directly from this repository

This public repository supports direct reproduction of the following software-artifact aspects:

- inspection of the strict J48-oriented implementation,
- inspection of the faster backend used in the study,
- reuse of the public estimator API,
- local experiments that compare the strict and fast wrappers on user-provided tabular datasets,
- small-scale workflow checks built around ordinary scikit-learn control flow.

In practical terms, a reader can install the released code, load local tabular datasets, and verify that:

- the estimators fit and predict correctly,
- the strict and fast wrappers expose the documented interface,
- the fast backend can be evaluated as a drop-in alternative under user-defined conditions.

## What cannot be reproduced from this repository alone

This repository does not by itself reproduce the full paper experiment stack.

The following parts of the manuscript depend on assets that are not bundled in the public artifact:

- the exact paired WEKA-vs-Python differential harnesses used in the research workspace,
- the fixed paper manifests and orchestrated batch runners used for the strict, fast, timing, mini-ablation, and pipeline campaigns,
- the manuscript-facing intermediate CSV summaries and figure-generation workflow,
- the local WEKA setup used to anchor the comparison to a fixed `weka.jar` artifact,
- restricted or non-redistributed IDS data releases.

Accordingly, this repository should not be read as a turnkey reproduction package for every table and figure in the paper.

## What is partially reproducible with additional local setup

Parts of the paper can be approached with additional local work by an external reader who is willing to assemble the missing comparison environment.

That setup would require, at minimum:

- obtaining the public datasets used in the paper from their original sources,
- preparing local copies of those datasets in the expected tabular form,
- obtaining a compatible local WEKA installation,
- recreating matched train/test partitions and fixed seed control,
- writing or recreating a paired comparison harness around the released Python artifact and the local WEKA baseline.

Under that broader setup, an external reader can attempt to reproduce the main comparison logic of the study, but not the exact internal research workspace used to generate the manuscript.

## Recommended reading order for reviewers

For technical review, this repository should be read together with:

- `README.md` for artifact scope and public API,
- `TECHNICAL_OVERVIEW.md` for the strict-vs-fast architecture,
- `REPRODUCIBILITY.md` for the artifact-level reproducibility model,
- `DATA_AVAILABILITY.md` for dataset restrictions and sourcing expectations,
- the paper itself for the full experimental design and interpretation.

## Practical interpretation of the release

The released artifact is sufficient for:

- code inspection,
- software reuse,
- API-level validation,
- local strict-vs-fast experiments,
- technical review of the implementation under study.

The released artifact is not sufficient, by itself, for:

- exact regeneration of every manuscript table and figure,
- exact rerun of the original paired WEKA comparison campaigns,
- exact rerun of all IDS experiments reported in the paper.

## Reproducibility claim supported by this release

The reproducibility claim supported by the public artifact is the following:

The released version makes the implementation inspectable and reusable, exposes the exact paper-facing software snapshot, and supports reproducibility of the reported comparison logic only when combined with fixed seeds, matched data preparation, and an external comparison environment consistent with the paper.

It does not claim turnkey reproduction of the full research workspace on arbitrary hosts.