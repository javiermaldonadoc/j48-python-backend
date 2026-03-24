# Reproducibility

This repository is intended to serve as the public software artifact associated with the paper.

The current paper-facing artifact snapshot is version `0.1.0`.

## What this repository supports directly

This repository supports direct inspection and reuse of:

- the J48-oriented Python implementation,
- the scikit-learn-compatible estimator interface,
- the distinction between the strict reference line and the faster backend line.

## What is not bundled here

This repository does not bundle:

- IDS datasets,
- local run outputs,
- internal campaign logs,
- private workspace notes used during development.

## Reproduction model

The intended reproduction model is:

1. obtain the required datasets from their original sources,
2. prepare them locally in the formats expected by the evaluation scripts,
3. run the released code for artifact version `0.1.0` and the paper-facing evaluation workflow from a clean environment,
4. compare the resulting outputs with the claims summarized in the paper.

## Public-facing evidence scope

For external readers and reviewers, the main role of this repository is to make the released implementation inspectable and reusable.

The full internal engineering history is intentionally excluded. The public artifact therefore emphasizes:

- released code,
- stable interface,
- concise technical documentation,
- explicit data-policy constraints.

## Reviewer guidance

For paper review, this repository should be read together with:

- the paper itself,
- the repository README,
- the paper reproduction guide,
- the technical overview,
- the data-availability statement.

If a submission version requires additional archival evidence, it is preferable to add a single curated appendix-style artifact note rather than exposing internal development logs.

See [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md) for a more explicit description of which parts of the paper can be reproduced from the public artifact alone and which still depend on the larger research workspace.