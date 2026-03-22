# FAST-014 Design

## Goal

Reduce `fit` time in the metadata-aware nominal training path of `j48.fast`
without changing split semantics, pruning behavior, or downstream tree export.

This item is intentionally narrower than a general nominal-training rewrite.
The focus is the path that matters for our `fast-vs-strict` and
`fast-vs-WEKA` batteries:

- `J48FastClassifier`
- `engine.prepare_fit_bundle(...)`
- encoded nominal features passed to `core.py` as numeric matrices plus
  explicit `nominal_features` and `nominal_value_domains`

## Why FAST-014 is still worth doing

Recent accepted work already improved:

- numeric split search
- pruning / subtree raising residuals
- predict-time preparation
- repeated-evaluation throughput

The remaining room for a classifier-side training improvement is most likely in
the nominal branch of `core.py`, not in another wrapper-level optimization.

The strongest current signals are:

- the generic object-heavy benchmark still shows high `fit_prepare_share` on
  `Adult`, `CarEvaluation`, `Mushroom`, `Nursery`, and `CreditApproval`
- the metadata-aware timing matrix keeps nominal-heavy datasets as the most
  interesting place to look for further `fit` gains
- `FAST-014 / attempt A` stayed exact but did not produce a reliable net win,
  which means the next design must be narrower and more selective

## What attempt A taught us

The rejected `FAST-014 / attempt A` tried dense nominal branch counting more
aggressively. It preserved fidelity but regressed overall in the focused
metadata-aware reruns.

The likely reasons are:

- dense counting was applied too broadly
- setup cost offset the savings on smaller domains / smaller nodes
- the implementation still paid for work that was not reused by both the
  multiway and binary nominal split paths

The next attempt should therefore:

- stay on the encoded fast path only
- reuse branch statistics across both nominal candidate functions
- avoid activating on nodes where setup cost is unlikely to amortize

## Proposed design: attempt B

### Scope

Add a metadata-aware encoded nominal fast path inside `core.py` for

- `_find_best_nominal_split_candidate(...)`
- `_find_best_binary_nominal_split_candidate(...)`

but only when the feature values are already encoded as finite integral codes
coming from the `j48.fast` fit bundle.

### Non-goals

Do not change:

- gain ratio definition
- missing-value semantics
- domain ordering
- binary-vs-multiway choice
- pruning / subtree raising
- export labels
- strict baseline behavior

The change should only alter how branch weights and class counts are computed.

## Planned implementation

### 1. Add a shared helper for encoded nominal stats

Introduce a helper in `core.py`, conceptually:

- `_prepare_encoded_nominal_branch_stats(...)`

Inputs:

- `x_feat`
- `y_sub`
- `weights`
- `feat`

Outputs:

- `known_mask`
- `known_weight`
- `code_values` as dense integer codes
- `domain_codes` in the exact order implied by `_nominal_domain_values(...)`
- `branch_weights`
- `branch_class_counts`
- `feat_counts`

### 2. Fast-path gate

Use the encoded helper only when all of these hold:

- feature is nominal
- array dtype is numeric, not `object`
- non-missing values are integral codes
- domain size is small-to-moderate
- node has enough known rows for setup cost to amortize

Initial conservative gate:

- `len(domain_codes) >= 3`
- `known_count >= 64`

Everything else falls back to the current exact baseline path.

### 3. Multiway nominal split

Replace per-value repeated mask creation with:

- one branch-weight vector
- one branch-by-class count matrix

Then compute:

- child entropy
- branch probabilities
- default child

from those precomputed arrays while preserving domain order.

### 4. Binary nominal split

Reuse the same branch-weight vector and branch-by-class count matrix.

For each candidate value:

- `pos_counts` is one row
- `neg_counts` is `feat_counts - pos_counts`

This avoids rebuilding a boolean mask and a `bincount` per candidate value.

### 5. Safe fallback

If the helper cannot prove the encoded assumptions safely, immediately fall
back to the current implementation.

## Why this design is different from attempt A

Attempt A was effectively "dense counting as a dataset-level idea."

Attempt B is narrower:

- only for the encoded metadata-aware path
- only when the node is large enough
- only when domain codes are already in the exact representation `fast` uses
- one shared stats build reused by both nominal candidate functions

That gives it a better chance of paying its setup cost.

## Expected benefits

Best-case:

- lower `fit` on `Adult`, `Nursery`, `Mushroom`, `CarEvaluation`,
  `CreditApproval`
- especially where nominal branch evaluation still does repeated Python-level
  work today

Likely:

- modest but real `fit` wins on metadata-aware nominal-heavy datasets
- little or no change on IDS numeric-heavy datasets

## Main risks

1. Setup overhead may still dominate on small nodes.
2. Domain/order handling could drift if the helper does not preserve
   `_nominal_domain_values(...)` exactly.
3. Export and comparison harnesses depend on stable nominal labels, so the
   helper must operate on encoded codes internally but continue to surface the
   original domain values externally.

## Validation plan

### Phase 1: focused safety

- `python -m unittest tests.test_c45_j48_strict -q`
- focused metadata-aware timing on:
  - `Adult`
  - `Nursery`
  - `Mushroom`
  - `CarEvaluation`
  - `CreditApproval`
- focused `fast-vs-strict` reruns on the same datasets

### Phase 2: broader spot checks

- `fast-vs-strict Layer B/C`
- `fast-vs-WEKA` spot checks on:
  - `Adult`
  - `Nursery`
  - `Mushroom`
  - `CarEvaluation`
  - `CreditApproval`

## Acceptance criteria

Accept only if all of these hold:

- exact prediction agreement against `strict` on the focused reruns
- stable tree statistics on the focused reruns
- at least one current nominal-heavy timing weakness narrows materially
- no broad regression across the other focused nominal-heavy datasets

Practical threshold:

- enough `fit` improvement to be clearly above timing noise on the
  metadata-aware harness
- and no obvious regression bigger than that win elsewhere

## Rejection criteria

Reject if:

- fidelity drifts at all in the focused `fast-vs-strict` reruns
- the gain is only marginal and inconsistent
- or setup cost makes the branch slower on most focused datasets

## Decision note

`FAST-014` is still worth trying, but only as this narrow encoded-nominal
training optimization. We should not reopen broad dense-counting heuristics or
dataset-wide nominal rewrites without a stronger hotspot signal.
