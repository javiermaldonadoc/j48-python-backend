# J48 Fast EA Profile Tuning

This note documents the profile-tuning round run for `PYJ48_FAST` inside the EA
flow.

## Scope

The goal of this run was not to change tree-core semantics, but to identify a
better EA-facing parameter profile than the strict J48-oriented default.

The run used:

- datasets: `NSL-KDD`, `CIC-IDS2017`
- seeds: `11223`, `12345`, `13579`, `22334`, `23456`
- runner:
  [run_j48_fast_ea_tuning_round2.sh](/home/javier/VisualStudio/IDEA/scripts/analysis/run_j48_fast_ea_tuning_round2.sh)
- aggregate artifact:
  [aggregate.csv](/home/javier/VisualStudio/IDEA/.local_runs/j48_fast_ea_profiles/j48_fast_ea_profiles_20260316_095809/aggregate.csv)

Profiles evaluated:

- `p3off_leaf1`
- `p3off_leaf1_cf015`
- `p3off_leaf1_cf025`
- `p3off_leaf1_cf035`
- `p3off_leaf1_nocollapse`
- `p3off_leaf1_unpruned`

## Median Results

| Dataset | Profile | Elapsed (s) | Acc | AvRec | AvPr | F1 | MCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NSL-KDD` | `p3off_leaf1` | `66.92` | `94.48` | `89.99` | `82.49` | `85.58` | `92.29` |
| `NSL-KDD` | `p3off_leaf1_cf015` | `98.36` | `94.54` | `88.11` | `80.23` | `83.57` | `92.22` |
| `NSL-KDD` | `p3off_leaf1_cf025` | `95.23` | `94.48` | `89.99` | `82.49` | `85.58` | `92.29` |
| `NSL-KDD` | `p3off_leaf1_cf035` | `94.58` | `94.61` | `91.43` | `80.71` | `84.53` | `92.36` |
| `NSL-KDD` | `p3off_leaf1_nocollapse` | `93.30` | `94.48` | `89.99` | `82.49` | `85.58` | `92.29` |
| `NSL-KDD` | `p3off_leaf1_unpruned` | `94.00` | `94.93` | `91.10` | `81.28` | `84.88` | `92.80` |
| `CIC-IDS2017` | `p3off_leaf1` | `34.21` | `91.47` | `86.33` | `82.60` | `82.99` | `90.76` |
| `CIC-IDS2017` | `p3off_leaf1_cf015` | `34.78` | `91.20` | `86.22` | `82.16` | `81.33` | `90.48` |
| `CIC-IDS2017` | `p3off_leaf1_cf025` | `35.17` | `91.47` | `86.33` | `82.60` | `82.99` | `90.76` |
| `CIC-IDS2017` | `p3off_leaf1_cf035` | `36.16` | `92.37` | `85.82` | `82.49` | `82.96` | `91.74` |
| `CIC-IDS2017` | `p3off_leaf1_nocollapse` | `34.68` | `91.47` | `86.33` | `82.60` | `82.99` | `90.76` |
| `CIC-IDS2017` | `p3off_leaf1_unpruned` | `33.88` | `92.12` | `85.68` | `82.36` | `82.81` | `91.47` |

## Interpretation

- `p3off_leaf1` is the best general-purpose EA preset.
  - It is clearly the best time/quality tradeoff on `NSL-KDD`.
  - It remains very strong on `CIC-IDS2017`.
- `p3off_leaf1_cf025` and `p3off_leaf1_nocollapse` are redundant.
  - They reproduce the same medians as `p3off_leaf1` but are slower.
- `p3off_leaf1_cf015` should be rejected.
  - It is slower and degrades `F1` and `AvPr`, especially on `NSL-KDD`.
- `p3off_leaf1_cf035` and `p3off_leaf1_unpruned` are specialized options.
  - On `CIC-IDS2017`, both improve `acc` and `mcc`.
  - On `NSL-KDD`, they improve selected metrics but cost much more time and
    hurt `F1`/`AvPr`.

## Recommendation

- Default EA preset candidate:
  - `p3off_leaf1`
- Optional dataset-specific candidates:
  - `p3off_leaf1_unpruned` for cases where `acc`/`mcc` matter more than runtime
  - `p3off_leaf1_cf035` only as a targeted alternative, not as a default

The main practical conclusion is that `PYJ48_FAST` benefits from an EA-specific
profile and should not be judged only through the strict J48-oriented default.
