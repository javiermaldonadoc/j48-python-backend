# J48 Fast Validation Results

This artifact consolidates the post-fix validation runs used to support the `j48.fast` engineering results.

## Source Runs

- `fast-vs-strict Layer A`: `/home/javier/VisualStudio/IDEA/.local_runs/weka_fast_batches/weka_fast_full_post_fast011c_20260321_093450/fast_vs_strict_synth/j48_fast_vs_strict_synth_20260321_093451/summary.csv`
- `fast-vs-strict Layer B/C`: `/home/javier/VisualStudio/IDEA/.local_runs/weka_fast_batches/weka_fast_full_post_fast011c_20260321_093450/fast_vs_strict/j48_fast_vs_strict_20260321_093555/summary.csv`
- `fast-vs-WEKA Layer B/C`: `/home/javier/VisualStudio/IDEA/.local_runs/weka_fast_batches/weka_fast_full_post_fast011c_20260321_093450/fast_vs_weka/j48_fast_vs_weka_20260321_093917/summary.csv`
- `timing`: `/home/javier/VisualStudio/IDEA/.local_runs/weka_fast_batches/weka_fast_full_post_fast011c_20260321_093450/timing/j48_fast_timing_20260321_094402/summary.csv`

## Headline Results

- `fast-vs-strict Layer A`: `441/441` exact
- `fast-vs-strict Layer B/C`: `741/741` exact
- `fast-vs-WEKA Layer B/C`: mean match `0.999428`, minimum `0.956522`, exact rows `632/741`
- Timing medians: `fit 4.253x`, `predict 1.157x`, `predict_only 0.631x`, `predict_proba 2.908x`

## Lowest Match By Dataset Against WEKA

| Dataset | Match Mean | Match Min | Exact Rows | Total Mismatches |
| --- | ---: | ---: | ---: | ---: |
| CreditApproval | 0.996101 | 0.956522 | 51/57 | 46 |
| CIC-IDS2017 | 0.997368 | 0.997000 | 7/57 | 150 |
| NSL-KDD | 0.999105 | 0.999000 | 6/57 | 51 |
| Adult | 0.999991 | 0.999570 | 55/57 | 8 |
| BreastCancerDiagnostic | 1.000000 | 1.000000 | 57/57 | 0 |
| BreastCancerOriginal | 1.000000 | 1.000000 | 57/57 | 0 |
| CIRA2020-AttNorm | 1.000000 | 1.000000 | 57/57 | 0 |
| CIRA2020-DoHNDoH | 1.000000 | 1.000000 | 57/57 | 0 |
| CarEvaluation | 1.000000 | 1.000000 | 57/57 | 0 |
| Ionosphere | 1.000000 | 1.000000 | 57/57 | 0 |
| Mushroom | 1.000000 | 1.000000 | 57/57 | 0 |
| Nursery | 1.000000 | 1.000000 | 57/57 | 0 |
| UNSW-NB15 | 1.000000 | 1.000000 | 57/57 | 0 |

## Match By Configuration Against WEKA

| Config | Match Mean | Match Min | Exact Rows |
| --- | ---: | ---: | ---: |
| no_subtree_raising | 0.996656 | 0.956522 | 12/13 |
| unpruned | 0.996656 | 0.956522 | 12/13 |
| no_split_actual | 0.997322 | 0.966184 | 11/13 |
| laplace | 0.997394 | 0.966184 | 11/13 |
| base | 0.997399 | 0.966184 | 12/13 |
| no_collapse | 0.997399 | 0.966184 | 12/13 |
| rep | 0.999692 | 0.997000 | 550/650 |
| no_mdl | 0.999967 | 0.999570 | 12/13 |

## Top Fit Speedups

| Dataset | Fit Speedup Median | Predict Speedup Median | WEKA Match Mean |
| --- | ---: | ---: | ---: |
| Adult | 12.054x | 2.246x | 0.999991 |
| NSL-KDD | 8.107x | 9.004x | 0.999105 |
| Nursery | 7.375x | 1.559x | 1.000000 |
| UNSW-NB15 | 6.568x | 14.389x | 1.000000 |
| CarEvaluation | 6.176x | 0.495x | 1.000000 |

## Interpretation

- `j48.fast` preserves the frozen `strict` baseline exactly on the post-fix validation runs.
- Relative to WEKA, the fast line stays very close overall, with residual drift concentrated in a small subset of dataset/config combinations.
- The performance gain is strongest in training time. Prediction improves overall in median, but `predict_only` is still mixed by dataset.

