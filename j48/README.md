# J48 Package Architecture

This package is the new home for the WEKA-faithful J48 line.

## Layers

- `core.py`: exact tree semantics under alignment with WEKA J48.
- `engine.py`: backend selection and input normalization. The initial backend is `numpy-strict`.
- `sklearn_api.py`: public `J48Classifier` wrapper compatible with the scikit-learn estimator API.

## Design goals

- Keep `J48` fidelity work isolated from the stable project `c45.py`.
- Preserve backwards compatibility through `EvLib/c45_j48.py`.
- Allow future exact performance backends (`numba`, `cython`, `gpu`) without changing the public estimator API.
- Make the J48 line reusable outside the current EA pipeline.

## Phase 1 focus

- Cover the main public options of `WEKA J48` before optimizing.
- Validate behavior through the differential harness in `scripts/analysis/`.
- Keep `strict` semantics separated from any future `fast` or `plus` variants.

## Current working documents

- [J48_FAST_CURRENT_CHECKLIST.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_CURRENT_CHECKLIST.md):
  current engineering checkpoint, accepted optimizations, and evidence still
  pending before the next large campaign
- [J48_FAST_OPTIMIZATION_BACKLOG.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_OPTIMIZATION_BACKLOG.md):
  active optimization backlog and attempt history
- [J48_FAST_FAST014_DESIGN.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_FAST014_DESIGN.md):
  scoped design for the next nominal-training optimization attempt
- [J48_FAST_VALIDATION_RESULTS.md](/home/javier/VisualStudio/IDEA/EvLib/j48/J48_FAST_VALIDATION_RESULTS.md):
  latest consolidated validation artifact for `j48.fast`
