from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, column_or_1d

from .core import C45TreeClassifier, warmup_numba_numeric_kernel
from .engine import J48EngineSpec, build_engine


def _input_shape(X: Any) -> tuple[int, ...]:
    shape = getattr(X, "shape", None)
    if shape is None:
        shape = np.asarray(X, dtype=object).shape
    return tuple(int(v) for v in shape)


try:
    from pandas.api.types import infer_dtype as _infer_dtype
except Exception:  # pragma: no cover - pandas is optional
    _infer_dtype = None

# `pandas.api.types.infer_dtype` results that cannot contain complex scalars.
_NON_COMPLEX_INFERRED = {
    "string", "bytes", "floating", "integer", "mixed-integer-float", "boolean",
    "empty", "decimal", "categorical", "datetime", "datetime64", "date", "time",
    "timedelta", "timedelta64", "period", "interval",
}


def _object_column_has_complex(values: np.ndarray) -> bool:
    if _infer_dtype is not None:
        inferred = _infer_dtype(values, skipna=True)
        if inferred == "complex":
            return True
        if inferred in _NON_COMPLEX_INFERRED:
            return False
    # Mixed or unknown content (or no pandas): inspect the values.
    return any(isinstance(v, (complex, np.complexfloating)) for v in values.tolist())


def _is_complex_input(X: Any) -> bool:
    """True for complex dtypes and for object data holding complex scalars."""
    dtypes = getattr(X, "dtypes", None)
    if dtypes is not None:
        for j, dtype in enumerate(list(dtypes)):
            kind = getattr(dtype, "kind", "")
            if kind == "c":
                return True
            if kind == "O" and _object_column_has_complex(np.asarray(X.iloc[:, j], dtype=object)):
                return True
        return False
    dtype = getattr(X, "dtype", None)
    arr = None
    if dtype is None:
        arr = np.asarray(X)
        dtype = arr.dtype
    kind = np.dtype(dtype).kind
    if kind == "c":
        return True
    if kind == "O":
        arr = np.asarray(X) if arr is None else arr
        return any(_object_column_has_complex(arr[:, j]) for j in range(arr.shape[1]))
    return False


class J48Classifier(ClassifierMixin, BaseEstimator):
    """
    scikit-learn-compatible wrapper for the strict J48 implementation.

    This class exposes public parameters aligned with J48 semantics and
    delegates the actual training work to the exact `C45TreeClassifier`
    engine.
    """

    def __init__(
        self,
        confidence_factor: float = 0.25,
        min_num_obj: int = 2,
        unpruned: bool = False,
        reduced_error_pruning: bool = False,
        num_folds: int = 3,
        collapse_tree: bool = True,
        subtree_raising: bool = True,
        binary_splits: bool = False,
        use_laplace: bool = False,
        use_mdl_correction: bool = True,
        use_gain_prefilter: bool = True,
        fractional_missing: bool = True,
        make_split_point_actual_value: bool = True,
        max_thresholds: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_gain_ratio: float = 1e-6,
        nominal_features: Optional[list[int]] = None,
        auto_detect_nominal: bool = False,
        nominal_value_domains: Optional[dict[Any, list[Any]]] = None,
        feature_names: Optional[list[str]] = None,
        backend: str = "numpy",
        fidelity: str = "strict",
        random_state: Optional[int] = None,
        cleanup: bool = True,
    ) -> None:
        self.confidence_factor = confidence_factor
        self.min_num_obj = min_num_obj
        self.unpruned = unpruned
        self.reduced_error_pruning = reduced_error_pruning
        self.num_folds = num_folds
        self.collapse_tree = collapse_tree
        self.subtree_raising = subtree_raising
        self.binary_splits = binary_splits
        self.use_laplace = use_laplace
        self.use_mdl_correction = use_mdl_correction
        self.use_gain_prefilter = use_gain_prefilter
        self.fractional_missing = fractional_missing
        self.make_split_point_actual_value = make_split_point_actual_value
        self.max_thresholds = max_thresholds
        self.max_depth = max_depth
        self.min_gain_ratio = min_gain_ratio
        self.nominal_features = nominal_features
        self.auto_detect_nominal = auto_detect_nominal
        self.nominal_value_domains = nominal_value_domains
        self.feature_names = feature_names
        self.backend = backend
        self.fidelity = fidelity
        self.random_state = random_state
        self.cleanup = cleanup

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "J48Classifier":
        self._validate_params()
        self._validate_X(X, reset=True)
        y = self._validate_y(y)
        columns = getattr(X, "columns", None)
        # Column names seen in fit; predict() requires the same names in the
        # same order when it also receives a DataFrame.
        self._fit_column_names_ = None if columns is None else [str(c) for c in columns]

        self.engine_ = build_engine(backend=self.backend, fidelity=self.fidelity)
        fit_bundle = self.engine_.prepare_fit_bundle(
            X,
            y,
            feature_names=self.feature_names,
            nominal_features=self.nominal_features,
            auto_detect_nominal=self.auto_detect_nominal,
            nominal_value_domains=self.nominal_value_domains,
        )
        return self.fit_prepared_bundle(fit_bundle, sample_weight=sample_weight)

    def _validate_params(self) -> None:
        if self.unpruned and self.reduced_error_pruning:
            raise ValueError("unpruned=True is incompatible with reduced_error_pruning=True")
        if int(self.min_num_obj) < 1:
            raise ValueError("min_num_obj must be >= 1")
        if int(self.num_folds) < 2 and self.reduced_error_pruning:
            raise ValueError("num_folds must be >= 2 when reduced_error_pruning=True")

    def _validate_X(self, X: Any, *, reset: bool) -> None:
        """
        scikit-learn-style checks on the raw feature matrix.

        Values are not converted or checked for finiteness here: missing
        values (NaN, None, "?") and non-numeric nominal columns are valid J48
        input and are handled by the engine.
        """
        name = type(self).__name__
        if sparse.issparse(X):
            raise TypeError(
                f"{name} does not support sparse input. "
                "Convert it to a dense array first, e.g. with X.toarray()."
            )
        shape = _input_shape(X)
        if len(shape) != 2:
            raise ValueError(
                f"Expected a 2D array, got a {len(shape)}D array instead. "
                "Reshape your data using array.reshape(-1, 1) if it has a single "
                "feature or array.reshape(1, -1) if it contains a single sample."
            )
        if reset and shape[0] < 1:
            raise ValueError(
                f"Found array with 0 sample(s) (shape={shape}) while a minimum of 1 is required by {name}."
            )
        if shape[1] < 1:
            raise ValueError(
                f"Found array with 0 feature(s) (shape={shape}) while a minimum of 1 is required by {name}."
            )
        if _is_complex_input(X):
            raise ValueError("Complex data not supported.")
        if not reset and shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {shape[1]} features, but {name} is expecting "
                f"{self.n_features_in_} features as input."
            )
        fitted_columns = getattr(self, "_fit_column_names_", None)
        columns = getattr(X, "columns", None)
        if not reset and fitted_columns is not None and columns is not None:
            names = [str(c) for c in columns]
            if names != fitted_columns:
                mismatched = [
                    f"position {i}: fitted {expected!r}, got {got!r}"
                    for i, (expected, got) in enumerate(zip(fitted_columns, names))
                    if expected != got
                ]
                raise ValueError(
                    "The feature names should match those that were passed during fit "
                    "(same names in the same order; columns are used by position). "
                    + "; ".join(mismatched[:5])
                )

    def _validate_y(self, y: Any) -> np.ndarray:
        if y is None:
            raise ValueError(
                f"{type(self).__name__} requires y to be passed, but the target y is None."
            )
        y = column_or_1d(y, warn=True)
        if y.dtype.kind == "f" and not np.all(np.isfinite(y)):
            raise ValueError("Input y contains NaN or infinity; drop those rows before fitting.")
        check_classification_targets(y)
        return y

    def _ensure_engine(self) -> None:
        desired_spec = J48EngineSpec(backend=str(self.backend), fidelity=str(self.fidelity))
        current = getattr(self, "engine_", None)
        if current is None or getattr(current, "spec", None) != desired_spec:
            self.engine_ = build_engine(backend=self.backend, fidelity=self.fidelity)

    def fit_prepared_bundle(
        self,
        fit_bundle: dict[str, Any],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "J48Classifier":
        self._validate_params()
        self._ensure_engine()
        feature_names = fit_bundle["feature_names"]
        X_arr = fit_bundle["X"]
        y_arr = fit_bundle["y"]

        self.core_estimator_ = C45TreeClassifier(
            min_samples_split=2,
            min_samples_leaf=int(self.min_num_obj),
            max_depth=self.max_depth,
            min_gain_ratio=float(self.min_gain_ratio),
            use_gain_prefilter=bool(self.use_gain_prefilter),
            use_mdl_correction=bool(self.use_mdl_correction),
            enable_pruning=not bool(self.unpruned) and not bool(self.reduced_error_pruning),
            reduced_error_pruning=bool(self.reduced_error_pruning),
            num_folds=int(self.num_folds),
            confidence_factor=float(self.confidence_factor),
            collapse_tree=bool(self.collapse_tree),
            enable_subtree_raising=(
                bool(self.subtree_raising)
                and not bool(self.unpruned)
                and not bool(self.reduced_error_pruning)
            ),
            enable_fractional_missing=bool(self.fractional_missing),
            make_split_point_actual_value=bool(self.make_split_point_actual_value),
            use_laplace=bool(self.use_laplace),
            nominal_features=fit_bundle["nominal_features"],
            binary_splits=bool(self.binary_splits),
            auto_detect_nominal=bool(fit_bundle.get("auto_detect_nominal", False)),
            nominal_value_domains=fit_bundle.get("nominal_value_domains"),
            feature_names=feature_names,
            max_thresholds=self.max_thresholds,
            random_state=self.random_state,
            cleanup=bool(self.cleanup),
            use_numba_numeric_kernel=(self.backend == "numpy_fast"),
        )
        self.core_estimator_.fit(X_arr, y_arr, sample_weight=sample_weight)

        self.classes_ = self.core_estimator_.classes_
        self.n_features_in_ = int(self.core_estimator_.n_features_)
        if feature_names is not None and len(feature_names) == self.n_features_in_:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.backend_name_ = self.engine_.name
        return self

    def _validate_prepared_matrix(self, X: Any) -> np.ndarray:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2:
            raise ValueError("J48Classifier expects a 2D prepared feature matrix")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} prepared features, got {X_arr.shape[1]}"
            )
        return X_arr

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "core_estimator_")
        self._validate_X(X, reset=False)
        X_arr = self.engine_.prepare_predict_data(X, expected_features=self.n_features_in_)
        if self.engine_.can_use_fast_hard_predict(X_arr, self.core_estimator_):
            return self.engine_.hard_predict(X_arr, self.core_estimator_)
        return self.core_estimator_.predict(X_arr)

    def predict_prepared(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "core_estimator_")
        X_arr = self._validate_prepared_matrix(X)
        if self.engine_.can_use_fast_hard_predict(X_arr, self.core_estimator_):
            return self.engine_.hard_predict(X_arr, self.core_estimator_)
        return self.core_estimator_.predict(X_arr)

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "core_estimator_")
        self._validate_X(X, reset=False)
        X_arr = self.engine_.prepare_predict_data(X, expected_features=self.n_features_in_)
        if self.engine_.can_use_fast_predict_proba(X_arr, self.core_estimator_):
            return self.engine_.predict_proba_fast(X_arr, self.core_estimator_)
        return self.core_estimator_.predict_proba(X_arr)

    def predict_proba_prepared(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "core_estimator_")
        X_arr = self._validate_prepared_matrix(X)
        if self.engine_.can_use_fast_predict_proba(X_arr, self.core_estimator_):
            return self.engine_.predict_proba_fast(X_arr, self.core_estimator_)
        return self.core_estimator_.predict_proba(X_arr)

    def export_tree(self) -> dict[str, Any]:
        check_is_fitted(self, "core_estimator_")
        return self.engine_.postprocess_export_tree(self.core_estimator_.export_tree())

    def get_tree_stats(self) -> dict[str, Any]:
        check_is_fitted(self, "core_estimator_")
        return self.core_estimator_.get_tree_stats()

    def iter_tree_nodes(self) -> list[dict[str, Any]]:
        check_is_fitted(self, "core_estimator_")
        exported = self.export_tree().get("root")
        if exported is None:
            return []
        rows: list[dict[str, Any]] = []
        stack = [exported]
        while stack:
            current = stack.pop()
            rows.append({k: v for k, v in current.items() if k != "children"})
            for child in reversed(current.get("children", [])):
                stack.append(child["child"])
        return rows

    def get_core_estimator(self) -> C45TreeClassifier:
        check_is_fitted(self, "core_estimator_")
        return self.core_estimator_

    def _more_tags(self) -> dict[str, Any]:
        # Tag API for scikit-learn < 1.6.
        return {
            "allow_nan": True,
            "requires_y": True,
            "X_types": ["2darray", "string"],
        }

    def __sklearn_tags__(self):
        # Tag API for scikit-learn >= 1.6.
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.target_tags.required = True
        return tags


class J48FastClassifier(J48Classifier):
    """
    Performance-oriented variant built on an encoded internal representation.

    It keeps the same public interface as `J48Classifier`, but uses
    `backend='numpy_fast'` by default to reduce the cost of nominal columns
    and `dtype=object` conversions.
    """

    def __init__(
        self,
        confidence_factor: float = 0.25,
        min_num_obj: int = 2,
        unpruned: bool = False,
        reduced_error_pruning: bool = False,
        num_folds: int = 3,
        collapse_tree: bool = True,
        subtree_raising: bool = True,
        binary_splits: bool = False,
        use_laplace: bool = False,
        use_mdl_correction: bool = True,
        use_gain_prefilter: bool = True,
        fractional_missing: bool = True,
        make_split_point_actual_value: bool = True,
        max_thresholds: Optional[int] = None,
        max_depth: Optional[int] = None,
        min_gain_ratio: float = 1e-6,
        nominal_features: Optional[list[int]] = None,
        auto_detect_nominal: bool = False,
        nominal_value_domains: Optional[dict[Any, list[Any]]] = None,
        feature_names: Optional[list[str]] = None,
        fidelity: str = "equivalent",
        random_state: Optional[int] = None,
        cleanup: bool = True,
    ) -> None:
        super().__init__(
            confidence_factor=confidence_factor,
            min_num_obj=min_num_obj,
            unpruned=unpruned,
            reduced_error_pruning=reduced_error_pruning,
            num_folds=num_folds,
            collapse_tree=collapse_tree,
            subtree_raising=subtree_raising,
            binary_splits=binary_splits,
            use_laplace=use_laplace,
            use_mdl_correction=use_mdl_correction,
            use_gain_prefilter=use_gain_prefilter,
            fractional_missing=fractional_missing,
            make_split_point_actual_value=make_split_point_actual_value,
            max_thresholds=max_thresholds,
            max_depth=max_depth,
            min_gain_ratio=min_gain_ratio,
            nominal_features=nominal_features,
            auto_detect_nominal=auto_detect_nominal,
            nominal_value_domains=nominal_value_domains,
            feature_names=feature_names,
            backend="numpy_fast",
            fidelity=fidelity,
            random_state=random_state,
            cleanup=cleanup,
        )

    def warmup_backend(self) -> None:
        warmup_numba_numeric_kernel()
