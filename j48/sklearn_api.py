from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .core import C45TreeClassifier, warmup_numba_numeric_kernel
from .engine import J48EngineSpec, build_engine


class J48Classifier(ClassifierMixin, BaseEstimator):
    """
    Wrapper scikit-learn-compatible para la implementación J48 estricta.

    Esta clase expone parámetros públicos orientados a la semántica de J48 y
    delega el entrenamiento real al motor exacto `C45TreeClassifier`.
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
        if self.unpruned and self.reduced_error_pruning:
            raise ValueError("unpruned=True is incompatible with reduced_error_pruning=True")
        if int(self.min_num_obj) < 1:
            raise ValueError("min_num_obj must be >= 1")
        if int(self.num_folds) < 2 and self.reduced_error_pruning:
            raise ValueError("num_folds must be >= 2 when reduced_error_pruning=True")

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
        if self.unpruned and self.reduced_error_pruning:
            raise ValueError("unpruned=True is incompatible with reduced_error_pruning=True")
        if int(self.min_num_obj) < 1:
            raise ValueError("min_num_obj must be >= 1")
        if int(self.num_folds) < 2 and self.reduced_error_pruning:
            raise ValueError("num_folds must be >= 2 when reduced_error_pruning=True")

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
        return {
            "allow_nan": True,
            "requires_y": True,
            "X_types": ["2darray", "string"],
        }


class J48FastClassifier(J48Classifier):
    """
    Variante orientada a rendimiento sobre una representación interna codificada.

    Mantiene la misma interfaz pública de `J48Classifier`, pero usa
    `backend='numpy_fast'` por defecto para reducir el costo de columnas
    nominales y conversiones de `dtype=object`.
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
