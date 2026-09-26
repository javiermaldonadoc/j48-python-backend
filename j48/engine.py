from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core import _weka_max_index
try:
    from numba import njit
    ENGINE_NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None
    ENGINE_NUMBA_AVAILABLE = False
try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is expected in the project env
    pd = None


_COMPILED_NOMINAL_OTHER_SENTINEL = -2147483648


@dataclass(frozen=True)
class J48EngineSpec:
    """
    Describes the execution backend associated with the J48 family.

    `numpy` preserves the current strict-baseline representation.
    `numpy_fast` applies a dense internal encoding for nominal attributes,
    aimed at reducing the cost of `dtype=object` without changing the
    estimator's public API.
    """

    backend: str = "numpy"
    fidelity: str = "strict"
    exact_splits: bool = True
    supports_nominal_multiway: bool = True
    supports_sample_weight: bool = True


class NumpyJ48Engine:
    """
    Exact NumPy-based backend.

    Preserves the semantics and representation of the `strict` baseline.
    """

    def __init__(self, spec: Optional[J48EngineSpec] = None) -> None:
        self.spec = spec or J48EngineSpec()
        if self.spec.backend not in {"numpy", "numpy_fast"}:
            raise ValueError(f"Unsupported J48 backend: {self.spec.backend}")
        if self.spec.fidelity not in {"strict", "equivalent"}:
            raise ValueError(f"Unsupported J48 fidelity mode: {self.spec.fidelity}")
        self._last_prepare_predict_cache_hit = False

    @property
    def name(self) -> str:
        return f"{self.spec.backend}-{self.spec.fidelity}"

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:
                value = str(value)
        return value

    @classmethod
    def _is_missing_scalar(cls, value: Any) -> bool:
        value = cls._to_python_scalar(value)
        if value is None:
            return True
        # pandas missing markers (pd.NA, pd.NaT), detected without importing pandas.
        if type(value).__name__ in ("NAType", "NaTType"):
            return True
        if isinstance(value, str):
            return value.strip() in {"", "?"}
        if isinstance(value, (float, np.floating)):
            return not np.isfinite(value)
        return False

    def infer_feature_names(
        self,
        X: Any,
        feature_names: Optional[list[str]] = None,
    ) -> Optional[list[str]]:
        if feature_names is not None:
            return [str(v) for v in feature_names]
        columns = getattr(X, "columns", None)
        if columns is None:
            return None
        return [str(v) for v in columns]

    def prepare_fit_data(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("J48Classifier expects a 2D feature matrix")
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = np.ravel(y_arr)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        return X_arr, y_arr

    def prepare_fit_bundle(
        self,
        X: Any,
        y: Any,
        *,
        feature_names: Optional[list[str]] = None,
        nominal_features: Optional[list[int]] = None,
        auto_detect_nominal: bool = False,
        nominal_value_domains: Optional[dict[Any, list[Any]]] = None,
    ) -> dict[str, Any]:
        X_arr, y_arr = self.prepare_fit_data(X, y)
        return {
            "X": X_arr,
            "y": y_arr,
            "feature_names": self.infer_feature_names(X, feature_names=feature_names),
            "nominal_features": None if nominal_features is None else [int(v) for v in nominal_features],
            "nominal_value_domains": None if nominal_value_domains is None else dict(nominal_value_domains),
            "auto_detect_nominal": bool(auto_detect_nominal),
        }

    def prepare_predict_data(self, X: Any, expected_features: int) -> np.ndarray:
        self._last_prepare_predict_cache_hit = False
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2:
            raise ValueError("J48Classifier expects a 2D feature matrix")
        if X_arr.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {X_arr.shape[1]}"
            )
        return X_arr

    def postprocess_export_tree(self, exported: dict[str, Any]) -> dict[str, Any]:
        return exported

    def can_use_fast_hard_predict(self, X: np.ndarray, estimator: Any) -> bool:
        return False

    def hard_predict(self, X: np.ndarray, estimator: Any) -> np.ndarray:
        raise NotImplementedError

    def can_use_fast_predict_proba(self, X: np.ndarray, estimator: Any) -> bool:
        return False

    def predict_proba_fast(self, X: np.ndarray, estimator: Any) -> np.ndarray:
        raise NotImplementedError

    def get_last_prepare_predict_cache_hit(self) -> bool:
        """
        Kept for backward compatibility. Since 0.2.0 prepared inputs are no
        longer cached across calls, so this always returns False.
        """
        return bool(self._last_prepare_predict_cache_hit)

    @staticmethod
    def stable_argmax_from_proba(proba: np.ndarray, tol: float = 1e-7) -> np.ndarray:
        if proba.ndim != 2:
            raise ValueError("Expected a 2D probability matrix")
        max_vals = np.max(proba, axis=1, keepdims=True)
        near_max = proba >= (max_vals - float(tol))
        return np.argmax(near_max, axis=1)

    def describe_hard_predict_path(self, X: np.ndarray, estimator: Any) -> dict[str, Any]:
        return {
            "fast_path": False,
            "path": "core_predict",
            "reason": "engine_does_not_support_fast_hard_predict",
        }

    def describe_predict_proba_path(self, X: np.ndarray, estimator: Any) -> dict[str, Any]:
        return {
            "fast_path": False,
            "path": "core_predict_proba",
            "reason": "engine_does_not_support_fast_predict_proba",
        }


if ENGINE_NUMBA_AVAILABLE:
    @njit(cache=True)
    def _predict_terminal_nodes_numba(
        X: np.ndarray,
        split_type: np.ndarray,
        feature_index: np.ndarray,
        threshold: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        default_child: np.ndarray,
        first_child: np.ndarray,
        missing_go_to_left: np.ndarray,
        edge_starts: np.ndarray,
        edge_ends: np.ndarray,
        edge_values: np.ndarray,
        edge_children: np.ndarray,
    ) -> np.ndarray:
        out = np.empty(X.shape[0], dtype=np.int32)
        for row_idx in range(X.shape[0]):
            node = 0
            while split_type[node] != -1:
                feat = feature_index[node]
                value = X[row_idx, feat]
                if split_type[node] == 0:
                    if np.isfinite(value):
                        node = left_child[node] if value <= threshold[node] else right_child[node]
                    else:
                        node = left_child[node] if missing_go_to_left[node] else right_child[node]
                    continue
                chosen = -1
                start = edge_starts[node]
                end = edge_ends[node]
                if np.isfinite(value):
                    target = int(value)
                    for edge_idx in range(start, end):
                        if edge_values[edge_idx] == target:
                            chosen = edge_children[edge_idx]
                            break
                if chosen < 0:
                    chosen = default_child[node]
                if chosen < 0:
                    chosen = first_child[node]
                node = chosen
            out[row_idx] = node
        return out

    @njit(cache=True)
    def _predict_proba_nodes_numba(
        X: np.ndarray,
        split_type: np.ndarray,
        feature_index: np.ndarray,
        threshold: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        default_child: np.ndarray,
        first_child: np.ndarray,
        has_other_branch: np.ndarray,
        missing_go_to_left: np.ndarray,
        left_prob: np.ndarray,
        edge_starts: np.ndarray,
        edge_ends: np.ndarray,
        edge_values: np.ndarray,
        edge_children: np.ndarray,
        edge_probs: np.ndarray,
        leaf_proba: np.ndarray,
        fractional_missing: bool,
    ) -> np.ndarray:
        n_rows = X.shape[0]
        n_classes = leaf_proba.shape[1]
        out = np.zeros((n_rows, n_classes), dtype=np.float64)
        n_nodes = split_type.shape[0]
        for row_idx in range(n_rows):
            if n_nodes == 0:
                continue
            node_stack = np.empty(n_nodes, dtype=np.int32)
            weight_stack = np.empty(n_nodes, dtype=np.float64)
            stack_size = 1
            node_stack[0] = 0
            weight_stack[0] = np.float64(1.0)
            while stack_size > 0:
                stack_size -= 1
                node = node_stack[stack_size]
                weight = weight_stack[stack_size]
                if weight <= 0.0:
                    continue
                if split_type[node] == -1:
                    for class_idx in range(n_classes):
                        out[row_idx, class_idx] += weight * leaf_proba[node, class_idx]
                    continue

                feat = feature_index[node]
                value = X[row_idx, feat]
                if split_type[node] == 0:
                    if np.isfinite(value):
                        child = left_child[node] if value <= threshold[node] else right_child[node]
                        if child >= 0:
                            node_stack[stack_size] = child
                            weight_stack[stack_size] = weight
                            stack_size += 1
                        continue

                    if fractional_missing:
                        # Push right first so the left branch is accumulated
                        # first, in the same order as the strict line.
                        lp = np.float64(left_prob[node])
                        rp = np.float64(1.0) - lp
                        child = right_child[node]
                        if child >= 0 and rp > 0.0:
                            node_stack[stack_size] = child
                            weight_stack[stack_size] = weight * rp
                            stack_size += 1
                        child = left_child[node]
                        if child >= 0 and lp > 0.0:
                            node_stack[stack_size] = child
                            weight_stack[stack_size] = weight * lp
                            stack_size += 1
                        continue

                    child = left_child[node] if missing_go_to_left[node] else right_child[node]
                    if child >= 0:
                        node_stack[stack_size] = child
                        weight_stack[stack_size] = weight
                        stack_size += 1
                    continue

                start = edge_starts[node]
                end = edge_ends[node]
                chosen = -1
                is_finite = np.isfinite(value)
                if is_finite:
                    target = int(value)
                    for edge_idx in range(start, end):
                        if edge_values[edge_idx] == target:
                            chosen = edge_children[edge_idx]
                            break
                if chosen >= 0:
                    node_stack[stack_size] = chosen
                    weight_stack[stack_size] = weight
                    stack_size += 1
                    continue

                if is_finite and has_other_branch[node]:
                    child = default_child[node]
                    if child >= 0:
                        node_stack[stack_size] = child
                        weight_stack[stack_size] = weight
                        stack_size += 1
                    continue

                if fractional_missing:
                    # Reverse push order: branches are accumulated in edge
                    # order, as in the strict line.
                    for edge_idx in range(end - 1, start - 1, -1):
                        child = edge_children[edge_idx]
                        prob = np.float64(edge_probs[edge_idx])
                        if child >= 0 and prob > 0.0:
                            node_stack[stack_size] = child
                            weight_stack[stack_size] = weight * prob
                            stack_size += 1
                    continue

                child = default_child[node]
                if child < 0:
                    child = first_child[node]
                if child >= 0:
                    node_stack[stack_size] = child
                    weight_stack[stack_size] = weight
                    stack_size += 1
        return out
else:
    def _predict_terminal_nodes_numba(
        X: np.ndarray,
        split_type: np.ndarray,
        feature_index: np.ndarray,
        threshold: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        default_child: np.ndarray,
        first_child: np.ndarray,
        missing_go_to_left: np.ndarray,
        edge_starts: np.ndarray,
        edge_ends: np.ndarray,
        edge_values: np.ndarray,
        edge_children: np.ndarray,
    ) -> np.ndarray:
        raise RuntimeError("numba is not available")

    def _predict_proba_nodes_numba(
        X: np.ndarray,
        split_type: np.ndarray,
        feature_index: np.ndarray,
        threshold: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        default_child: np.ndarray,
        first_child: np.ndarray,
        has_other_branch: np.ndarray,
        missing_go_to_left: np.ndarray,
        left_prob: np.ndarray,
        edge_starts: np.ndarray,
        edge_ends: np.ndarray,
        edge_values: np.ndarray,
        edge_children: np.ndarray,
        edge_probs: np.ndarray,
        leaf_proba: np.ndarray,
        fractional_missing: bool,
    ) -> np.ndarray:
        raise RuntimeError("numba is not available")


class EncodedNumpyJ48FastEngine(NumpyJ48Engine):
    """
    Fast backend built on a dense `float64` representation.

    The initial optimization encodes nominal attributes as integers and keeps
    missing values as `NaN`. This reduces the `dtype=object` cost in hot paths
    without changing the estimator's public contract.
    """

    def __init__(self, spec: Optional[J48EngineSpec] = None) -> None:
        super().__init__(spec=spec or J48EngineSpec(backend="numpy_fast", fidelity="equivalent"))
        if self.spec.backend != "numpy_fast":
            raise ValueError(f"Unsupported fast J48 backend: {self.spec.backend}")
        self._fast_object_heavy_fit_path_enabled: bool = True
        self._fast_feature_names: Optional[list[str]] = None
        self._fast_feature_count: int = 0
        self._fast_nominal_features: list[int] = []
        self._fast_numeric_features: list[int] = []
        self._fast_nominal_feature_set: set[int] = set()
        self._fast_nominal_domains: dict[int, list[int]] = {}
        self._fast_nominal_label_maps: dict[int, list[Any]] = {}
        self._fast_nominal_label_tuples: dict[int, tuple[Any, ...]] = {}
        self._fast_nominal_code_maps: dict[int, dict[Any, int]] = {}
        self._fast_nominal_unseen_codes: dict[int, float] = {}
        self._compiled_tree_cache: Optional[dict[str, Any]] = None
        self._compiled_tree_root: Any = None

    def _refresh_fast_predict_metadata(self) -> None:
        self._fast_nominal_feature_set = set(self._fast_nominal_features)
        self._fast_numeric_features = [
            feat for feat in range(self._fast_feature_count) if feat not in self._fast_nominal_feature_set
        ]
        self._fast_nominal_label_tuples = {
            int(feat): tuple(labels)
            for feat, labels in self._fast_nominal_label_maps.items()
        }

    def _detect_nominal_features(
        self,
        X_arr: np.ndarray,
        feature_names: Optional[list[str]],
        nominal_features: Optional[list[int]],
        auto_detect_nominal: bool,
    ) -> list[int]:
        nominal = set(int(v) for v in (nominal_features or []))
        if not auto_detect_nominal:
            return sorted(nominal)

        for feat in range(X_arr.shape[1]):
            if feat in nominal:
                continue
            col = np.asarray(X_arr[:, feat], dtype=object)
            non_missing = [
                self._to_python_scalar(v)
                for v in col.tolist()
                if not self._is_missing_scalar(v)
            ]
            if not non_missing:
                continue
            try:
                np.asarray(non_missing, dtype=np.float64)
            except Exception:
                nominal.add(feat)
        return sorted(nominal)

    def _can_use_object_heavy_fit_path(
        self,
        X: Any,
        nominal_features: Optional[list[int]],
        auto_detect_nominal: bool,
    ) -> bool:
        if not self._fast_object_heavy_fit_path_enabled:
            return False
        if auto_detect_nominal:
            return False
        if pd is None:
            return False
        if not (
            hasattr(X, "iloc")
            and hasattr(X, "dtypes")
            and getattr(X, "ndim", None) == 2
        ):
            return False
        n_features = int(getattr(X, "shape", (0, 0))[1])
        nominal_count = len(set(int(v) for v in (nominal_features or [])))
        return n_features > 0 and nominal_count >= max(1, (n_features + 1) // 2)

    def _resolve_nominal_domain(
        self,
        feat: int,
        feature_names: Optional[list[str]],
        observed_values: list[Any],
        nominal_value_domains: Optional[dict[Any, list[Any]]],
    ) -> list[Any]:
        merged = list(dict.fromkeys(observed_values))
        if nominal_value_domains is None:
            return merged

        raw_domain = None
        if feat in nominal_value_domains:
            raw_domain = nominal_value_domains[feat]
        elif feature_names is not None and feat < len(feature_names):
            name = feature_names[feat]
            if name in nominal_value_domains:
                raw_domain = nominal_value_domains[name]
        if raw_domain is None:
            return merged

        normalized = [
            self._to_python_scalar(v)
            for v in raw_domain
            if not self._is_missing_scalar(v)
        ]
        domain = list(dict.fromkeys(normalized))
        for value in merged:
            if value not in domain:
                domain.append(value)
        return domain

    def _encode_numeric_column(self, raw_col: np.ndarray) -> np.ndarray:
        if raw_col.dtype.kind in "biuf":
            # Vectorized equivalent of the per-value loop below.
            out = raw_col.astype(np.float64, copy=True)
            out[~np.isfinite(out)] = np.nan
            return out
        out = np.empty(raw_col.shape[0], dtype=np.float64)
        obj_col = np.asarray(raw_col, dtype=object)
        for idx, value in enumerate(obj_col.tolist()):
            value = self._to_python_scalar(value)
            if self._is_missing_scalar(value):
                out[idx] = np.nan
                continue
            try:
                out[idx] = float(value)
            except Exception:
                out[idx] = np.nan
        return out

    def _encode_numeric_series(self, series: Any) -> np.ndarray:
        if pd is None:
            return self._encode_numeric_column(np.asarray(series))
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.to_numpy(dtype=np.float64, na_value=np.nan)

    def _encode_nominal_column(
        self,
        raw_col: np.ndarray,
        feat: int,
        feature_names: Optional[list[str]],
        nominal_value_domains: Optional[dict[Any, list[Any]]],
    ) -> tuple[np.ndarray, list[Any], dict[Any, int]]:
        obj_col = np.asarray(raw_col, dtype=object)
        observed = [
            self._to_python_scalar(v)
            for v in obj_col.tolist()
            if not self._is_missing_scalar(v)
        ]
        domain_labels = self._resolve_nominal_domain(
            feat,
            feature_names,
            list(dict.fromkeys(observed)),
            nominal_value_domains,
        )
        code_map = {label: idx for idx, label in enumerate(domain_labels)}

        out = np.empty(obj_col.shape[0], dtype=np.float64)
        for idx, value in enumerate(obj_col.tolist()):
            value = self._to_python_scalar(value)
            if self._is_missing_scalar(value):
                out[idx] = np.nan
                continue
            code = code_map.get(value)
            if code is None:
                out[idx] = np.nan
            else:
                out[idx] = float(code)
        return out, domain_labels, code_map

    def _encode_nominal_series_fit(
        self,
        series: Any,
        feat: int,
        feature_names: Optional[list[str]],
        nominal_value_domains: Optional[dict[Any, list[Any]]],
    ) -> tuple[np.ndarray, list[Any], dict[Any, int]]:
        if pd is not None and (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            return self._encode_nominal_series_fit_categorical(
                series,
                feat,
                feature_names,
                nominal_value_domains,
            )
        return self._encode_nominal_series_fit_map(
            series,
            feat,
            feature_names,
            nominal_value_domains,
        )

    def _encode_nominal_series_fit_map(
        self,
        series: Any,
        feat: int,
        feature_names: Optional[list[str]],
        nominal_value_domains: Optional[dict[Any, list[Any]]],
    ) -> tuple[np.ndarray, list[Any], dict[Any, int]]:
        if pd is None:
            return self._encode_nominal_column(
                np.asarray(series),
                feat,
                feature_names,
                nominal_value_domains,
            )
        normalized = series.map(self._to_python_scalar)
        missing_mask = normalized.isna()
        text = normalized.astype("string")
        missing_mask |= text.fillna("").str.strip().isin(["", "?"])
        observed = [
            self._to_python_scalar(v)
            for v in normalized[~missing_mask].tolist()
        ]
        domain_labels = self._resolve_nominal_domain(
            feat,
            feature_names,
            list(dict.fromkeys(observed)),
            nominal_value_domains,
        )
        code_map = {label: idx for idx, label in enumerate(domain_labels)}
        encoded = normalized.map(code_map).astype(np.float64)
        encoded[missing_mask] = np.nan
        return encoded.to_numpy(dtype=np.float64, copy=False), domain_labels, code_map

    def _encode_nominal_series_fit_categorical(
        self,
        series: Any,
        feat: int,
        feature_names: Optional[list[str]],
        nominal_value_domains: Optional[dict[Any, list[Any]]],
    ) -> tuple[np.ndarray, list[Any], dict[Any, int]]:
        if pd is None:
            return self._encode_nominal_column(
                np.asarray(series),
                feat,
                feature_names,
                nominal_value_domains,
            )

        text = series.astype("string")
        missing_mask = text.isna()
        stripped = text.fillna("").str.strip()
        missing_mask |= stripped.isin(["", "?"])
        observed = [
            self._to_python_scalar(v)
            for v in pd.unique(text[~missing_mask]).tolist()
        ]
        domain_labels = self._resolve_nominal_domain(
            feat,
            feature_names,
            list(dict.fromkeys(observed)),
            nominal_value_domains,
        )
        code_map = {label: idx for idx, label in enumerate(domain_labels)}
        categorical = pd.Categorical(
            text.where(~missing_mask, None),
            categories=domain_labels,
            ordered=False,
        )
        codes = categorical.codes.astype(np.float64, copy=False)
        codes[codes < 0.0] = np.nan
        return codes, domain_labels, code_map

    def _encode_nominal_series_predict(self, series: Any, feat: int) -> np.ndarray:
        if pd is None:
            obj_col = np.asarray(series, dtype=object)
            code_map = self._fast_nominal_code_maps.get(feat, {})
            out = np.empty(obj_col.shape[0], dtype=np.float64)
            unseen_code = self._fast_nominal_unseen_codes.get(feat, np.nan)
            for idx, value in enumerate(obj_col.tolist()):
                value = self._to_python_scalar(value)
                if self._is_missing_scalar(value):
                    out[idx] = np.nan
                    continue
                code = code_map.get(value)
                out[idx] = unseen_code if code is None else float(code)
            return out

        if isinstance(series.dtype, pd.CategoricalDtype):
            return self._encode_nominal_series_predict_categorical(series, feat)

        return self._encode_nominal_values_predict_factorized(
            series.to_numpy(dtype=object, copy=False),
            feat,
        )

    def _encode_nominal_values_predict_factorized(
        self,
        values: np.ndarray,
        feat: int,
    ) -> np.ndarray:
        if pd is None:
            obj_col = np.asarray(values, dtype=object)
            code_map = self._fast_nominal_code_maps.get(feat, {})
            out = np.empty(obj_col.shape[0], dtype=np.float64)
            unseen_code = self._fast_nominal_unseen_codes.get(feat, np.nan)
            for idx, value in enumerate(obj_col.tolist()):
                value = self._to_python_scalar(value)
                if self._is_missing_scalar(value):
                    out[idx] = np.nan
                    continue
                code = code_map.get(value)
                out[idx] = unseen_code if code is None else float(code)
            return out

        factor_codes, uniques = pd.factorize(values, sort=False, use_na_sentinel=True)
        out = np.empty(len(values), dtype=np.float64)
        missing_rows = factor_codes < 0
        if np.any(missing_rows):
            out[missing_rows] = np.nan
        if len(uniques) == 0:
            return out

        unseen_code = self._fast_nominal_unseen_codes.get(feat, np.nan)
        code_map = self._fast_nominal_code_maps.get(feat, {})
        uniq_mapped = np.empty(len(uniques), dtype=np.float64)
        for idx, value in enumerate(uniques.tolist()):
            value = self._to_python_scalar(value)
            if self._is_missing_scalar(value):
                uniq_mapped[idx] = np.nan
                continue
            code = code_map.get(value)
            uniq_mapped[idx] = unseen_code if code is None else float(code)

        valid_rows = ~missing_rows
        if np.any(valid_rows):
            out[valid_rows] = uniq_mapped[factor_codes[valid_rows]]
        return out

    def _encode_nominal_series_predict_categorical(self, series: Any, feat: int) -> np.ndarray:
        codes = series.cat.codes.to_numpy(dtype=np.int64, copy=False)
        out = np.empty(codes.shape[0], dtype=np.float64)
        missing_rows = codes < 0
        if np.any(missing_rows):
            out[missing_rows] = np.nan
        if not np.any(~missing_rows):
            return out

        categories = [self._to_python_scalar(v) for v in series.cat.categories.tolist()]
        if tuple(categories) == self._fast_nominal_label_tuples.get(feat, ()):
            out[~missing_rows] = codes[~missing_rows].astype(np.float64, copy=False)
            return out

        unseen_code = self._fast_nominal_unseen_codes.get(feat, np.nan)
        code_map = self._fast_nominal_code_maps.get(feat, {})
        mapped_categories = np.empty(len(categories), dtype=np.float64)
        for idx, value in enumerate(categories):
            if self._is_missing_scalar(value):
                mapped_categories[idx] = np.nan
                continue
            code = code_map.get(value)
            mapped_categories[idx] = unseen_code if code is None else float(code)
        out[~missing_rows] = mapped_categories[codes[~missing_rows]]
        return out

    def prepare_fit_bundle(
        self,
        X: Any,
        y: Any,
        *,
        feature_names: Optional[list[str]] = None,
        nominal_features: Optional[list[int]] = None,
        auto_detect_nominal: bool = False,
        nominal_value_domains: Optional[dict[Any, list[Any]]] = None,
    ) -> dict[str, Any]:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = np.ravel(y_arr)
        if self._can_use_object_heavy_fit_path(X, nominal_features, auto_detect_nominal):
            n_rows, n_features = X.shape
            if n_rows != y_arr.shape[0]:
                raise ValueError("X and y have inconsistent lengths")
            inferred_feature_names = self.infer_feature_names(X, feature_names=feature_names)
            resolved_nominal_features = sorted(set(int(v) for v in (nominal_features or [])))
            nominal_feature_set = set(resolved_nominal_features)
            dtype_list = list(X.dtypes)
            numeric_typed_features = [
                feat
                for feat in range(n_features)
                if feat not in nominal_feature_set and pd.api.types.is_numeric_dtype(dtype_list[feat])
            ]
            numeric_typed_feature_set = set(numeric_typed_features)

            X_fast = np.empty((n_rows, n_features), dtype=np.float64)
            if numeric_typed_features:
                X_fast[:, numeric_typed_features] = X.iloc[:, numeric_typed_features].to_numpy(
                    dtype=np.float64,
                    na_value=np.nan,
                )

            resolved_domains: dict[int, list[int]] = {}
            label_maps: dict[int, list[Any]] = {}
            code_maps: dict[int, dict[Any, int]] = {}

            for feat in range(n_features):
                if feat in numeric_typed_feature_set:
                    continue
                series = X.iloc[:, feat]
                if feat in nominal_feature_set:
                    enc_col, labels, code_map = self._encode_nominal_series_fit(
                        series,
                        feat,
                        inferred_feature_names,
                        nominal_value_domains,
                    )
                    X_fast[:, feat] = enc_col
                    resolved_domains[feat] = list(range(len(labels)))
                    label_maps[feat] = labels
                    code_maps[feat] = code_map
                else:
                    X_fast[:, feat] = self._encode_numeric_series(series)

            self._fast_feature_names = None if inferred_feature_names is None else list(inferred_feature_names)
            self._fast_nominal_features = list(resolved_nominal_features)
            self._fast_feature_count = int(n_features)
            self._fast_nominal_domains = resolved_domains
            self._fast_nominal_label_maps = label_maps
            self._fast_nominal_code_maps = code_maps
            self._fast_nominal_unseen_codes = {
                feat: float(len(labels))
                for feat, labels in label_maps.items()
            }
            self._refresh_fast_predict_metadata()
            self._compiled_tree_cache = None

            bundle = {
                "X": X_fast,
                "y": y_arr,
                "feature_names": self._fast_feature_names,
                "nominal_features": list(resolved_nominal_features),
                "nominal_value_domains": resolved_domains,
                "auto_detect_nominal": False,
            }
            return bundle

        X_arr, y_arr = self.prepare_fit_data(X, y)

        inferred_feature_names = self.infer_feature_names(X, feature_names=feature_names)
        resolved_nominal_features = self._detect_nominal_features(
            X_arr,
            inferred_feature_names,
            nominal_features,
            auto_detect_nominal,
        )
        X_fast = np.empty(X_arr.shape, dtype=np.float64)
        resolved_domains: dict[int, list[int]] = {}
        label_maps: dict[int, list[Any]] = {}
        code_maps: dict[int, dict[Any, int]] = {}

        for feat in range(X_arr.shape[1]):
            raw_col = X_arr[:, feat]
            series = X.iloc[:, feat] if pd is not None and hasattr(X, "iloc") else None
            if feat in resolved_nominal_features:
                if series is not None:
                    enc_col, labels, code_map = self._encode_nominal_series_fit(
                        series,
                        feat,
                        inferred_feature_names,
                        nominal_value_domains,
                    )
                else:
                    enc_col, labels, code_map = self._encode_nominal_column(
                        raw_col,
                        feat,
                        inferred_feature_names,
                        nominal_value_domains,
                    )
                X_fast[:, feat] = enc_col
                resolved_domains[feat] = list(range(len(labels)))
                label_maps[feat] = labels
                code_maps[feat] = code_map
            else:
                if series is not None:
                    X_fast[:, feat] = self._encode_numeric_series(series)
                else:
                    X_fast[:, feat] = self._encode_numeric_column(raw_col)

        self._fast_feature_names = None if inferred_feature_names is None else list(inferred_feature_names)
        self._fast_nominal_features = list(resolved_nominal_features)
        self._fast_feature_count = int(X_arr.shape[1])
        self._fast_nominal_domains = resolved_domains
        self._fast_nominal_label_maps = label_maps
        self._fast_nominal_code_maps = code_maps
        self._fast_nominal_unseen_codes = {
            feat: float(len(labels))
            for feat, labels in label_maps.items()
        }
        self._refresh_fast_predict_metadata()
        self._compiled_tree_cache = None

        bundle = {
            "X": X_fast,
            "y": y_arr,
            "feature_names": self._fast_feature_names,
            "nominal_features": list(resolved_nominal_features),
            "nominal_value_domains": resolved_domains,
            "auto_detect_nominal": False,
        }
        return bundle

    def prepare_predict_data(self, X: Any, expected_features: int) -> np.ndarray:
        self._last_prepare_predict_cache_hit = False
        is_pandas_2d = (
            pd is not None
            and hasattr(X, "iloc")
            and hasattr(X, "dtypes")
            and getattr(X, "ndim", None) == 2
        )
        if is_pandas_2d:
            n_rows, n_cols = X.shape
            if n_cols != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {n_cols}"
                )
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            if X_arr.ndim != 2:
                raise ValueError("J48Classifier expects a 2D feature matrix")
            if X_arr.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {X_arr.shape[1]}"
                )

        if is_pandas_2d:
            nominal_set = self._fast_nominal_feature_set
            numeric_features = self._fast_numeric_features
            if not nominal_set:
                dtypes = list(X.dtypes)
                if all(pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes):
                    out = np.ascontiguousarray(X.to_numpy(dtype=np.float64, na_value=np.nan))
                    return out
                out = np.empty((n_rows, n_cols), dtype=np.float64)
                for feat in range(n_cols):
                    series = X.iloc[:, feat]
                    if pd.api.types.is_numeric_dtype(series.dtype):
                        out[:, feat] = series.to_numpy(dtype=np.float64, na_value=np.nan)
                    else:
                        out[:, feat] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
                return out
            X_fast = np.empty((n_rows, n_cols), dtype=np.float64)
            if numeric_features:
                dtypes = list(X.dtypes)
                bulk_numeric_features = [
                    feat for feat in numeric_features if pd.api.types.is_numeric_dtype(dtypes[feat])
                ]
                coerced_numeric_features = [
                    feat for feat in numeric_features if feat not in bulk_numeric_features
                ]
                if bulk_numeric_features:
                    X_fast[:, bulk_numeric_features] = np.ascontiguousarray(
                        X.iloc[:, bulk_numeric_features].to_numpy(dtype=np.float64, na_value=np.nan)
                    )
                for feat in coerced_numeric_features:
                    series = X.iloc[:, feat]
                    X_fast[:, feat] = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
            for feat in self._fast_nominal_features:
                X_fast[:, feat] = self._encode_nominal_series_predict(X.iloc[:, feat], feat)
            return X_fast

        numeric_features = self._fast_numeric_features
        if not self._fast_nominal_features and np.issubdtype(X_arr.dtype, np.number):
            # All-numeric input: a float64 C-contiguous view needs no copy.
            # Safe because prepared inputs are only read, never cached.
            return np.ascontiguousarray(X_arr, dtype=np.float64)

        X_fast = np.empty(X_arr.shape, dtype=np.float64)
        if numeric_features:
            if np.issubdtype(X_arr.dtype, np.number):
                X_fast[:, numeric_features] = X_arr[:, numeric_features].astype(np.float64, copy=False)
            else:
                for feat in numeric_features:
                    X_fast[:, feat] = self._encode_numeric_column(X_arr[:, feat])

        for feat in self._fast_nominal_features:
            raw_col = X_arr[:, feat]
            X_fast[:, feat] = self._encode_nominal_values_predict_factorized(raw_col, feat)
        return X_fast

    def _fast_predict_path_details(self, X: np.ndarray, estimator: Any) -> dict[str, Any]:
        details = {
            "engine_numba_available": bool(ENGINE_NUMBA_AVAILABLE),
            "fractional_missing_enabled": bool(getattr(estimator, "enable_fractional_missing", False)),
            "has_any_nonfinite": bool(np.any(~np.isfinite(X))),
            "has_nominal_nonfinite": False,
            "has_nominal_unseen": False,
            "nominal_feature_count": int(len(self._fast_nominal_features)),
            "uses_missing_aware_traversal": False,
            "requires_proba_mixture": False,
        }
        if not ENGINE_NUMBA_AVAILABLE:
            details.update(
                {
                    "fast_path": False,
                    "path": "core_fallback",
                    "reason": "numba_unavailable",
                }
            )
            return details

        nominal_set = set(self._fast_nominal_features)
        for feat in nominal_set:
            col = np.asarray(X[:, feat], dtype=np.float64)
            if np.any(~np.isfinite(col)):
                details["has_nominal_nonfinite"] = True
            unseen_code = self._fast_nominal_unseen_codes.get(feat)
            if unseen_code is not None and np.any(col == unseen_code):
                details["has_nominal_unseen"] = True

        details["uses_missing_aware_traversal"] = bool(
            details["has_any_nonfinite"]
            or details["has_nominal_unseen"]
            or details["has_nominal_nonfinite"]
        )
        details["requires_proba_mixture"] = bool(
            details["fractional_missing_enabled"]
            and details["uses_missing_aware_traversal"]
        )

        details.update(
            {
                "fast_path": True,
                "path": (
                    "fast_compiled_missing_aware"
                    if details["uses_missing_aware_traversal"]
                    else "fast_compiled"
                ),
                "reason": (
                    "fractional_missing_numba_mixture"
                    if details["requires_proba_mixture"]
                    else (
                        "missing_or_unseen_supported"
                        if details["uses_missing_aware_traversal"]
                        else "fast_path_available"
                    )
                ),
            }
        )
        return details

    def _compile_tree(self, estimator: Any) -> dict[str, Any]:
        root = estimator.root_
        if root is None:
            return {
                "split_type": np.zeros(0, dtype=np.int8),
                "feature_index": np.zeros(0, dtype=np.int32),
                "threshold": np.zeros(0, dtype=np.float64),
                "left_child": np.zeros(0, dtype=np.int32),
                "right_child": np.zeros(0, dtype=np.int32),
                "default_child": np.zeros(0, dtype=np.int32),
                "first_child": np.zeros(0, dtype=np.int32),
                "has_other_branch": np.zeros(0, dtype=np.bool_),
                "missing_go_to_left": np.zeros(0, dtype=np.bool_),
                "left_prob": np.zeros(0, dtype=np.float64),
                "edge_starts": np.zeros(0, dtype=np.int32),
                "edge_ends": np.zeros(0, dtype=np.int32),
                "edge_values": np.zeros(0, dtype=np.int32),
                "edge_children": np.zeros(0, dtype=np.int32),
                "edge_probs": np.zeros(0, dtype=np.float64),
                "leaf_pred_idx": np.zeros(0, dtype=np.int32),
                "leaf_proba": np.zeros((0, estimator.n_classes_), dtype=np.float64),
                "leaf_proba_raw": np.zeros((0, estimator.n_classes_), dtype=np.float64),
            }

        split_type: list[int] = []
        feature_index: list[int] = []
        threshold: list[float] = []
        left_child: list[int] = []
        right_child: list[int] = []
        default_child: list[int] = []
        first_child: list[int] = []
        has_other_branch: list[bool] = []
        missing_go_to_left: list[bool] = []
        left_prob: list[float] = []
        edge_starts: list[int] = []
        edge_ends: list[int] = []
        leaf_pred_idx: list[int] = []
        leaf_proba: list[np.ndarray] = []
        leaf_proba_raw: list[np.ndarray] = []
        edge_values: list[int] = []
        edge_children: list[int] = []
        edge_probs: list[float] = []

        def leaf_distribution(node: Any, laplace: bool) -> np.ndarray:
            counts = getattr(node, "class_counts", None)
            prob_counts = getattr(node, "probability_counts", None)
            n_classes = int(estimator.n_classes_)
            if counts is not None:
                counts = np.asarray(counts, dtype=np.float64)
                total = float(np.sum(counts))
                if total > 0.0:
                    if laplace:
                        return ((counts + 1.0) / (total + float(n_classes))).astype(np.float64, copy=False)
                    return (counts / total).astype(np.float64, copy=False)
            if prob_counts is not None:
                prob_counts = np.asarray(prob_counts, dtype=np.float64)
                total = float(np.sum(prob_counts))
                if total > 0.0:
                    if laplace:
                        return ((prob_counts + 1.0) / (total + float(n_classes))).astype(np.float64, copy=False)
                    return (prob_counts / total).astype(np.float64, copy=False)
            dist = np.zeros(n_classes, dtype=np.float64)
            pred_idx = getattr(node, "prediction_idx", None)
            if pred_idx is None:
                dist[:] = 1.0 / float(n_classes)
            else:
                dist[int(pred_idx)] = 1.0
            return dist

        node_index: dict[int, int] = {}
        stack = [root]
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in node_index:
                continue
            idx = len(split_type)
            node_index[node_id] = idx
            split_type.append(-1 if node.is_leaf else (0 if node.split_type == "numeric" else 1))
            feature_index.append(-1 if node.feature_index is None else int(node.feature_index))
            threshold.append(np.nan if node.threshold is None else float(node.threshold))
            left_child.append(-1)
            right_child.append(-1)
            default_child.append(-1)
            first_child.append(-1)
            has_other_branch.append(False)
            missing_go_to_left.append(bool(getattr(node, "missing_go_to_left", True)))
            left_prob.append(float(getattr(node, "left_prob", 0.5)))
            edge_starts.append(0)
            edge_ends.append(0)
            leaf_pred_idx.append(0 if node.prediction_idx is None else int(node.prediction_idx))
            if node.is_leaf:
                use_laplace = bool(getattr(estimator, "use_laplace", False))
                proba = leaf_distribution(node, use_laplace)
                proba_raw = leaf_distribution(node, False) if use_laplace else proba
            else:
                proba = np.zeros(estimator.n_classes_, dtype=np.float64)
                proba_raw = proba
            leaf_proba.append(proba)
            leaf_proba_raw.append(proba_raw)

            if node.is_leaf:
                continue
            if node.split_type == "numeric":
                if node.right is not None:
                    stack.append(node.right)
                if node.left is not None:
                    stack.append(node.left)
            elif node.nominal_children:
                for child in reversed(list(node.nominal_children.values())):
                    if child is not None:
                        stack.append(child)

        nodes_by_idx = [None] * len(node_index)
        pending = [root]
        seen = set()
        while pending:
            node = pending.pop()
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            idx = node_index[node_id]
            nodes_by_idx[idx] = node
            if node.is_leaf:
                continue
            if node.split_type == "numeric":
                if node.left is not None:
                    pending.append(node.left)
                if node.right is not None:
                    pending.append(node.right)
            elif node.nominal_children:
                for child in node.nominal_children.values():
                    if child is not None:
                        pending.append(child)

        for idx, node in enumerate(nodes_by_idx):
            if node is None or node.is_leaf:
                continue
            if node.split_type == "numeric":
                left_child[idx] = -1 if node.left is None else node_index[id(node.left)]
                right_child[idx] = -1 if node.right is None else node_index[id(node.right)]
                continue

            start = len(edge_values)
            if node.nominal_children:
                for edge, child in node.nominal_children.items():
                    if child is None:
                        continue
                    if edge == node.nominal_default_child:
                        default_child[idx] = node_index[id(child)]
                    if edge == "__WEKA_OTHER__":
                        has_other_branch[idx] = True
                    if first_child[idx] < 0:
                        first_child[idx] = node_index[id(child)]
                    try:
                        edge_value = int(edge)
                    except Exception:
                        edge_value = _COMPILED_NOMINAL_OTHER_SENTINEL
                    edge_values.append(edge_value)
                    edge_children.append(node_index[id(child)])
                    edge_probs.append(float((node.nominal_child_probs or {}).get(edge, 0.0)))
            edge_starts[idx] = start
            edge_ends[idx] = len(edge_values)

        return {
            "split_type": np.asarray(split_type, dtype=np.int8),
            "feature_index": np.asarray(feature_index, dtype=np.int32),
            "threshold": np.asarray(threshold, dtype=np.float64),
            "left_child": np.asarray(left_child, dtype=np.int32),
            "right_child": np.asarray(right_child, dtype=np.int32),
            "default_child": np.asarray(default_child, dtype=np.int32),
            "first_child": np.asarray(first_child, dtype=np.int32),
            "has_other_branch": np.asarray(has_other_branch, dtype=np.bool_),
            "missing_go_to_left": np.asarray(missing_go_to_left, dtype=np.bool_),
            "left_prob": np.asarray(left_prob, dtype=np.float64),
            "edge_starts": np.asarray(edge_starts, dtype=np.int32),
            "edge_ends": np.asarray(edge_ends, dtype=np.int32),
            "edge_values": np.asarray(edge_values, dtype=np.int32),
            "edge_children": np.asarray(edge_children, dtype=np.int32),
            "edge_probs": np.asarray(edge_probs, dtype=np.float64),
            "leaf_pred_idx": np.asarray(leaf_pred_idx, dtype=np.int32),
            "leaf_proba": np.asarray(leaf_proba, dtype=np.float64),
            "leaf_proba_raw": np.asarray(leaf_proba_raw, dtype=np.float64),
        }

    def _ensure_compiled_tree(self, estimator: Any) -> dict[str, Any]:
        # Recompile whenever the estimator holds a different tree (e.g. after
        # a refit through `fit_prepared_bundle`, which reuses this engine).
        # Also recompile caches that predate a compiled field (unpickled
        # models from an older release).
        if (
            self._compiled_tree_cache is None
            or self._compiled_tree_root is not estimator.root_
            or "leaf_proba_raw" not in self._compiled_tree_cache
        ):
            self._compiled_tree_cache = self._compile_tree(estimator)
            self._compiled_tree_root = estimator.root_
        return self._compiled_tree_cache

    def _restore_nominal_value(self, feature_index: Optional[int], raw_value: Any) -> Any:
        if feature_index is None:
            return raw_value
        feat = int(feature_index)
        labels = self._fast_nominal_label_maps.get(feat)
        if labels is None:
            return raw_value
        try:
            if raw_value is None:
                return raw_value
            if isinstance(raw_value, str):
                stripped = raw_value.strip()
                if stripped == "":
                    return raw_value
                raw_idx = int(float(stripped))
            else:
                raw_idx = int(float(raw_value))
        except Exception:
            return raw_value
        if 0 <= raw_idx < len(labels):
            return labels[raw_idx]
        return raw_value

    def _postprocess_exported_node(self, node: dict[str, Any]) -> None:
        if node.get("split_type") == "nominal":
            feat = node.get("feature_index")
            if "nominal_default_child" in node:
                node["nominal_default_child"] = self._restore_nominal_value(feat, node["nominal_default_child"])
            if "nominal_child_probs" in node and isinstance(node["nominal_child_probs"], list):
                for row in node["nominal_child_probs"]:
                    row["value"] = self._restore_nominal_value(feat, row.get("value"))
            for child in node.get("children", []):
                child["edge"] = self._restore_nominal_value(feat, child.get("edge"))
                if "child" in child and isinstance(child["child"], dict):
                    self._postprocess_exported_node(child["child"])
        else:
            for child in node.get("children", []):
                if "child" in child and isinstance(child["child"], dict):
                    self._postprocess_exported_node(child["child"])

    def postprocess_export_tree(self, exported: dict[str, Any]) -> dict[str, Any]:
        restored = copy.deepcopy(exported)
        root = restored.get("root")
        if isinstance(root, dict):
            self._postprocess_exported_node(root)
        return restored

    def can_use_fast_hard_predict(self, X: np.ndarray, estimator: Any) -> bool:
        return bool(self._fast_predict_path_details(X, estimator)["fast_path"])

    def hard_predict(self, X: np.ndarray, estimator: Any) -> np.ndarray:
        details = self._fast_predict_path_details(X, estimator)
        if details["requires_proba_mixture"]:
            # WEKA's J48.classifyInstance: the class comes from the
            # distribution without Laplace smoothing, choosing the first
            # class on (near-)ties, exactly as the strict line does.
            proba = self.predict_proba_fast(X, estimator, laplace=False)
            return estimator.classes_[_weka_max_index(proba)]
        if ENGINE_NUMBA_AVAILABLE:
            compiled = self._ensure_compiled_tree(estimator)
            if compiled["split_type"].size == 0:
                return estimator.classes_[:0]
            terminal = _predict_terminal_nodes_numba(
                X,
                compiled["split_type"],
                compiled["feature_index"],
                compiled["threshold"],
                compiled["left_child"],
                compiled["right_child"],
                compiled["default_child"],
                compiled["first_child"],
                compiled["missing_go_to_left"],
                compiled["edge_starts"],
                compiled["edge_ends"],
                compiled["edge_values"],
                compiled["edge_children"],
            )
            return estimator.classes_[compiled["leaf_pred_idx"][terminal]]

        root = estimator.root_
        if root is None:
            return estimator.classes_[:0]

        out_idx = np.empty(X.shape[0], dtype=np.int32)
        for row_idx in range(X.shape[0]):
            node = root
            while not node.is_leaf:
                feat = int(node.feature_index)
                value = float(X[row_idx, feat])
                if node.split_type == "numeric":
                    if np.isfinite(value):
                        node = node.left if value <= float(node.threshold) else node.right
                    else:
                        node = node.left if bool(node.missing_go_to_left) else node.right
                    continue

                child = None
                if np.isfinite(value) and node.nominal_children is not None:
                    child = node.nominal_children.get(int(value))
                if child is None and node.nominal_children is not None:
                    child = node.nominal_children.get(node.nominal_default_child)
                if child is None and node.nominal_children:
                    child = next(iter(node.nominal_children.values()))
                if child is None:
                    break
                node = child

            out_idx[row_idx] = int(node.prediction_idx or 0)
        return estimator.classes_[out_idx]

    def can_use_fast_predict_proba(self, X: np.ndarray, estimator: Any) -> bool:
        return bool(self._fast_predict_path_details(X, estimator)["fast_path"])

    def predict_proba_fast(self, X: np.ndarray, estimator: Any, laplace: Optional[bool] = None) -> np.ndarray:
        compiled = self._ensure_compiled_tree(estimator)
        use_laplace = bool(getattr(estimator, "use_laplace", False)) if laplace is None else bool(laplace)
        leaf_proba = compiled["leaf_proba"] if use_laplace else compiled["leaf_proba_raw"]
        if compiled["split_type"].size == 0:
            return np.zeros((0, estimator.n_classes_), dtype=np.float64)
        details = self._fast_predict_path_details(X, estimator)
        if details["uses_missing_aware_traversal"]:
            proba = _predict_proba_nodes_numba(
                X,
                compiled["split_type"],
                compiled["feature_index"],
                compiled["threshold"],
                compiled["left_child"],
                compiled["right_child"],
                compiled["default_child"],
                compiled["first_child"],
                compiled["has_other_branch"],
                compiled["missing_go_to_left"],
                compiled["left_prob"],
                compiled["edge_starts"],
                compiled["edge_ends"],
                compiled["edge_values"],
                compiled["edge_children"],
                compiled["edge_probs"],
                leaf_proba,
                bool(getattr(estimator, "enable_fractional_missing", False)),
            )
            row_sums = np.sum(proba, axis=1)
            zero_rows = row_sums == 0.0
            if np.any(zero_rows):
                proba[zero_rows] = np.float64(1.0 / float(estimator.n_classes_))
            return proba

        terminal = _predict_terminal_nodes_numba(
            X,
            compiled["split_type"],
            compiled["feature_index"],
            compiled["threshold"],
            compiled["left_child"],
            compiled["right_child"],
            compiled["default_child"],
            compiled["first_child"],
            compiled["missing_go_to_left"],
            compiled["edge_starts"],
            compiled["edge_ends"],
            compiled["edge_values"],
            compiled["edge_children"],
        )
        return leaf_proba[terminal]

    def describe_hard_predict_path(self, X: np.ndarray, estimator: Any) -> dict[str, Any]:
        return self._fast_predict_path_details(X, estimator)

    def describe_predict_proba_path(self, X: np.ndarray, estimator: Any) -> dict[str, Any]:
        return self._fast_predict_path_details(X, estimator)


def build_engine(
    backend: str = "numpy",
    fidelity: str = "strict",
) -> NumpyJ48Engine:
    spec = J48EngineSpec(backend=str(backend), fidelity=str(fidelity))
    if spec.backend == "numpy_fast":
        return EncodedNumpyJ48FastEngine(spec=spec)
    return NumpyJ48Engine(spec=spec)
