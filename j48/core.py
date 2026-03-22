from __future__ import annotations

"""
Core J48-targeting implementation used by the experimental tree package.

The code in this module is the baseline engine under alignment with WEKA J48.
It remains intentionally isolated from the stable project classifier path so
fidelity work and future performance backends can evolve without breaking the
existing C4.5 implementation used elsewhere in the repository.
"""

import copy
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any
from statistics import NormalDist
import logging

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)

_NOMINAL_OTHER_BRANCH = "__WEKA_OTHER__"


def _to_python_scalar(value: Any) -> Any:
    """
    Convert NumPy scalars and bytes to plain Python types for comparisons,
    JSON export, and consistent nominal-attribute handling.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            value = str(value)
    return value


def _is_missing_scalar(value: Any) -> bool:
    """
    Detect missing values robustly for numeric and object columns.
    """
    value = _to_python_scalar(value)
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "?"}
    if isinstance(value, (float, np.floating)):
        return not np.isfinite(value)
    return False


def _to_jsonable(value: Any) -> Any:
    """
    Convert internal values to simple serializable forms.
    """
    value = _to_python_scalar(value)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _entropy(y: np.ndarray) -> float:
    """
    Compute the entropy $H(Y)$ of a label array.

    Optimization: uses `bincount` for integer arrays, which is faster than
    `unique()`.

    Args:
        y: Label array with integer-encoded classes.

    Returns:
        Entropy in bits (base 2).
    """
    if y.size == 0:
        return 0.0
    
    # bincount is about 2x faster than unique for integer arrays.
    counts = np.bincount(y)
    counts = counts[counts > 0]  # Filter out zero counts.
    
    if counts.size <= 1:
        return 0.0
        
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _entropy_from_counts_fast(counts: np.ndarray, total: int, log2_lut: np.ndarray) -> float:
    """
        Compute entropy $H(Y)$ from per-class counts using:
      H = log2(total) - (1/total) * sum_i c_i * log2(c_i)

        Uses a precomputed `log2` lookup table to avoid recalculating logarithms
        thousands of times during split search.
    """
    if total <= 1:
        return 0.0
    # Treat log2(0) as 0 because c_i * log2(c_i) = 0 when c_i = 0.
    c_log_c = float(np.dot(counts, log2_lut[counts]))
    return float(log2_lut[total] - (c_log_c / total))


def _gain_ratio(
    y: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    base_entropy: Optional[float] = None,
) -> float:
    """
    Compute the Gain Ratio (C4.5) = Information Gain / Intrinsic Value.

    Gain Ratio corrects the Information Gain bias toward attributes with many
    values by penalizing splits that create many small partitions.

    Args:
        y: Labels at the parent node.
        y_left: Labels at the left child.
        y_right: Labels at the right child.
        base_entropy: Precomputed parent entropy, if available.

    Returns:
        Gain Ratio, where larger is better.
    """
    n = y.size
    if n == 0:
        return 0.0

    if base_entropy is None:
        base_entropy = _entropy(y)

    n_left = y_left.size
    n_right = y_right.size

    # Do not allow empty splits.
    if n_left == 0 or n_right == 0:
        return 0.0

    # Information Gain
    h_left = _entropy(y_left)
    h_right = _entropy(y_right)
    info_gain = base_entropy - (n_left / n) * h_left - (n_right / n) * h_right

    # Intrinsic Value penalizes unbalanced splits.
    p_left = n_left / n
    p_right = n_right / n
    intrinsic_value = 0.0
    
    if p_left > 0:
        intrinsic_value -= p_left * np.log2(p_left)
    if p_right > 0:
        intrinsic_value -= p_right * np.log2(p_right)

    # Avoid division by zero.
    if intrinsic_value == 0.0:
        return 0.0

    return float(info_gain / intrinsic_value)


def _entropy_from_weighted_counts(counts: np.ndarray) -> float:
    """
    Entropy from weighted per-class counts.

    Unlike `_entropy_from_counts_fast`, this version accepts fractional
    counts, which are needed when weighted instances are propagated.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0

    positive = counts[counts > 0.0]
    if positive.size <= 1:
        return 0.0

    probs = positive / total
    return float(-(probs * np.log2(probs)).sum())


def _entropy_from_weighted_counts_matrix(counts: np.ndarray) -> np.ndarray:
    """
    Row-wise entropy for a matrix of weighted per-class counts.

    Preserves the same semantics as `_entropy_from_weighted_counts`, but
    avoids a Python loop when evaluating many numeric thresholds in batch.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("counts must be a 2D matrix")
    if counts.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    totals = np.sum(counts, axis=1, dtype=np.float64)
    entropies = np.zeros(counts.shape[0], dtype=np.float64)
    valid_rows = totals > 0.0
    if not np.any(valid_rows):
        return entropies

    positive = counts > 0.0
    safe_counts = np.where(positive, counts, 1.0)
    c_log_c = np.where(positive, counts * np.log2(safe_counts), 0.0).sum(axis=1)
    entropies[valid_rows] = np.log2(totals[valid_rows]) - (c_log_c[valid_rows] / totals[valid_rows])
    return entropies


def _binary_entropy_from_positive_weight(positive_weight: np.ndarray, total_weight: np.ndarray) -> np.ndarray:
    """
    Vectorized binary entropy from positive-class weight and total row weight.

    Used to accelerate numeric-threshold evaluation in binary problems.
    binarios, evitando construir una matriz one-hot completa.
    """
    positive_weight = np.asarray(positive_weight, dtype=np.float64)
    total_weight = np.asarray(total_weight, dtype=np.float64)
    entropies = np.zeros_like(total_weight, dtype=np.float64)
    valid = total_weight > 0.0
    if not np.any(valid):
        return entropies

    pos = np.clip(positive_weight[valid], 0.0, total_weight[valid])
    neg = total_weight[valid] - pos
    c_log_c = np.zeros_like(pos, dtype=np.float64)
    pos_mask = pos > 0.0
    neg_mask = neg > 0.0
    c_log_c[pos_mask] += pos[pos_mask] * np.log2(pos[pos_mask])
    c_log_c[neg_mask] += neg[neg_mask] * np.log2(neg[neg_mask])
    entropies[valid] = np.log2(total_weight[valid]) - (c_log_c / total_weight[valid])
    return entropies


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _binary_entropy_scalar_numba(positive_weight: float, total_weight: float) -> float:
        if total_weight <= 0.0:
            return 0.0
        pos = positive_weight
        if pos < 0.0:
            pos = 0.0
        if pos > total_weight:
            pos = total_weight
        neg = total_weight - pos
        terms = 0.0
        if pos > 0.0:
            terms += pos * np.log2(pos)
        if neg > 0.0:
            terms += neg * np.log2(neg)
        return np.log2(total_weight) - (terms / total_weight)


    @njit(cache=True)
    def _extract_sorted_numeric_feature_numba(
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = x_feat.shape[0]
        valid_count = 0
        for i in range(n):
            if np.isfinite(x_feat[i]):
                valid_count += 1

        if valid_count == 0:
            return (
                0,
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
            )

        valid_x = np.empty(valid_count, dtype=np.float64)
        valid_y = np.empty(valid_count, dtype=np.int64)
        valid_w = np.empty(valid_count, dtype=np.float64)
        valid_idx = np.empty(valid_count, dtype=np.int32)
        out = 0
        for i in range(n):
            value = x_feat[i]
            if np.isfinite(value):
                valid_x[out] = value
                valid_y[out] = y_sub[i]
                valid_w[out] = weights[i]
                valid_idx[out] = i
                out += 1

        order = np.argsort(valid_x)
        x_sorted = np.empty(valid_count, dtype=np.float64)
        y_sorted = np.empty(valid_count, dtype=np.int64)
        w_sorted = np.empty(valid_count, dtype=np.float64)
        sorted_local_idx = np.empty(valid_count, dtype=np.int32)
        for i in range(valid_count):
            src = order[i]
            x_sorted[i] = valid_x[src]
            y_sorted[i] = valid_y[src]
            w_sorted[i] = valid_w[src]
            sorted_local_idx[i] = valid_idx[src]

        return (1, x_sorted, y_sorted, w_sorted, sorted_local_idx)


    @njit(cache=True)
    def _find_best_binary_numeric_split_numba(
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        w_sorted: np.ndarray,
        total_weight: float,
        min_leaf: float,
        n_classes: int,
        max_thresholds: int,
        use_mdl_correction: bool,
    ) -> tuple[int, int, float, float, float, float, float, int, float]:
        n = x_sorted.shape[0]
        if n <= 1:
            return (0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

        known_weight = 0.0
        total_pos = 0.0
        for i in range(n):
            wi = w_sorted[i]
            known_weight += wi
            if y_sorted[i] == 1:
                total_pos += wi

        min_split = min(25.0, max(min_leaf, 0.1 * (known_weight / max(float(n_classes), 1.0))))
        valid_positions = np.empty(n - 1, dtype=np.int32)
        left_weights = np.empty(n - 1, dtype=np.float64)
        left_pos_weights = np.empty(n - 1, dtype=np.float64)
        count = 0
        prefix_weight = 0.0
        prefix_pos = 0.0
        for i in range(n - 1):
            wi = w_sorted[i]
            prefix_weight += wi
            if y_sorted[i] == 1:
                prefix_pos += wi
            if x_sorted[i] + 1e-5 < x_sorted[i + 1]:
                right_weight = known_weight - prefix_weight
                if prefix_weight >= min_split - 1e-12 and right_weight >= min_split - 1e-12:
                    valid_positions[count] = i
                    left_weights[count] = prefix_weight
                    left_pos_weights[count] = prefix_pos
                    count += 1

        if count == 0:
            return (0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0, known_weight)

        feat_entropy = _binary_entropy_scalar_numba(total_pos, known_weight)
        best_info_gain = -1.0
        best_pos = -1
        best_left = 0.0
        best_right = 0.0

        if max_thresholds >= 0 and count > max_thresholds:
            num_eval = max_thresholds
        else:
            num_eval = count

        for eval_idx in range(num_eval):
            if num_eval == count:
                cand_idx = eval_idx
            elif num_eval == 1:
                cand_idx = 0
            else:
                cand_idx = int((eval_idx * (count - 1)) / (num_eval - 1))

            pos_idx = valid_positions[cand_idx]
            left_weight = left_weights[cand_idx]
            right_weight = known_weight - left_weight
            left_pos = left_pos_weights[cand_idx]
            right_pos = total_pos - left_pos
            h_left = _binary_entropy_scalar_numba(left_pos, left_weight)
            h_right = _binary_entropy_scalar_numba(right_pos, right_weight)
            child_entropy = ((left_weight / known_weight) * h_left) + ((right_weight / known_weight) * h_right)
            info_gain = (known_weight / total_weight) * (feat_entropy - child_entropy)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_pos = pos_idx
                best_left = left_weight
                best_right = right_weight

        if best_info_gain <= 0.0:
            return (0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, count, known_weight)

        if use_mdl_correction and count > 1:
            info_gain_adj = best_info_gain - (np.log2(float(count)) / total_weight)
        else:
            info_gain_adj = best_info_gain
        if info_gain_adj <= 0.0:
            return (0, -1, best_info_gain, info_gain_adj, 0.0, best_left, best_right, count, known_weight)

        missing_weight = total_weight - known_weight
        p_left = best_left / total_weight
        p_right = best_right / total_weight
        p_missing = missing_weight / total_weight if total_weight > 0.0 else 0.0
        intrinsic = 0.0
        if p_left > 0.0:
            intrinsic -= p_left * np.log2(p_left)
        if p_right > 0.0:
            intrinsic -= p_right * np.log2(p_right)
        if p_missing > 0.0:
            intrinsic -= p_missing * np.log2(p_missing)
        if intrinsic <= 0.0:
            return (0, -1, best_info_gain, info_gain_adj, intrinsic, best_left, best_right, count, known_weight)

        gain_ratio = info_gain_adj / intrinsic
        if gain_ratio <= 0.0:
            return (0, -1, best_info_gain, info_gain_adj, intrinsic, best_left, best_right, count, known_weight)

        return (1, best_pos, best_info_gain, info_gain_adj, intrinsic, best_left, best_right, count, known_weight)


    @njit(cache=True)
    def _find_best_binary_numeric_split_unsorted_numba(
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
        min_leaf: float,
        n_classes: int,
        max_thresholds: int,
        use_mdl_correction: bool,
    ) -> tuple[int, int, float, float, float, float, float, int, float, np.ndarray]:
        ok, x_sorted, y_sorted, w_sorted, sorted_local_idx = _extract_sorted_numeric_feature_numba(
            x_feat, y_sub, weights
        )
        if ok == 0:
            return (
                0,
                -1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0.0,
                np.empty(0, dtype=np.int32),
            )

        ok2, best_pos, best_info_gain, info_gain_adj, intrinsic, best_left, best_right, count, known_weight = _find_best_binary_numeric_split_numba(
            x_sorted,
            y_sorted,
            w_sorted,
            total_weight,
            min_leaf,
            n_classes,
            max_thresholds,
            use_mdl_correction,
        )
        if ok2 == 0:
            return (
                0,
                best_pos,
                best_info_gain,
                info_gain_adj,
                intrinsic,
                best_left,
                best_right,
                count,
                known_weight,
                np.empty(0, dtype=np.int32),
            )
        return (
            ok2,
            best_pos,
            best_info_gain,
            info_gain_adj,
            intrinsic,
            best_left,
            best_right,
            count,
            known_weight,
            sorted_local_idx,
        )


    @njit(cache=True)
    def _entropy_from_counts_numba(counts: np.ndarray, total_weight: float) -> float:
        if total_weight <= 0.0:
            return 0.0
        positive = 0
        for i in range(counts.shape[0]):
            if counts[i] > 0.0:
                positive += 1
        if positive <= 1:
            return 0.0

        terms = 0.0
        for i in range(counts.shape[0]):
            c = counts[i]
            if c > 0.0:
                terms += c * np.log2(c)
        return np.log2(total_weight) - (terms / total_weight)


    @njit(cache=True)
    def _find_best_multiclass_numeric_split_numba(
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        w_sorted: np.ndarray,
        total_weight: float,
        min_leaf: float,
        n_classes: int,
        max_thresholds: int,
        use_mdl_correction: bool,
    ) -> tuple[int, int, float, float, float, float, float, int, float]:
        n = x_sorted.shape[0]
        if n <= 1:
            return (0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

        known_weight = 0.0
        feat_counts = np.zeros(n_classes, dtype=np.float64)
        for i in range(n):
            wi = w_sorted[i]
            known_weight += wi
            feat_counts[y_sorted[i]] += wi

        min_split = min(25.0, max(min_leaf, 0.1 * (known_weight / max(float(n_classes), 1.0))))
        valid_positions = np.empty(n - 1, dtype=np.int32)
        left_weights = np.empty(n - 1, dtype=np.float64)
        left_counts_store = np.zeros((n - 1, n_classes), dtype=np.float64)
        left_counts = np.zeros(n_classes, dtype=np.float64)
        count = 0
        prefix_weight = 0.0
        for i in range(n - 1):
            wi = w_sorted[i]
            prefix_weight += wi
            left_counts[y_sorted[i]] += wi
            if x_sorted[i] + 1e-5 < x_sorted[i + 1]:
                right_weight = known_weight - prefix_weight
                if prefix_weight >= min_split - 1e-12 and right_weight >= min_split - 1e-12:
                    valid_positions[count] = i
                    left_weights[count] = prefix_weight
                    for c in range(n_classes):
                        left_counts_store[count, c] = left_counts[c]
                    count += 1

        if count == 0:
            return (0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0, known_weight)

        feat_entropy = _entropy_from_counts_numba(feat_counts, known_weight)
        best_info_gain = -1.0
        best_pos = -1
        best_left = 0.0
        best_right = 0.0

        if max_thresholds >= 0 and count > max_thresholds:
            num_eval = max_thresholds
        else:
            num_eval = count

        right_counts = np.zeros(n_classes, dtype=np.float64)
        for eval_idx in range(num_eval):
            if num_eval == count:
                cand_idx = eval_idx
            elif num_eval == 1:
                cand_idx = 0
            else:
                cand_idx = int((eval_idx * (count - 1)) / (num_eval - 1))

            left_weight = left_weights[cand_idx]
            right_weight = known_weight - left_weight
            for c in range(n_classes):
                right_counts[c] = feat_counts[c] - left_counts_store[cand_idx, c]
            h_left = _entropy_from_counts_numba(left_counts_store[cand_idx], left_weight)
            h_right = _entropy_from_counts_numba(right_counts, right_weight)
            child_entropy = ((left_weight / known_weight) * h_left) + ((right_weight / known_weight) * h_right)
            info_gain = (known_weight / total_weight) * (feat_entropy - child_entropy)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_pos = valid_positions[cand_idx]
                best_left = left_weight
                best_right = right_weight

        if best_info_gain <= 0.0:
            return (0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, count, known_weight)

        if use_mdl_correction and count > 1:
            info_gain_adj = best_info_gain - (np.log2(float(count)) / total_weight)
        else:
            info_gain_adj = best_info_gain
        if info_gain_adj <= 0.0:
            return (0, -1, best_info_gain, info_gain_adj, 0.0, best_left, best_right, count, known_weight)

        missing_weight = total_weight - known_weight
        p_left = best_left / total_weight
        p_right = best_right / total_weight
        p_missing = missing_weight / total_weight if total_weight > 0.0 else 0.0
        intrinsic = 0.0
        if p_left > 0.0:
            intrinsic -= p_left * np.log2(p_left)
        if p_right > 0.0:
            intrinsic -= p_right * np.log2(p_right)
        if p_missing > 0.0:
            intrinsic -= p_missing * np.log2(p_missing)
        if intrinsic <= 0.0:
            return (0, -1, best_info_gain, info_gain_adj, intrinsic, best_left, best_right, count, known_weight)

        gain_ratio = info_gain_adj / intrinsic
        if gain_ratio <= 0.0:
            return (0, -1, best_info_gain, info_gain_adj, intrinsic, best_left, best_right, count, known_weight)

        return (1, best_pos, best_info_gain, info_gain_adj, intrinsic, best_left, best_right, count, known_weight)


    @njit(cache=True)
    def _find_best_multiclass_numeric_split_unsorted_numba(
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
        min_leaf: float,
        n_classes: int,
        max_thresholds: int,
        use_mdl_correction: bool,
    ) -> tuple[int, int, float, float, float, float, float, int, float, np.ndarray]:
        ok, x_sorted, y_sorted, w_sorted, sorted_local_idx = _extract_sorted_numeric_feature_numba(
            x_feat, y_sub, weights
        )
        if ok == 0:
            return (
                0,
                -1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0.0,
                np.empty(0, dtype=np.int32),
            )

        ok2, best_pos, best_info_gain, info_gain_adj, intrinsic, best_left, best_right, count, known_weight = _find_best_multiclass_numeric_split_numba(
            x_sorted,
            y_sorted,
            w_sorted,
            total_weight,
            min_leaf,
            n_classes,
            max_thresholds,
            use_mdl_correction,
        )
        if ok2 == 0:
            return (
                0,
                best_pos,
                best_info_gain,
                info_gain_adj,
                intrinsic,
                best_left,
                best_right,
                count,
                known_weight,
                np.empty(0, dtype=np.int32),
            )
        return (
            ok2,
            best_pos,
            best_info_gain,
            info_gain_adj,
            intrinsic,
            best_left,
            best_right,
            count,
            known_weight,
            sorted_local_idx,
        )
else:
    def _binary_entropy_scalar_numba(positive_weight: float, total_weight: float) -> float:
        raise RuntimeError("numba is not available")


    def _extract_sorted_numeric_feature_numba(
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise RuntimeError("numba is not available")


    def _find_best_binary_numeric_split_numba(
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        w_sorted: np.ndarray,
        total_weight: float,
        min_leaf: float,
        n_classes: int,
        max_thresholds: int,
        use_mdl_correction: bool,
    ) -> tuple[int, int, float, float, float, float, float, int, float]:
        raise RuntimeError("numba is not available")


    def _find_best_multiclass_numeric_split_numba(
        x_sorted: np.ndarray,
        y_sorted: np.ndarray,
        w_sorted: np.ndarray,
        total_weight: float,
        min_leaf: float,
        n_classes: int,
        max_thresholds: int,
        use_mdl_correction: bool,
    ) -> tuple[int, int, float, float, float, float, float, int, float]:
        raise RuntimeError("numba is not available")


    def _find_best_binary_numeric_split_unsorted_numba(
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
        min_leaf: float,
        n_classes: int,
        max_thresholds: int,
        use_mdl_correction: bool,
    ) -> tuple[int, int, float, float, float, float, float, int, float, np.ndarray]:
        raise RuntimeError("numba is not available")


    def _find_best_multiclass_numeric_split_unsorted_numba(
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
        min_leaf: float,
        n_classes: int,
        max_thresholds: int,
        use_mdl_correction: bool,
    ) -> tuple[int, int, float, float, float, float, float, int, float, np.ndarray]:
        raise RuntimeError("numba is not available")


def warmup_numba_numeric_kernel() -> None:
    if not NUMBA_AVAILABLE:
        return
    x = np.array([0.0, 1.0], dtype=np.float64)
    y = np.array([0, 1], dtype=np.int64)
    w = np.array([1.0, 1.0], dtype=np.float64)
    _find_best_binary_numeric_split_unsorted_numba(x, y, w, 2.0, 1.0, 2, -1, False)
    _find_best_binary_numeric_split_numba(x, y, w, 2.0, 1.0, 2, -1, False)
    x3 = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y3 = np.array([0, 1, 2], dtype=np.int64)
    w3 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    _find_best_multiclass_numeric_split_unsorted_numba(x3, y3, w3, 3.0, 1.0, 3, -1, False)
    _find_best_multiclass_numeric_split_numba(x3, y3, w3, 3.0, 1.0, 3, -1, False)


@dataclass
class _Node:
    """
    C4.5 tree node (internal node or leaf).

    Attributes:
        is_leaf: Whether this is a leaf node (True) or an internal node (False).
        prediction: Majority class at this node.
        feature_index: Feature index used for the split (internal nodes only).
        threshold: Split threshold (internal numeric nodes only).
        left: Left child (`x[feature_index] <= threshold`).
        right: Right child (`x[feature_index] > threshold`).
        class_counts: Weighted class distribution at this node.
    """
    is_leaf: bool
    prediction: Optional[Any] = None
    prediction_idx: Optional[int] = None
    feature_index: Optional[int] = None
    split_type: Optional[str] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    nominal_children: Optional[dict[Any, "_Node"]] = None
    nominal_child_probs: Optional[dict[Any, float]] = None
    nominal_default_child: Optional[Any] = None
    class_counts: Optional[np.ndarray] = None
    probability_counts: Optional[np.ndarray] = None
    missing_go_to_left: bool = True
    left_prob: float = 0.5
    train_indices: Optional[np.ndarray] = None
    train_weights: Optional[np.ndarray] = None
    split_gain_ratio: Optional[float] = None
    split_info_gain: Optional[float] = None
    split_info_gain_adj: Optional[float] = None
    split_intrinsic_value: Optional[float] = None
    split_known_weight: Optional[float] = None
    split_missing_weight: Optional[float] = None
    _cached_n_samples: Optional[float] = field(default=None, repr=False, compare=False)
    _cached_leaf_training_errors: Optional[float] = field(default=None, repr=False, compare=False)
    _cached_leaf_estimated_errors: Optional[float] = field(default=None, repr=False, compare=False)
    _cached_subtree_estimated_errors: Optional[float] = field(default=None, repr=False, compare=False)


class C45TreeClassifier:
    """
    Optimized implementation of the C4.5 algorithm (Quinlan, 1993).

    Main characteristics:
    - Uses Gain Ratio instead of Information Gain to avoid bias.
    - Handles continuous attributes through dynamic binarization.
    - Uses vectorized prediction for better performance.
    - Exposes a scikit-learn-compatible API.

    Implemented optimizations:
    - Entropy computation with bincount (about 2x faster).
    - Vectorized batch prediction (about 3-5x faster).
    - Optional cap on threshold candidates (up to 10x faster).
    - Float32 conversion to reduce memory use.
    
    Parameters
    ----------
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
        
    min_samples_leaf : int, default=1
        Minimum number of samples required in each leaf.
        
    max_depth : int, optional
        Maximum tree depth. If None, the tree grows until the leaves are pure
        or the min_samples_split/min_samples_leaf constraints stop growth.
        
    min_gain_ratio : float, default=1e-6
        Minimum gain ratio required to accept a split.
        Splits below this gain ratio are discarded early.

    use_gain_prefilter : bool, default=False
        If True, applies a C4.5-style prefilter: among the best candidate for
        each attribute, only those with adjusted information gain greater than
        or equal to the node average are allowed to compete.

    use_mdl_correction : bool, default=False
                If True, applies an approximate MDL penalty for numeric attributes
                during split evaluation:
          gain_adj = gain - log2(num_candidate_splits) / n_samples

    enable_pruning : bool, default=False
        If True, applies C4.5-style post-pruning (pessimistic/confidence-based).

    reduced_error_pruning : bool, default=False
        If True, uses reduced-error pruning with a hold-out split from the
        training data instead of confidence-based pessimistic post-pruning.

    num_folds : int, default=3
        Number of folds used by reduced-error pruning. One fold is reserved
        for pruning and the rest are used to grow the tree.

    confidence_factor : float, default=0.25
        Confidence parameter for pessimistic pruning (analogous to J48 `-C`).

    collapse_tree : bool, default=True
        If True, allows collapsing subtrees into a leaf when the leaf's
        estimated error is no worse than the subtree's error.

    enable_subtree_raising : bool, default=False
        If True, applies an approximation of subtree raising during post-pruning.

    enable_fractional_missing : bool, default=False
        If True, uses deterministic fractional propagation for missing values
        (`NaN`), in line with the weighted semantics documented by WEKA J48.

    make_split_point_actual_value : bool, default=False
        If True, uses the observed value at the split boundary as the threshold
        (the default J48 style) instead of the midpoint.

    use_laplace : bool, default=False
        If True, applies Laplace smoothing when estimating leaf class
        probabilities, analogous to the WEKA J48 `-A` option.

    nominal_features : list[int], optional
        Column indices that should be treated as nominal. For them, multi-way
        splits are created by observed value, aligned with J48's default
        nominal behavior.

    binary_splits : bool, default=False
        If True, nominal attributes are evaluated with J48 `-B`-style binary
        splits, using the best one-vs-rest partition per value.

    auto_detect_nominal : bool, default=False
        If True, attempts to infer nominal columns from non-numeric values when
        they are not declared explicitly in `nominal_features`.

    nominal_value_domains : dict, optional
        Explicit nominal domains per attribute. Keys may be column indices or
        attribute names, and values must be the ordered list of allowed
        categories. This helps reproduce ARFF-like datasets where the complete
        nominal domain exists even if not all values appear in the training
        subset.

    feature_names : list[str], optional
        Column names used for traces and tree export.

    max_thresholds : int, optional, default=100
        Maximum number of threshold candidates to evaluate per feature.
        If None, evaluates all midpoints (closer to original C4.5, but slower).
        Typical values are 50-200 for a speed/precision balance.
        
    random_state : int, optional
        Random seed for reproducibility.

    cleanup : bool, default=True
        If True, removes references to the training set once the tree has been
        built and pruned, emulating J48's default behavior when ``-L`` is not
        used.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique classes seen in the training data.
        
    n_classes_ : int
        Number of classes.
        
    n_features_ : int
        Number of features in the training data.
        
    root_ : _Node
        Root node of the trained tree.
    
    Examples
    --------
    >>> from EvLib.c45 import C45TreeClassifier
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1]])
    >>> y = np.array([0, 1])
    >>> clf = C45TreeClassifier(random_state=42)
    >>> clf.fit(X, y)
    >>> clf.predict([[0.5, 0.5]])
    array([0])
    
    References
    ----------
    Quinlan, J. R. (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann.
    """

    def __init__(
        self,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: Optional[int] = None,
        min_gain_ratio: float = 1e-6,
        use_gain_prefilter: bool = False,
        use_mdl_correction: bool = False,
        enable_pruning: bool = False,
        reduced_error_pruning: bool = False,
        num_folds: int = 3,
        confidence_factor: float = 0.25,
        collapse_tree: bool = True,
        enable_subtree_raising: bool = False,
        enable_fractional_missing: bool = False,
        make_split_point_actual_value: bool = False,
        use_laplace: bool = False,
        nominal_features: Optional[list[int]] = None,
        binary_splits: bool = False,
        auto_detect_nominal: bool = False,
        nominal_value_domains: Optional[dict[Any, list[Any]]] = None,
        feature_names: Optional[list[str]] = None,
        max_thresholds: Optional[int] = 100,
        random_state: Optional[int] = None,
        cleanup: bool = True,
        split_debug_target_path: Optional[list[str]] = None,
        gain_prefilter_slack: float = 1e-3,
        use_numba_numeric_kernel: bool = False,
    ):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_gain_ratio = min_gain_ratio
        self.use_gain_prefilter = bool(use_gain_prefilter)
        self.use_mdl_correction = bool(use_mdl_correction)
        self.enable_pruning = bool(enable_pruning)
        self.reduced_error_pruning = bool(reduced_error_pruning)
        self.num_folds = int(num_folds)
        self.confidence_factor = float(confidence_factor)
        self.collapse_tree = bool(collapse_tree)
        self.enable_subtree_raising = bool(enable_subtree_raising)
        self.enable_fractional_missing = bool(enable_fractional_missing)
        self.make_split_point_actual_value = bool(make_split_point_actual_value)
        self.use_laplace = bool(use_laplace)
        self.nominal_features = None if nominal_features is None else [int(i) for i in nominal_features]
        self.binary_splits = bool(binary_splits)
        self.auto_detect_nominal = bool(auto_detect_nominal)
        self.nominal_value_domains = None if nominal_value_domains is None else dict(nominal_value_domains)
        self.feature_names = None if feature_names is None else [str(v) for v in feature_names]
        self.max_thresholds = max_thresholds
        self.random_state = random_state
        self.cleanup = bool(cleanup)
        self.split_debug_target_path = (
            None if split_debug_target_path is None else [str(v) for v in split_debug_target_path]
        )
        self.gain_prefilter_slack = float(gain_prefilter_slack)
        self.use_numba_numeric_kernel = bool(use_numba_numeric_kernel)

        # Atributos a llenar durante fit()
        self.n_classes_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.root_: Optional[_Node] = None
        self._log2_lut_: Optional[np.ndarray] = None
        self._rng_: Optional[np.random.RandomState] = None
        self._train_X_: Optional[np.ndarray] = None
        self._train_y_encoded_: Optional[np.ndarray] = None
        self._nominal_features_: set[int] = set()
        self._nominal_value_domains_: dict[int, list[Any]] = {}
        self._feature_names_: Optional[list[str]] = None
        self._split_debug_trace_: list[dict[str, Any]] = []
        self._relocate_values_cache_: dict[int, np.ndarray] = {}

    def _infer_nominal_features(self, X: np.ndarray) -> set[int]:
        nominal = set(self.nominal_features or [])
        if not self.auto_detect_nominal:
            return nominal

        n_features = X.shape[1]
        for feat in range(n_features):
            if feat in nominal:
                continue
            col = np.asarray(X[:, feat], dtype=object)
            non_missing = [v for v in col.tolist() if not _is_missing_scalar(v)]
            if not non_missing:
                continue
            try:
                np.asarray([_to_python_scalar(v) for v in non_missing], dtype=np.float64)
            except Exception:
                nominal.add(feat)
        return nominal

    def _is_nominal_feature(self, feature_index: int) -> bool:
        return int(feature_index) in self._nominal_features_

    def _collect_nominal_value_domains(self, X: np.ndarray) -> dict[int, list[Any]]:
        domains: dict[int, list[Any]] = {}
        for feat in sorted(int(v) for v in self._nominal_features_):
            col = np.asarray(X[:, feat])
            if col.dtype != object:
                if np.issubdtype(col.dtype, np.floating):
                    valid = col[np.isfinite(col)]
                else:
                    valid = col
                values = [self._normalize_nominal_value(v) for v in valid.tolist()]
            else:
                obj_col = np.asarray(col, dtype=object)
                values = [
                    self._normalize_nominal_value(v)
                    for v in obj_col.tolist()
                    if not _is_missing_scalar(v)
                ]
            domains[feat] = list(dict.fromkeys(values))
        return domains

    def _resolve_nominal_value_domains(self, X: np.ndarray) -> dict[int, list[Any]]:
        observed_domains = self._collect_nominal_value_domains(X)
        if not self.nominal_value_domains:
            return observed_domains

        resolved = {int(feat): list(values) for feat, values in observed_domains.items()}
        for raw_key, raw_values in self.nominal_value_domains.items():
            if isinstance(raw_key, str):
                if self._feature_names_ is None or raw_key not in self._feature_names_:
                    raise ValueError(f"unknown nominal domain feature name: {raw_key}")
                feat = int(self._feature_names_.index(raw_key))
            else:
                feat = int(raw_key)

            if feat not in self._nominal_features_:
                raise ValueError(
                    f"nominal_value_domains specified for non-nominal feature index {feat}"
                )

            normalized = [
                self._normalize_nominal_value(v)
                for v in raw_values
                if not _is_missing_scalar(v)
            ]
            merged_values = list(dict.fromkeys(normalized))
            for observed in observed_domains.get(feat, []):
                if observed not in merged_values:
                    merged_values.append(observed)
            resolved[feat] = merged_values
        return resolved

    def _nominal_domain_values(
        self,
        feature_index: int,
        observed_values: list[Any],
    ) -> list[Any]:
        domain = list(self._nominal_value_domains_.get(int(feature_index), []))
        if not domain:
            return list(dict.fromkeys(observed_values))
        observed_set = set(observed_values)
        for value in observed_values:
            if value not in domain:
                domain.append(value)
        return domain

    def _coerce_numeric_column(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float64, copy=False)
        if np.issubdtype(arr.dtype, np.integer):
            return arr.astype(np.float64, copy=False)

        out = np.empty(arr.shape[0], dtype=np.float64)
        obj_arr = np.asarray(arr, dtype=object)
        for i, value in enumerate(obj_arr.tolist()):
            value = _to_python_scalar(value)
            if _is_missing_scalar(value):
                out[i] = np.nan
                continue
            try:
                out[i] = float(value)
            except Exception:
                out[i] = np.nan
        return out

    def _feature_missing_mask(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.floating):
            return ~np.isfinite(arr.astype(np.float64, copy=False))
        if np.issubdtype(arr.dtype, np.integer):
            return np.zeros(arr.shape[0], dtype=bool)
        obj_arr = np.asarray(arr, dtype=object)
        return np.array([_is_missing_scalar(v) for v in obj_arr.tolist()], dtype=bool)

    def _normalize_nominal_value(self, value: Any) -> Any:
        return _to_python_scalar(value)

    def _nominal_match_mask(self, values: np.ndarray, target_value: Any) -> np.ndarray:
        arr = np.asarray(values)
        if arr.dtype != object:
            try:
                return arr == target_value
            except Exception:
                pass
        obj_arr = np.asarray(arr, dtype=object)
        return np.array(
            [self._normalize_nominal_value(v) == target_value for v in obj_arr.tolist()],
            dtype=bool,
        )

    def _matrix_has_missing(self, X: np.ndarray) -> bool:
        X_arr = np.asarray(X)
        if np.issubdtype(X_arr.dtype, np.floating):
            return bool(np.any(~np.isfinite(X_arr)))
        if np.issubdtype(X_arr.dtype, np.integer):
            return False
        for feat in range(X_arr.shape[1]):
            if np.any(self._feature_missing_mask(X_arr[:, feat])):
                return True
        return False

    def _matrix_can_stay_numeric(self, X: np.ndarray, nominal_feature_hint: bool) -> bool:
        arr = np.asarray(X)
        if arr.dtype == object:
            return False
        if np.issubdtype(arr.dtype, np.floating):
            return True
        if np.issubdtype(arr.dtype, np.integer):
            return True
        return not nominal_feature_hint

    def _feature_name(self, feature_index: Optional[int]) -> Optional[str]:
        if feature_index is None:
            return None
        if self._feature_names_ is None:
            return f"f{int(feature_index)}"
        if 0 <= int(feature_index) < len(self._feature_names_):
            return self._feature_names_[int(feature_index)]
        return f"f{int(feature_index)}"

    def _split_debug_enabled_for_path(self, current_path: list[str]) -> bool:
        if self.split_debug_target_path is None:
            return False
        return list(current_path) == list(self.split_debug_target_path)

    def _serialize_split_debug_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "feature_index": int(candidate["feature"]),
            "feature_name": self._feature_name(int(candidate["feature"])),
            "split_type": str(candidate["split_type"]),
            "gain_ratio": float(candidate["gain_ratio"]),
            "info_gain": float(candidate["info_gain"]),
            "info_gain_adj": float(candidate["info_gain_adj"]),
            "intrinsic": float(candidate["intrinsic"]),
            "known_weight": float(candidate["known_weight"]),
            "missing_weight": float(candidate["missing_weight"]),
        }
        if candidate.get("threshold") is not None:
            payload["threshold"] = float(candidate["threshold"])
        if "values" in candidate:
            payload["values"] = [_to_jsonable(v) for v in candidate["values"]]
            payload["default_child"] = _to_jsonable(candidate.get("default_child"))
        return payload

    def _maybe_record_split_debug(
        self,
        current_path: list[str],
        depth: int,
        class_counts: np.ndarray,
        total_weight: float,
        candidates_before_prefilter: list[dict[str, Any]],
        candidates_after_prefilter: list[dict[str, Any]],
        mean_gain: Optional[float],
        selected_candidate: dict[str, Any],
    ) -> None:
        if not self._split_debug_enabled_for_path(current_path):
            return
        self._split_debug_trace_.append(
            {
                "path_conditions": list(current_path),
                "depth": int(depth),
                "total_weight": float(total_weight),
                "class_counts": [float(v) for v in np.asarray(class_counts, dtype=np.float64).tolist()],
                "mean_gain": None if mean_gain is None else float(mean_gain),
                "candidates_before_prefilter": [
                    self._serialize_split_debug_candidate(c) for c in candidates_before_prefilter
                ],
                "candidates_after_prefilter": [
                    self._serialize_split_debug_candidate(c) for c in candidates_after_prefilter
                ],
                "selected_candidate": self._serialize_split_debug_candidate(selected_candidate),
            }
        )

    def _iter_child_items(self, node: _Node) -> list[tuple[Any, _Node]]:
        if node.is_leaf:
            return []
        if node.split_type == "nominal":
            if not node.nominal_children:
                return []
            return [(k, child) for k, child in node.nominal_children.items() if child is not None]
        items = []
        if node.left is not None:
            items.append(("left", node.left))
        if node.right is not None:
            items.append(("right", node.right))
        return items

    def _concat_weighted_parts(
        self,
        indices_parts: list[np.ndarray],
        weight_parts: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not indices_parts:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
        return (
            np.concatenate(indices_parts).astype(np.int32, copy=False),
            np.concatenate(weight_parts).astype(np.float64, copy=False),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "C45TreeClassifier":
        """
        Build the decision tree from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        y : array-like of shape (n_samples,)
            Target labels.

        sample_weight : array-like of shape (n_samples,), optional
            Per-instance weights. When provided, the tree is trained using
            those weights in all count, entropy, and pruning computations.

        Returns
        -------
        self : C45TreeClassifier
            Fitted classifier.
        """
        X = np.asarray(X)
        nominal_hint = bool(self.nominal_features) or bool(self.auto_detect_nominal)
        if not self._matrix_can_stay_numeric(X, nominal_hint):
            X = np.asarray(X, dtype=object)
        elif not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float32, copy=False)
        y = np.asarray(y)

        self._rng_ = np.random.RandomState(self.random_state)
        self._split_debug_trace_ = []
        self._relocate_values_cache_ = {}

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self._train_X_ = X
        self._nominal_features_ = self._infer_nominal_features(X)
        if self.feature_names is not None and len(self.feature_names) == n_features:
            self._feature_names_ = list(self.feature_names)
        else:
            self._feature_names_ = [f"f{i}" for i in range(n_features)]
        self._nominal_value_domains_ = self._resolve_nominal_value_domains(X)
        self._log2_lut_ = np.zeros(n_samples + 1, dtype=np.float64)
        if n_samples > 0:
            self._log2_lut_[1:] = np.log2(np.arange(1, n_samples + 1, dtype=np.float64))

        # Encode classes as consecutive integers 0, 1, 2, ...
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.classes_ = np.asarray([_to_python_scalar(v) for v in self.classes_.tolist()], dtype=object)
        self.n_classes_ = self.classes_.shape[0]
        self._train_y_encoded_ = y_encoded

        if sample_weight is None:
            weights = np.ones(n_samples, dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.ndim != 1 or weights.shape[0] != n_samples:
                raise ValueError("sample_weight must have shape (n_samples,)")
            if np.any(~np.isfinite(weights)):
                raise ValueError("sample_weight must contain only finite values")
            if np.any(weights < 0.0):
                raise ValueError("sample_weight must be non-negative")

        if self.reduced_error_pruning:
            if self.num_folds < 2:
                raise ValueError("num_folds must be >= 2 when reduced_error_pruning=True")
            if n_samples < self.num_folds:
                raise ValueError("num_folds must be <= n_samples when reduced_error_pruning=True")
            self.root_ = self._fit_with_reduced_error_pruning(X, y_encoded, weights)
        else:
            indices = np.arange(n_samples, dtype=np.int32)
            self.root_ = self._build_tree(X, y_encoded, indices, weights, depth=0, path_conditions=[])
            if self.enable_pruning and self.root_ is not None:
                if self.collapse_tree:
                    self.root_ = self._collapse_subtree(self.root_)
                self.root_ = self._prune_tree(self.root_)

        if self.cleanup:
            self._cleanup_after_fit()

        logger.debug(f"C4.5 tree built with max_thresholds={self.max_thresholds}")
        
        return self

    def _make_leaf_node(
        self,
        prediction: Any,
        prediction_idx: int,
        class_counts: np.ndarray,
        probability_counts: Optional[np.ndarray] = None,
        train_indices: Optional[np.ndarray] = None,
        train_weights: Optional[np.ndarray] = None,
    ) -> _Node:
        return _Node(
            is_leaf=True,
            prediction=prediction,
            prediction_idx=prediction_idx,
            class_counts=class_counts,
            probability_counts=None if probability_counts is None else probability_counts.copy(),
            train_indices=None if train_indices is None else train_indices.copy(),
            train_weights=None if train_weights is None else train_weights.copy(),
        )

    def _find_best_numeric_split_candidate(
        self,
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
        feat: int,
    ) -> Optional[dict[str, Any]]:
        if self.use_numba_numeric_kernel and NUMBA_AVAILABLE:
            max_thresholds = -1 if self.max_thresholds is None else int(self.max_thresholds)
            x_feat_f64 = np.asarray(x_feat, dtype=np.float64)
            y_sub_i64 = y_sub.astype(np.int64, copy=False)
            w_f64 = np.asarray(weights, dtype=np.float64)
            if self.n_classes_ == 2:
                kernel_result = _find_best_binary_numeric_split_unsorted_numba(
                    x_feat_f64,
                    y_sub_i64,
                    w_f64,
                    float(total_weight),
                    float(self.min_samples_leaf),
                    int(self.n_classes_),
                    max_thresholds,
                    bool(self.use_mdl_correction),
                )
            else:
                kernel_result = _find_best_multiclass_numeric_split_unsorted_numba(
                    x_feat_f64,
                    y_sub_i64,
                    w_f64,
                    float(total_weight),
                    float(self.min_samples_leaf),
                    int(self.n_classes_),
                    max_thresholds,
                    bool(self.use_mdl_correction),
                )
            ok, best_pos_numba, best_info_gain, best_info_gain_adj, intrinsic, best_left_weight, best_right_weight, total_candidate_points, known_weight_numba, sorted_local_idx = kernel_result
            if ok == 0:
                return None
            pos = int(best_pos_numba)
            left_value = float(x_feat_f64[int(sorted_local_idx[pos])])
            right_value = float(x_feat_f64[int(sorted_local_idx[pos + 1])])
            midpoint = float((left_value + right_value) * 0.5)
            threshold = self._relocate_split_point(feat, midpoint) if self.make_split_point_actual_value else midpoint
            missing_weight = max(total_weight - float(known_weight_numba), 0.0)
            best_local_gr = (float(best_info_gain_adj) / float(intrinsic)) if intrinsic > 0.0 else 0.0
            if best_local_gr <= 0.0:
                return None
            return {
                "split_type": "numeric",
                "gain_ratio": float(best_local_gr),
                "info_gain": float(best_info_gain),
                "info_gain_adj": float(best_info_gain_adj),
                "intrinsic": float(intrinsic),
                "balance": float(min(best_left_weight, best_right_weight)),
                "feature": int(feat),
                "threshold": float(threshold),
                "split_pos": pos,
                "known_weight": float(known_weight_numba),
                "missing_weight": float(missing_weight),
                "left_weight": float(best_left_weight),
                "right_weight": float(best_right_weight),
                "candidate_split_count": int(total_candidate_points),
                "sorted_local_idx": np.asarray(sorted_local_idx, dtype=np.int32),
            }

        valid_mask = np.isfinite(x_feat)
        if not np.any(valid_mask):
            return None

        known_local_idx = np.flatnonzero(valid_mask)
        if known_local_idx.size == 0:
            return None
        x_valid = x_feat[known_local_idx]
        order = np.argsort(x_valid, kind="mergesort")
        sorted_local_idx = known_local_idx[order]
        return self._evaluate_numeric_split_candidate_sorted(
            x_feat=x_feat,
            y_sub=y_sub,
            weights=weights,
            total_weight=total_weight,
            feat=feat,
            sorted_local_idx=sorted_local_idx,
        )

    def _evaluate_numeric_split_candidate_sorted(
        self,
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
        feat: int,
        sorted_local_idx: np.ndarray,
    ) -> Optional[dict[str, Any]]:
        x_sorted = x_feat[sorted_local_idx]
        y_sorted = y_sub[sorted_local_idx]
        w_sorted = weights[sorted_local_idx].astype(np.float64, copy=False)
        known_weight = float(np.sum(w_sorted))
        if known_weight < max(2.0, 2.0 * float(self.min_samples_leaf)) - 1e-12:
            return None

        if self.use_numba_numeric_kernel and NUMBA_AVAILABLE:
            max_thresholds = -1 if self.max_thresholds is None else int(self.max_thresholds)
            y_sorted_i64 = y_sorted.astype(np.int64, copy=False)
            if self.n_classes_ == 2:
                kernel_result = _find_best_binary_numeric_split_numba(
                    x_sorted,
                    y_sorted_i64,
                    w_sorted,
                    float(total_weight),
                    float(self.min_samples_leaf),
                    int(self.n_classes_),
                    max_thresholds,
                    bool(self.use_mdl_correction),
                )
            else:
                kernel_result = _find_best_multiclass_numeric_split_numba(
                    x_sorted,
                    y_sorted_i64,
                    w_sorted,
                    float(total_weight),
                    float(self.min_samples_leaf),
                    int(self.n_classes_),
                    max_thresholds,
                    bool(self.use_mdl_correction),
                )
            ok, best_pos_numba, best_info_gain, best_info_gain_adj, intrinsic, best_left_weight, best_right_weight, total_candidate_points, known_weight_numba = kernel_result
            if ok == 0:
                return None
            pos = int(best_pos_numba)
            midpoint = float((x_sorted[pos] + x_sorted[pos + 1]) * 0.5)
            threshold = self._relocate_split_point(feat, midpoint) if self.make_split_point_actual_value else midpoint
            missing_weight = max(total_weight - float(known_weight_numba), 0.0)
            best_local_gr = (float(best_info_gain_adj) / float(intrinsic)) if intrinsic > 0.0 else 0.0
            if best_local_gr <= 0.0:
                return None
            return {
                "split_type": "numeric",
                "gain_ratio": float(best_local_gr),
                "info_gain": float(best_info_gain),
                "info_gain_adj": float(best_info_gain_adj),
                "intrinsic": float(intrinsic),
                "balance": float(min(best_left_weight, best_right_weight)),
                "feature": int(feat),
                "threshold": float(threshold),
                "split_pos": pos,
                "known_weight": float(known_weight_numba),
                "missing_weight": float(missing_weight),
                "left_weight": float(best_left_weight),
                "right_weight": float(best_right_weight),
                "candidate_split_count": int(total_candidate_points),
                "sorted_local_idx": sorted_local_idx,
            }

        split_positions = np.where(x_sorted[:-1] + 1e-5 < x_sorted[1:])[0]
        if split_positions.size == 0:
            return None

        min_leaf = float(self.min_samples_leaf)
        min_split = min(
            25.0,
            max(
                min_leaf,
                0.1 * (known_weight / max(float(self.n_classes_ or 1), 1.0)),
            ),
        )
        weight_prefix = np.cumsum(w_sorted, dtype=np.float64)
        left_weight = weight_prefix[split_positions]
        right_weight = known_weight - left_weight
        valid = (left_weight >= min_split - 1e-12) & (right_weight >= min_split - 1e-12)
        split_positions = split_positions[valid]
        if split_positions.size == 0:
            return None
        left_weight = left_weight[valid]
        right_weight = right_weight[valid]
        total_candidate_points = int(split_positions.size)

        if self.max_thresholds is not None and split_positions.size > self.max_thresholds:
            sel = np.linspace(0, split_positions.size - 1, self.max_thresholds, dtype=int)
            split_positions = split_positions[sel]
            left_weight = left_weight[sel]
            right_weight = right_weight[sel]

        if self.n_classes_ == 2:
            pos_prefix = np.cumsum(np.where(y_sorted == 1, w_sorted, 0.0), dtype=np.float64)
            total_pos = float(pos_prefix[-1])
            left_pos = pos_prefix[split_positions]
            right_pos = total_pos - left_pos
            h_left = _binary_entropy_from_positive_weight(left_pos, left_weight)
            h_right = _binary_entropy_from_positive_weight(right_pos, right_weight)
            feat_counts = np.array([known_weight - total_pos, total_pos], dtype=np.float64)
        else:
            sorted_onehot = np.zeros((x_sorted.size, self.n_classes_), dtype=np.float64)
            sorted_onehot[np.arange(x_sorted.size), y_sorted] = w_sorted
            left_prefix = np.cumsum(sorted_onehot, axis=0, dtype=np.float64)
            left_counts_mat = left_prefix[split_positions]
            feat_counts = left_prefix[-1]
            right_counts_mat = feat_counts[None, :] - left_counts_mat
            h_left = _entropy_from_weighted_counts_matrix(left_counts_mat)
            h_right = _entropy_from_weighted_counts_matrix(right_counts_mat)

        feat_entropy = _entropy_from_weighted_counts(feat_counts)
        child_entropy = (
            (left_weight / known_weight) * h_left
            + (right_weight / known_weight) * h_right
        )
        info_gain = (known_weight / total_weight) * (feat_entropy - child_entropy)
        best_local_idx = int(np.argmax(info_gain))
        best_info_gain = float(info_gain[best_local_idx])
        if best_info_gain <= 0.0:
            return None

        if self.use_mdl_correction and total_candidate_points > 1:
            mdl_penalty = np.log2(float(total_candidate_points)) / total_weight
            best_info_gain_adj = best_info_gain - mdl_penalty
        else:
            best_info_gain_adj = best_info_gain
        if best_info_gain_adj <= 0.0:
            return None

        missing_weight = max(total_weight - known_weight, 0.0)
        best_left_weight = float(left_weight[best_local_idx])
        best_right_weight = float(right_weight[best_local_idx])
        p_left = best_left_weight / total_weight
        p_right = best_right_weight / total_weight
        p_missing = missing_weight / total_weight if total_weight > 0.0 else 0.0
        intrinsic = 0.0
        if p_left > 0.0:
            intrinsic -= p_left * np.log2(p_left)
        if p_right > 0.0:
            intrinsic -= p_right * np.log2(p_right)
        if p_missing > 0.0:
            intrinsic -= p_missing * np.log2(p_missing)
        best_local_gr = (best_info_gain_adj / intrinsic) if intrinsic > 0.0 else 0.0
        if best_local_gr <= 0.0:
            return None

        pos = int(split_positions[best_local_idx])
        midpoint = float((x_sorted[pos] + x_sorted[pos + 1]) * 0.5)
        threshold = self._relocate_split_point(feat, midpoint) if self.make_split_point_actual_value else midpoint
        return {
            "split_type": "numeric",
            "gain_ratio": best_local_gr,
            "info_gain": best_info_gain,
            "info_gain_adj": best_info_gain_adj,
            "intrinsic": float(intrinsic),
            "balance": float(min(best_left_weight, best_right_weight)),
            "feature": int(feat),
            "threshold": float(threshold),
            "split_pos": pos,
            "known_weight": known_weight,
            "missing_weight": missing_weight,
            "left_weight": best_left_weight,
            "right_weight": best_right_weight,
            "candidate_split_count": total_candidate_points,
            "sorted_local_idx": sorted_local_idx,
        }

    def _find_best_nominal_split_candidate(
        self,
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
        feat: int,
    ) -> Optional[dict[str, Any]]:
        missing_mask = self._feature_missing_mask(x_feat)
        valid_mask = ~missing_mask
        if not np.any(valid_mask):
            return None

        x_valid = np.asarray(x_feat[valid_mask])
        y_valid = y_sub[valid_mask]
        w_valid = weights[valid_mask]
        known_weight = float(np.sum(w_valid))
        if known_weight < max(2.0, 2.0 * float(self.min_samples_leaf)) - 1e-12:
            return None

        observed_values = list(
            dict.fromkeys(self._normalize_nominal_value(v) for v in x_valid.tolist())
        )
        value_order = self._nominal_domain_values(feat, observed_values)
        if len(value_order) <= 1:
            return None

        branch_values: list[Any] = []
        branch_weights: list[float] = []
        branch_counts: list[np.ndarray] = []
        min_leaf = float(self.min_samples_leaf)
        for value in value_order:
            branch_mask = self._nominal_match_mask(x_valid, value)
            branch_values.append(value)
            if not np.any(branch_mask):
                branch_weights.append(0.0)
                branch_counts.append(np.zeros(self.n_classes_, dtype=np.float64))
                continue
            branch_weight = float(np.sum(w_valid[branch_mask]))
            if branch_weight <= 1e-12:
                branch_weights.append(0.0)
                branch_counts.append(np.zeros(self.n_classes_, dtype=np.float64))
                continue
            branch_weights.append(branch_weight)
            branch_counts.append(
                np.bincount(
                    y_valid[branch_mask],
                    weights=w_valid[branch_mask],
                    minlength=self.n_classes_,
                ).astype(np.float64, copy=False)
            )

        if len(branch_values) <= 1:
            return None

        branch_weights_arr = np.asarray(branch_weights, dtype=np.float64)
        if np.count_nonzero(branch_weights_arr >= min_leaf - 1e-12) < 2:
            return None

        feat_counts = np.bincount(
            y_valid, weights=w_valid, minlength=self.n_classes_
        ).astype(np.float64, copy=False)
        feat_entropy = _entropy_from_weighted_counts(feat_counts)
        child_entropy = 0.0
        for branch_weight, counts in zip(branch_weights_arr, branch_counts):
            child_entropy += (branch_weight / known_weight) * _entropy_from_weighted_counts(counts)

        info_gain = (known_weight / total_weight) * (feat_entropy - child_entropy)
        missing_weight = max(total_weight - known_weight, 0.0)
        intrinsic = 0.0
        for prob in (branch_weights_arr / total_weight):
            if prob > 0.0:
                intrinsic -= float(prob * np.log2(prob))
        if missing_weight > 0.0 and total_weight > 0.0:
            p_missing = missing_weight / total_weight
            intrinsic -= float(p_missing * np.log2(p_missing))

        gain_ratio = (info_gain / intrinsic) if intrinsic > 0.0 else 0.0
        if gain_ratio <= 0.0:
            return None

        probs = branch_weights_arr / float(np.sum(branch_weights_arr))
        branch_prob_map = {
            branch_values[i]: float(probs[i]) for i in range(len(branch_values))
        }
        default_child = branch_values[int(np.argmax(branch_weights_arr))]
        return {
            "split_type": "nominal",
            "gain_ratio": float(gain_ratio),
            "info_gain": float(info_gain),
            "info_gain_adj": float(info_gain),
            "intrinsic": float(intrinsic),
            "balance": float(np.min(branch_weights_arr)),
            "feature": int(feat),
            "values": branch_values,
            "branch_probs": branch_prob_map,
            "default_child": default_child,
            "known_weight": known_weight,
            "missing_weight": missing_weight,
        }

    def _find_best_binary_nominal_split_candidate(
        self,
        x_feat: np.ndarray,
        y_sub: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
        feat: int,
    ) -> Optional[dict[str, Any]]:
        missing_mask = self._feature_missing_mask(x_feat)
        valid_mask = ~missing_mask
        if not np.any(valid_mask):
            return None

        x_valid = np.asarray(x_feat[valid_mask])
        y_valid = y_sub[valid_mask]
        w_valid = weights[valid_mask]
        known_weight = float(np.sum(w_valid))
        if known_weight < max(2.0, 2.0 * float(self.min_samples_leaf)) - 1e-12:
            return None

        observed_values = list(
            dict.fromkeys(self._normalize_nominal_value(v) for v in x_valid.tolist())
        )
        value_order = self._nominal_domain_values(feat, observed_values)
        if len(value_order) <= 1:
            return None

        min_leaf = float(self.min_samples_leaf)
        feat_counts = np.bincount(
            y_valid, weights=w_valid, minlength=self.n_classes_
        ).astype(np.float64, copy=False)
        feat_entropy = _entropy_from_weighted_counts(feat_counts)
        missing_weight = max(total_weight - known_weight, 0.0)

        best_candidate: Optional[dict[str, Any]] = None
        best_rank: Optional[tuple[float, float, float, int]] = None

        for value_idx, value in enumerate(value_order):
            branch_mask = self._nominal_match_mask(x_valid, value)
            if not np.any(branch_mask) or np.all(branch_mask):
                continue

            pos_weight = float(np.sum(w_valid[branch_mask]))
            neg_weight = known_weight - pos_weight
            if pos_weight < min_leaf - 1e-12 or neg_weight < min_leaf - 1e-12:
                continue

            pos_counts = np.bincount(
                y_valid[branch_mask],
                weights=w_valid[branch_mask],
                minlength=self.n_classes_,
            ).astype(np.float64, copy=False)
            neg_counts = feat_counts - pos_counts

            child_entropy = (
                (pos_weight / known_weight) * _entropy_from_weighted_counts(pos_counts)
                + (neg_weight / known_weight) * _entropy_from_weighted_counts(neg_counts)
            )
            info_gain = (known_weight / total_weight) * (feat_entropy - child_entropy)

            intrinsic = 0.0
            for prob in (pos_weight / total_weight, neg_weight / total_weight):
                if prob > 0.0:
                    intrinsic -= float(prob * np.log2(prob))
            if missing_weight > 0.0 and total_weight > 0.0:
                p_missing = missing_weight / total_weight
                intrinsic -= float(p_missing * np.log2(p_missing))

            gain_ratio = (info_gain / intrinsic) if intrinsic > 0.0 else 0.0
            if gain_ratio <= 0.0:
                continue

            candidate = {
                "split_type": "nominal",
                "gain_ratio": float(gain_ratio),
                "info_gain": float(info_gain),
                "info_gain_adj": float(info_gain),
                "intrinsic": float(intrinsic),
                "balance": float(min(pos_weight, neg_weight)),
                "feature": int(feat),
                "values": [value, _NOMINAL_OTHER_BRANCH],
                "branch_probs": {
                    value: float(pos_weight / known_weight),
                    _NOMINAL_OTHER_BRANCH: float(neg_weight / known_weight),
                },
                "default_child": _NOMINAL_OTHER_BRANCH,
                "known_weight": known_weight,
                "missing_weight": missing_weight,
                "binary_nominal_value": value,
            }
            rank = (
                float(candidate["gain_ratio"]),
                float(candidate["info_gain_adj"]),
                float(candidate["balance"]),
                -int(value_idx),
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_candidate = candidate

        return best_candidate

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
        depth: int,
        path_conditions: Optional[list[str]] = None,
    ) -> _Node:
        """
        Recursively build a C4.5 subtree.

        Parameters
        ----------
        X : ndarray
            Full feature matrix.
        y : ndarray
            Full label vector (integer-encoded).
        indices : ndarray
            Indices of the samples that belong to this node.
        depth : int
            Current tree depth.

        Returns
        -------
        node : _Node
            Constructed node (leaf or internal node).
        """
        current_path = list(path_conditions or [])

        if indices.size == 0:
            empty_counts = np.zeros(self.n_classes_, dtype=np.float64)
            return self._make_leaf_node(
                prediction=self.classes_[0],
                prediction_idx=0,
                class_counts=empty_counts,
                train_indices=np.array([], dtype=np.int32),
                train_weights=np.array([], dtype=np.float64),
            )

        weights = np.asarray(weights, dtype=np.float64)
        keep_mask = weights > 1e-12
        indices = indices[keep_mask]
        weights = weights[keep_mask]

        if indices.size == 0:
            empty_counts = np.zeros(self.n_classes_, dtype=np.float64)
            return self._make_leaf_node(
                prediction=self.classes_[0],
                prediction_idx=0,
                class_counts=empty_counts,
                train_indices=np.array([], dtype=np.int32),
                train_weights=np.array([], dtype=np.float64),
            )

        y_sub = y[indices]
        total_weight = float(np.sum(weights))

        # Count weighted classes at this node.
        counts = np.bincount(y_sub, weights=weights, minlength=self.n_classes_).astype(np.float64, copy=False)
        pred_class_idx = int(np.argmax(counts))
        prediction = self.classes_[pred_class_idx]

        # --- Stopping conditions ---

        # 1. Pure node (single class).
        if np.count_nonzero(counts > 1e-12) == 1:
            return self._make_leaf_node(
                prediction=prediction,
                prediction_idx=pred_class_idx,
                class_counts=counts,
                train_indices=indices,
                train_weights=weights,
            )

        # 2. Too few samples to split.
        if total_weight < float(self.min_samples_split) - 1e-12:
            return self._make_leaf_node(
                prediction=prediction,
                prediction_idx=pred_class_idx,
                class_counts=counts,
                train_indices=indices,
                train_weights=weights,
            )

        # 3. Maximum depth reached.
        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf_node(
                prediction=prediction,
                prediction_idx=pred_class_idx,
                class_counts=counts,
                train_indices=indices,
                train_weights=weights,
            )

        # --- Search for the best split according to Gain Ratio. ---

        split_candidates = []

        for feat in range(self.n_features_):
            raw_feat = X[indices, feat]
            if self._is_nominal_feature(feat):
                if self.binary_splits:
                    candidate = self._find_best_binary_nominal_split_candidate(
                        raw_feat, y_sub, weights, total_weight, feat
                    )
                else:
                    candidate = self._find_best_nominal_split_candidate(
                        raw_feat, y_sub, weights, total_weight, feat
                    )
            else:
                numeric_feat = self._coerce_numeric_column(raw_feat)
                candidate = self._find_best_numeric_split_candidate(
                    numeric_feat, y_sub, weights, total_weight, feat
                )
            if candidate is not None:
                split_candidates.append(candidate)

        if not split_candidates:
            return self._make_leaf_node(
                prediction=prediction,
                prediction_idx=pred_class_idx,
                class_counts=counts,
                train_indices=indices,
                train_weights=weights,
            )

        # C4.5 prefilter: keep only candidates with gain >= average node gain.
        prefilter_mean_gain: Optional[float] = None
        prefilter_candidates = list(split_candidates)
        if self.use_gain_prefilter:
            gains = [c["info_gain_adj"] for c in split_candidates]
            if gains:
                prefilter_mean_gain = float(np.mean(gains))
                eligible = [
                    c
                    for c in split_candidates
                    if c["info_gain_adj"] >= (prefilter_mean_gain - self.gain_prefilter_slack)
                ]
                if eligible:
                    split_candidates = eligible

        # Final selection closest to WEKA:
        # iterate in feature order and replace only when the gain ratio
        # improves strictly. In ties, keep the first candidate.
        best = split_candidates[0]
        best_gain_ratio = float(best["gain_ratio"])
        for candidate in split_candidates[1:]:
            candidate_gr = float(candidate["gain_ratio"])
            if candidate_gr > best_gain_ratio + 1e-12:
                best = candidate
                best_gain_ratio = candidate_gr
        best_gain_ratio = float(best["gain_ratio"])
        best_feature = int(best["feature"])
        best_threshold = None if best.get("threshold") is None else float(best["threshold"])

        self._maybe_record_split_debug(
            current_path=current_path,
            depth=depth,
            class_counts=counts,
            total_weight=total_weight,
            candidates_before_prefilter=prefilter_candidates,
            candidates_after_prefilter=split_candidates,
            mean_gain=prefilter_mean_gain,
            selected_candidate=best,
        )

        # If no valid split is found, or the gain ratio is too small, make a leaf.
        if best_feature is None or best_gain_ratio < self.min_gain_ratio:
            return self._make_leaf_node(
                prediction=prediction,
                prediction_idx=pred_class_idx,
                class_counts=counts,
                train_indices=indices,
                train_weights=weights,
            )

        feat_values = X[indices, best_feature]
        if best["split_type"] == "nominal":
            missing_mask = self._feature_missing_mask(feat_values)
            valid_mask = ~missing_mask
            known_idx = indices[valid_mask]
            known_w = weights[valid_mask].astype(np.float64, copy=False)
            known_values = np.asarray(feat_values[valid_mask])

            branch_idx_parts: dict[Any, list[np.ndarray]] = {value: [] for value in best["values"]}
            branch_w_parts: dict[Any, list[np.ndarray]] = {value: [] for value in best["values"]}
            matched_mask = np.zeros(known_idx.size, dtype=bool)
            for value in best["values"]:
                if value == _NOMINAL_OTHER_BRANCH:
                    continue
                branch_mask = self._nominal_match_mask(known_values, value)
                if np.any(branch_mask):
                    branch_idx_parts[value].append(known_idx[branch_mask])
                    branch_w_parts[value].append(known_w[branch_mask])
                    matched_mask |= branch_mask

            if np.any(~matched_mask):
                default_value = best["default_child"]
                branch_idx_parts[default_value].append(known_idx[~matched_mask])
                branch_w_parts[default_value].append(known_w[~matched_mask])

            if np.any(missing_mask):
                missing_idx = indices[missing_mask]
                missing_w = weights[missing_mask].astype(np.float64, copy=False)
                if self.enable_fractional_missing:
                    for value in best["values"]:
                        routed_w = missing_w * float(best["branch_probs"][value])
                        keep = routed_w > 1e-12
                        if np.any(keep):
                            branch_idx_parts[value].append(missing_idx[keep])
                            branch_w_parts[value].append(routed_w[keep])
                else:
                    default_value = best["default_child"]
                    branch_idx_parts[default_value].append(missing_idx)
                    branch_w_parts[default_value].append(missing_w)

            children = {}
            for value in best["values"]:
                child_idx, child_w = self._concat_weighted_parts(
                    branch_idx_parts[value],
                    branch_w_parts[value],
                )
                if child_idx.size == 0 or not np.any(child_w > 1e-12):
                    children[value] = self._make_leaf_node(
                        prediction=prediction,
                        prediction_idx=pred_class_idx,
                        class_counts=np.zeros(self.n_classes_, dtype=np.float64),
                        probability_counts=counts,
                        train_indices=np.array([], dtype=np.int32),
                        train_weights=np.array([], dtype=np.float64),
                    )
                else:
                    child_condition = self._export_branch_condition("nominal", best_feature, None, value)
                    child_path = current_path + ([child_condition] if child_condition is not None else [])
                    children[value] = self._build_tree(
                        X,
                        y,
                        child_idx,
                        child_w,
                        depth + 1,
                        path_conditions=child_path,
                    )

            node = _Node(
                is_leaf=False,
                prediction=prediction,
                prediction_idx=pred_class_idx,
                feature_index=best_feature,
                split_type="nominal",
                nominal_children=children,
                nominal_child_probs={k: float(v) for k, v in best["branch_probs"].items()},
                nominal_default_child=best["default_child"],
                class_counts=counts,
                train_indices=indices.copy(),
                train_weights=weights.copy(),
                split_gain_ratio=best_gain_ratio,
                split_info_gain=float(best["info_gain"]),
                split_info_gain_adj=float(best["info_gain_adj"]),
                split_intrinsic_value=float(best["intrinsic"]),
                split_known_weight=float(best["known_weight"]),
                split_missing_weight=float(best["missing_weight"]),
            )
            return self._maybe_collapse_unpruned_split(node)

        known_local_sorted = np.asarray(best.get("sorted_local_idx"), dtype=np.int32)
        sorted_weights = weights[known_local_sorted]
        best_left_idx = indices[known_local_sorted[: best["split_pos"] + 1]]
        best_right_idx = indices[known_local_sorted[best["split_pos"] + 1:]]
        best_left_w = sorted_weights[: best["split_pos"] + 1].astype(np.float64, copy=False)
        best_right_w = sorted_weights[best["split_pos"] + 1:].astype(np.float64, copy=False)

        left_known_w = float(np.sum(best_left_w))
        right_known_w = float(np.sum(best_right_w))
        total_known_w = left_known_w + right_known_w
        left_prob = (left_known_w / total_known_w) if total_known_w > 0.0 else 0.5
        missing_go_to_left = left_prob >= 0.5

        numeric_feat = self._coerce_numeric_column(feat_values)
        missing_mask = ~np.isfinite(numeric_feat)
        if np.any(missing_mask):
            missing_idx = indices[missing_mask]
            missing_w = weights[missing_mask].astype(np.float64, copy=False)
            if self.enable_fractional_missing and missing_idx.size > 0:
                left_missing_w = missing_w * left_prob
                right_missing_w = missing_w * (1.0 - left_prob)
                left_keep = left_missing_w > 1e-12
                right_keep = right_missing_w > 1e-12
                if np.any(left_keep):
                    best_left_idx = np.concatenate((best_left_idx, missing_idx[left_keep]))
                    best_left_w = np.concatenate((best_left_w, left_missing_w[left_keep]))
                if np.any(right_keep):
                    best_right_idx = np.concatenate((best_right_idx, missing_idx[right_keep]))
                    best_right_w = np.concatenate((best_right_w, right_missing_w[right_keep]))
            else:
                if missing_go_to_left:
                    best_left_idx = np.concatenate((best_left_idx, missing_idx))
                    best_left_w = np.concatenate((best_left_w, missing_w))
                else:
                    best_right_idx = np.concatenate((best_right_idx, missing_idx))
                    best_right_w = np.concatenate((best_right_w, missing_w))

        left_condition = self._export_branch_condition("numeric", best_feature, best_threshold, "left")
        right_condition = self._export_branch_condition("numeric", best_feature, best_threshold, "right")
        left_child = self._build_tree(
            X,
            y,
            best_left_idx,
            best_left_w,
            depth + 1,
            path_conditions=current_path + ([left_condition] if left_condition is not None else []),
        )
        right_child = self._build_tree(
            X,
            y,
            best_right_idx,
            best_right_w,
            depth + 1,
            path_conditions=current_path + ([right_condition] if right_condition is not None else []),
        )

        node = _Node(
            is_leaf=False,
            prediction=prediction,
            prediction_idx=pred_class_idx,
            feature_index=best_feature,
            split_type="numeric",
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            class_counts=counts,
            missing_go_to_left=missing_go_to_left,
            left_prob=left_prob,
            train_indices=indices.copy(),
            train_weights=weights.copy(),
            split_gain_ratio=best_gain_ratio,
            split_info_gain=float(best["info_gain"]),
            split_info_gain_adj=float(best["info_gain_adj"]),
            split_intrinsic_value=float(best["intrinsic"]),
            split_known_weight=float(best["known_weight"]),
            split_missing_weight=float(best["missing_weight"]),
        )
        return self._maybe_collapse_unpruned_split(node)

    def _relocate_split_point(self, feature_index: int, threshold: float) -> float:
        """
        Relocate the split point to the largest observed value in the full
        dataset that is less than or equal to the provisional threshold.

        This follows the semantics documented by WEKA for `setSplitPoint`.
        """
        if self._train_X_ is None:
            return float(threshold)

        finite_values = self._get_relocate_values(int(feature_index))
        if finite_values.size == 0:
            return float(threshold)

        pos = int(np.searchsorted(finite_values, threshold, side="right")) - 1
        if pos < 0:
            return float(threshold)
        return float(finite_values[pos])

    def _get_relocate_values(self, feature_index: int) -> np.ndarray:
        cached = self._relocate_values_cache_.get(int(feature_index))
        if cached is not None:
            return cached
        if self._train_X_ is None:
            return np.zeros(0, dtype=np.float64)

        feat_values = self._coerce_numeric_column(self._train_X_[:, feature_index])
        finite_values = np.sort(feat_values[np.isfinite(feat_values)], kind="mergesort")
        self._relocate_values_cache_[int(feature_index)] = finite_values
        return finite_values

    def _node_n_samples(self, node: _Node) -> float:
        if node._cached_n_samples is not None:
            return float(node._cached_n_samples)
        if node.class_counts is None:
            return 0.0
        total = float(np.sum(node.class_counts))
        node._cached_n_samples = total
        return total

    def _leaf_training_errors(self, node: _Node) -> float:
        if node._cached_leaf_training_errors is not None:
            return float(node._cached_leaf_training_errors)
        n = self._node_n_samples(node)
        if n <= 0 or node.class_counts is None:
            return 0.0
        err = float(n - float(np.max(node.class_counts)))
        node._cached_leaf_training_errors = err
        return err

    def _subtree_training_errors(self, node: _Node) -> float:
        if node.is_leaf:
            return self._leaf_training_errors(node)
        child_items = self._iter_child_items(node)
        if not child_items:
            return self._leaf_training_errors(node)
        return float(sum(self._subtree_training_errors(child) for _, child in child_items))

    def _maybe_collapse_unpruned_split(self, node: _Node) -> _Node:
        """
        In pure unpruned mode, discard splits that do not reduce training error
        relative to the node's leaf prediction.

        This captures branches with positive gain but no real predictive
        improvement, which are precisely the divergences observed against WEKA
        under `-U`.
        """
        if node.is_leaf or self.enable_pruning or self.reduced_error_pruning:
            return node

        child_items = self._iter_child_items(node)
        if not child_items:
            return node

        leaf_err = self._leaf_training_errors(node)
        subtree_err = self._subtree_training_errors(node)
        if leaf_err <= subtree_err + 1e-12:
            self._clear_split(node)
        return node

    def _collapse_subtree(self, node: _Node) -> _Node:
        if node.is_leaf:
            return node

        if node.split_type == "nominal" and node.nominal_children:
            node.nominal_children = {
                edge: self._collapse_subtree(child)
                for edge, child in node.nominal_children.items()
            }
        else:
            if node.left is not None:
                node.left = self._collapse_subtree(node.left)
            if node.right is not None:
                node.right = self._collapse_subtree(node.right)

        child_items = self._iter_child_items(node)
        if not child_items:
            return node

        leaf_err = self._leaf_training_errors(node)
        subtree_err = self._subtree_training_errors(node)
        if leaf_err <= subtree_err + 1e-12:
            self._clear_split(node)
        return node

    def _add_errs(self, n: float, errors: float) -> float:
        """
        Extra pessimistic errors in the style of C4.5/J48 `addErrs`.

        Returns the number of additional errors above the observed error.
        """
        n = float(n)
        if n <= 0.0:
            return 0.0

        e = min(max(float(errors), 0.0), n)
        cf = min(max(self.confidence_factor, 1e-9), 0.5 - 1e-12)

        def _normal_extra(err_count: float) -> float:
            if err_count + 0.5 >= n:
                return float(max(n - err_count, 0.0))
            z = NormalDist().inv_cdf(1.0 - cf)
            f = (err_count + 0.5) / n
            z2_n = (z * z) / n
            rad_sq = (f / n) - ((f * f) / n) + (z * z) / (4.0 * n * n)
            rad_sq = max(rad_sq, 0.0)
            numer = f + 0.5 * z2_n + z * np.sqrt(rad_sq)
            denom = 1.0 + z2_n
            r = numer / denom
            return float((r * n) - err_count)

        # Low-error case: interpolation used by J48.
        if e < 1.0:
            base = n * (1.0 - np.power(cf, 1.0 / n))
            if e == 0.0:
                return base
            one_extra = _normal_extra(1.0)
            return float(base + e * (one_extra - base))

        # Edge case: almost everything is misclassified.
        if e + 0.5 >= n:
            return float(max(n - e, 0.0))

        # Normal approximation with continuity correction.
        return _normal_extra(e)

    @staticmethod
    def _invalidate_cached_metrics(node: _Node) -> None:
        node._cached_n_samples = None
        node._cached_leaf_training_errors = None
        node._cached_leaf_estimated_errors = None
        node._cached_subtree_estimated_errors = None

    def _node_estimated_errors_as_leaf(self, node: _Node) -> float:
        if node._cached_leaf_estimated_errors is not None:
            return float(node._cached_leaf_estimated_errors)
        n = self._node_n_samples(node)
        if n <= 0:
            return 0.0
        err = self._leaf_training_errors(node)
        est = float(err + self._add_errs(n, err))
        node._cached_leaf_estimated_errors = est
        return est

    def _node_estimated_errors_from_counts(self, counts: np.ndarray, pred_idx: Optional[int]) -> float:
        n = float(np.sum(counts))
        if n <= 0.0:
            return 0.0
        if pred_idx is None or pred_idx < 0 or pred_idx >= counts.size:
            pred_idx = int(np.argmax(counts))
        err = max(0.0, n - float(counts[pred_idx]))
        return float(err + self._add_errs(n, err))

    def _route_external_indices_with_weights(
        self,
        node: _Node,
        X_data: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
    ) -> dict[Any, tuple[np.ndarray, np.ndarray]]:
        """
        Route weighted instances from an arbitrary set to the node's children
        using the same semantics as prediction.
        """
        if indices.size == 0:
            return {
                edge: (np.array([], dtype=np.int32), np.array([], dtype=np.float64))
                for edge, _ in self._iter_child_items(node)
            }

        x_feat = X_data[indices, node.feature_index]
        weights = weights.astype(np.float64, copy=False)

        if node.split_type == "nominal":
            child_items = self._iter_child_items(node)
            idx_parts = {edge: [] for edge, _ in child_items}
            weight_parts = {edge: [] for edge, _ in child_items}

            missing_mask = self._feature_missing_mask(x_feat)
            valid_mask = ~missing_mask

            if np.any(valid_mask):
                known_idx = indices[valid_mask]
                known_w = weights[valid_mask]
                known_values = np.asarray(x_feat[valid_mask])
                matched_mask = np.zeros(known_idx.size, dtype=bool)
                for edge, _ in child_items:
                    branch_mask = self._nominal_match_mask(known_values, edge)
                    if np.any(branch_mask):
                        idx_parts[edge].append(known_idx[branch_mask])
                        weight_parts[edge].append(known_w[branch_mask])
                        matched_mask |= branch_mask

                unseen_mask = ~matched_mask
                if np.any(unseen_mask):
                    fallback_idx = known_idx[unseen_mask]
                    fallback_w = known_w[unseen_mask]
                    if self.enable_fractional_missing:
                        for edge, _ in child_items:
                            prob = float((node.nominal_child_probs or {}).get(edge, 0.0))
                            routed_w = fallback_w * prob
                            keep = routed_w > 1e-12
                            if np.any(keep):
                                idx_parts[edge].append(fallback_idx[keep])
                                weight_parts[edge].append(routed_w[keep])
                    else:
                        default_edge = node.nominal_default_child
                        if default_edge in idx_parts:
                            idx_parts[default_edge].append(fallback_idx)
                            weight_parts[default_edge].append(fallback_w)

            if np.any(missing_mask):
                miss_idx = indices[missing_mask]
                miss_w = weights[missing_mask]
                if self.enable_fractional_missing:
                    for edge, _ in child_items:
                        prob = float((node.nominal_child_probs or {}).get(edge, 0.0))
                        routed_w = miss_w * prob
                        keep = routed_w > 1e-12
                        if np.any(keep):
                            idx_parts[edge].append(miss_idx[keep])
                            weight_parts[edge].append(routed_w[keep])
                else:
                    default_edge = node.nominal_default_child
                    if default_edge in idx_parts:
                        idx_parts[default_edge].append(miss_idx)
                        weight_parts[default_edge].append(miss_w)

            return {
                edge: self._concat_weighted_parts(idx_parts[edge], weight_parts[edge])
                for edge, _ in child_items
            }

        numeric_feat = self._coerce_numeric_column(x_feat)
        finite_mask = np.isfinite(numeric_feat)
        missing_mask = ~finite_mask
        left_indices_parts = []
        left_weights_parts = []
        right_indices_parts = []
        right_weights_parts = []

        if np.any(finite_mask):
            known_idx = indices[finite_mask]
            known_w = weights[finite_mask]
            known_left_mask = numeric_feat[finite_mask] <= node.threshold
            if np.any(known_left_mask):
                left_indices_parts.append(known_idx[known_left_mask])
                left_weights_parts.append(known_w[known_left_mask])
            if np.any(~known_left_mask):
                right_indices_parts.append(known_idx[~known_left_mask])
                right_weights_parts.append(known_w[~known_left_mask])

        if np.any(missing_mask):
            miss_idx = indices[missing_mask]
            miss_w = weights[missing_mask]
            if self.enable_fractional_missing:
                left_w = miss_w * float(node.left_prob)
                right_w = miss_w * float(1.0 - node.left_prob)
                left_keep = left_w > 1e-12
                right_keep = right_w > 1e-12
                if np.any(left_keep):
                    left_indices_parts.append(miss_idx[left_keep])
                    left_weights_parts.append(left_w[left_keep])
                if np.any(right_keep):
                    right_indices_parts.append(miss_idx[right_keep])
                    right_weights_parts.append(right_w[right_keep])
            else:
                if bool(node.missing_go_to_left):
                    left_indices_parts.append(miss_idx)
                    left_weights_parts.append(miss_w)
                else:
                    right_indices_parts.append(miss_idx)
                    right_weights_parts.append(miss_w)

        return {
            "left": self._concat_weighted_parts(left_indices_parts, left_weights_parts),
            "right": self._concat_weighted_parts(right_indices_parts, right_weights_parts),
        }

    def _route_indices_with_weights(
        self,
        node: _Node,
        indices: np.ndarray,
        weights: np.ndarray,
    ) -> dict[Any, tuple[np.ndarray, np.ndarray]]:
        if self._train_X_ is None:
            raise RuntimeError("training data was cleaned before routing completed")
        return self._route_external_indices_with_weights(
            node,
            self._train_X_,
            indices,
            weights,
        )

    def _stratify_indices_like_weka(
        self,
        shuffled_indices: np.ndarray,
        y_encoded: np.ndarray,
        n_folds: int,
    ) -> np.ndarray:
        shuffled_indices = np.asarray(shuffled_indices, dtype=np.int32)
        if shuffled_indices.size == 0 or n_folds <= 1 or int(self.n_classes_ or 0) <= 1:
            return shuffled_indices

        # Literal implementation of WEKA's `Instances.stratify`:
        # first regroup by class while preserving randomized order, then
        # apply `stratStep`. A vectorized NumPy version does not reproduce the
        # exact swaps of the original mutable structure.
        grouped = shuffled_indices.tolist()
        index = 1
        while index < len(grouped):
            class_value = y_encoded[grouped[index - 1]]
            j = index
            while j < len(grouped):
                if y_encoded[grouped[j]] == class_value:
                    grouped[index], grouped[j] = grouped[j], grouped[index]
                    index += 1
                j += 1
            index += 1

        stratified: list[int] = []
        start = 0
        while len(stratified) < len(grouped):
            j = start
            while j < len(grouped):
                stratified.append(grouped[j])
                j += n_folds
            start += 1
        return np.asarray(stratified, dtype=np.int32)

    @staticmethod
    def _fold_slice_bounds(n_instances: int, n_folds: int, fold: int) -> tuple[int, int]:
        base = n_instances // n_folds
        remainder = n_instances % n_folds
        fold_size = base + (1 if fold < remainder else 0)
        offset = fold if fold < remainder else remainder
        first = fold * base + offset
        return first, first + fold_size

    @staticmethod
    def _java_random_state(seed: Optional[int]) -> int:
        mask = (1 << 48) - 1
        return ((1 if seed is None else int(seed)) ^ 0x5DEECE66D) & mask

    @classmethod
    def _java_shuffle_indices(
        cls,
        indices: np.ndarray,
        seed: Optional[int] = None,
        state: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Match WEKA/Java `Collections.shuffle` style randomization while
        allowing the same RNG state to be reused across multiple shuffles.
        """
        shuffled = np.asarray(indices, dtype=np.int32).copy()
        if shuffled.size <= 1:
            return shuffled, cls._java_random_state(seed) if state is None else int(state)

        mask = (1 << 48) - 1
        state = cls._java_random_state(seed) if state is None else int(state) & mask

        def next_bits(bits: int) -> int:
            nonlocal state
            state = (state * 0x5DEECE66D + 0xB) & mask
            return state >> (48 - bits)

        def next_int(bound: int) -> int:
            if bound <= 0:
                raise ValueError("bound must be positive")
            if (bound & -bound) == bound:
                return (bound * next_bits(31)) >> 31
            while True:
                bits = next_bits(31)
                value = bits % bound
                if bits - value + (bound - 1) >= 0:
                    return value

        for j in range(shuffled.size - 1, 0, -1):
            k = next_int(j + 1)
            shuffled[j], shuffled[k] = shuffled[k], shuffled[j]
        return shuffled, state

    @classmethod
    def _java_randomized_indices(cls, n_instances: int, seed: Optional[int]) -> np.ndarray:
        """
        Match WEKA's `Instances.randomize(new Random(seed))` ordering.
        """
        shuffled, _ = cls._java_shuffle_indices(np.arange(n_instances, dtype=np.int32), seed=seed)
        return shuffled

    def _make_pruning_split_indices(
        self,
        y_encoded: np.ndarray,
        n_folds: int,
        fold: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        ordered = self._stratify_indices_like_weka(
            np.arange(y_encoded.shape[0], dtype=np.int32),
            y_encoded,
            n_folds,
        )
        first, last = self._fold_slice_bounds(ordered.size, n_folds, fold)
        prune_idx = ordered[first:last].astype(np.int32, copy=False)
        grow_idx = np.concatenate((ordered[:first], ordered[last:])).astype(np.int32, copy=False)
        grow_idx, _ = self._java_shuffle_indices(grow_idx, seed=self.random_state)
        return prune_idx, grow_idx

    def _leaf_weighted_error(
        self,
        pred_idx: Optional[int],
        y_encoded: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        if indices.size == 0:
            return 0.0
        if pred_idx is None:
            return float(np.sum(weights))
        return float(np.sum(weights[y_encoded[indices] != int(pred_idx)]))

    def _subtree_weighted_error_on_external(
        self,
        node: _Node,
        X_data: np.ndarray,
        y_encoded: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        if indices.size == 0:
            return 0.0

        child_items = self._iter_child_items(node)
        if node.is_leaf or not child_items:
            return self._leaf_weighted_error(node.prediction_idx, y_encoded, indices, weights)

        routed = self._route_external_indices_with_weights(node, X_data, indices, weights)
        total_error = 0.0
        for edge, child in child_items:
            child_idx, child_w = routed.get(
                edge,
                (np.array([], dtype=np.int32), np.array([], dtype=np.float64)),
            )
            total_error += self._subtree_weighted_error_on_external(
                child,
                X_data,
                y_encoded,
                child_idx,
                child_w,
            )
        return total_error

    def _prune_tree_reduced_error(
        self,
        node: _Node,
        X_prune: np.ndarray,
        y_prune: np.ndarray,
        indices: np.ndarray,
        weights: np.ndarray,
    ) -> _Node:
        if node.is_leaf:
            return node

        child_items = self._iter_child_items(node)
        if not child_items:
            return node
        if indices.size == 0:
            self._clear_split(node)
            return node

        routed = self._route_external_indices_with_weights(node, X_prune, indices, weights)
        if node.split_type == "nominal" and node.nominal_children:
            pruned_children = {}
            for edge, child in child_items:
                child_idx, child_w = routed.get(
                    edge,
                    (np.array([], dtype=np.int32), np.array([], dtype=np.float64)),
                )
                pruned_children[edge] = self._prune_tree_reduced_error(
                    child,
                    X_prune,
                    y_prune,
                    child_idx,
                    child_w,
                )
            node.nominal_children = pruned_children
        else:
            if node.left is not None:
                left_idx, left_w = routed.get(
                    "left",
                    (np.array([], dtype=np.int32), np.array([], dtype=np.float64)),
                )
                node.left = self._prune_tree_reduced_error(
                    node.left,
                    X_prune,
                    y_prune,
                    left_idx,
                    left_w,
                )
            if node.right is not None:
                right_idx, right_w = routed.get(
                    "right",
                    (np.array([], dtype=np.int32), np.array([], dtype=np.float64)),
                )
                node.right = self._prune_tree_reduced_error(
                    node.right,
                    X_prune,
                    y_prune,
                    right_idx,
                    right_w,
                )

        leaf_error = self._leaf_weighted_error(node.prediction_idx, y_prune, indices, weights)
        subtree_error = self._subtree_weighted_error_on_external(
            node,
            X_prune,
            y_prune,
            indices,
            weights,
        )
        if leaf_error <= subtree_error + 1e-12:
            self._clear_split(node)
        return node

    def _fit_with_reduced_error_pruning(
        self,
        X: np.ndarray,
        y_encoded: np.ndarray,
        weights: np.ndarray,
    ) -> _Node:
        prune_idx, grow_idx = self._make_pruning_split_indices(
            y_encoded,
            self.num_folds,
            fold=self.num_folds - 1,
        )
        if grow_idx.size == 0:
            raise ValueError("reduced_error_pruning requires at least one grow fold")

        X_grow = X[grow_idx]
        y_grow = y_encoded[grow_idx]
        w_grow = weights[grow_idx].astype(np.float64, copy=False)
        self._train_X_ = X_grow
        self._train_y_encoded_ = y_grow
        grow_local_idx = np.arange(grow_idx.shape[0], dtype=np.int32)

        if prune_idx.size == 0:
            return self._build_tree(X_grow, y_grow, grow_local_idx, w_grow, depth=0, path_conditions=[])

        X_prune = X[prune_idx]
        y_prune = y_encoded[prune_idx]
        w_prune = weights[prune_idx].astype(np.float64, copy=False)
        prune_local_idx = np.arange(prune_idx.shape[0], dtype=np.int32)
        root = self._build_tree(X_grow, y_grow, grow_local_idx, w_grow, depth=0, path_conditions=[])
        return self._prune_tree_reduced_error(
            root,
            X_prune,
            y_prune,
            prune_local_idx,
            w_prune,
        )

    def _subtree_estimated_errors_with_incoming(
        self,
        node: _Node,
        incoming_indices: np.ndarray,
        incoming_weights: np.ndarray,
    ) -> float:
        """
        Estimate subtree error when it receives additional weighted instances,
        using explicit instance routing instead of projection from aggregated
        counts.
        """
        incoming_indices = np.asarray(incoming_indices, dtype=np.int32)
        incoming_weights = np.asarray(incoming_weights, dtype=np.float64)

        if incoming_indices.size == 0:
            return self._subtree_estimated_errors(node)

        child_items = self._iter_child_items(node)
        if node.is_leaf or not child_items:
            extra_counts = np.bincount(
                self._train_y_encoded_[incoming_indices],
                weights=incoming_weights,
                minlength=self.n_classes_,
            ).astype(np.float64, copy=False)
            base_counts = (
                node.class_counts.astype(np.float64, copy=False)
                if node.class_counts is not None
                else np.zeros(self.n_classes_, dtype=np.float64)
            )
            total_counts = base_counts + extra_counts
            pred_idx = int(np.argmax(total_counts)) if np.any(total_counts > 0.0) else node.prediction_idx
            return self._node_estimated_errors_from_counts(total_counts, pred_idx)

        routed = self._route_indices_with_weights(node, incoming_indices, incoming_weights)
        total_error = 0.0
        for edge, child in child_items:
            child_idx, child_w = routed.get(
                edge,
                (np.array([], dtype=np.int32), np.array([], dtype=np.float64)),
            )
            total_error += self._subtree_estimated_errors_with_incoming(child, child_idx, child_w)
        return total_error

    def _augment_subtree_with_incoming(
        self,
        node: _Node,
        incoming_indices: np.ndarray,
        incoming_weights: np.ndarray,
    ) -> _Node:
        """
        Update a promoted subtree by adding weighted sibling instances so that
        the structure resulting from subtree raising has class distributions
        consistent with training.
        """
        incoming_indices = np.asarray(incoming_indices, dtype=np.int32)
        incoming_weights = np.asarray(incoming_weights, dtype=np.float64)

        if node.class_counts is None:
            node.class_counts = np.zeros(self.n_classes_, dtype=np.float64)
        else:
            node.class_counts = node.class_counts.astype(np.float64, copy=True)

        if incoming_indices.size > 0:
            extra_counts = np.bincount(
                self._train_y_encoded_[incoming_indices],
                weights=incoming_weights,
                minlength=self.n_classes_,
            ).astype(np.float64, copy=False)
            node.class_counts = node.class_counts + extra_counts

            if node.train_indices is None:
                node.train_indices = incoming_indices.copy()
                node.train_weights = incoming_weights.copy()
            else:
                node.train_indices = np.concatenate((node.train_indices, incoming_indices)).astype(np.int32, copy=False)
                node.train_weights = np.concatenate((node.train_weights, incoming_weights)).astype(np.float64, copy=False)

        if np.any(node.class_counts > 0.0):
            node.prediction_idx = int(np.argmax(node.class_counts))
            node.prediction = self.classes_[node.prediction_idx]
        self._invalidate_cached_metrics(node)

        child_items = self._iter_child_items(node)
        if node.is_leaf or not child_items or incoming_indices.size == 0:
            return node

        routed = self._route_indices_with_weights(node, incoming_indices, incoming_weights)
        if node.split_type == "nominal":
            for edge, child in child_items:
                child_idx, child_w = routed.get(
                    edge,
                    (np.array([], dtype=np.int32), np.array([], dtype=np.float64)),
                )
                if child_idx.size > 0:
                    node.nominal_children[edge] = self._augment_subtree_with_incoming(child, child_idx, child_w)
        else:
            left_idx, left_w = routed.get("left", (np.array([], dtype=np.int32), np.array([], dtype=np.float64)))
            right_idx, right_w = routed.get("right", (np.array([], dtype=np.int32), np.array([], dtype=np.float64)))
            if left_idx.size > 0 and node.left is not None:
                node.left = self._augment_subtree_with_incoming(node.left, left_idx, left_w)
            if right_idx.size > 0 and node.right is not None:
                node.right = self._augment_subtree_with_incoming(node.right, right_idx, right_w)
        return node

    def _estimate_raise_cost(
        self,
        promoted: _Node,
        incoming_indices: np.ndarray,
        incoming_weights: np.ndarray,
    ) -> float:
        """
        Cost of subtree raising by explicitly evaluating how sibling instances
        are redistributed through the promoted subtree.
        """
        if incoming_indices.size == 0:
            return self._subtree_estimated_errors(promoted)

        return self._subtree_estimated_errors_with_incoming(
            promoted,
            incoming_indices,
            incoming_weights,
        )

    def _subtree_estimated_errors(self, node: _Node) -> float:
        if node._cached_subtree_estimated_errors is not None:
            return float(node._cached_subtree_estimated_errors)
        if node.is_leaf:
            est = self._node_estimated_errors_as_leaf(node)
            node._cached_subtree_estimated_errors = est
            return est
        child_items = self._iter_child_items(node)
        if not child_items:
            est = self._node_estimated_errors_as_leaf(node)
            node._cached_subtree_estimated_errors = est
            return est
        est = float(sum(self._subtree_estimated_errors(child) for _, child in child_items))
        node._cached_subtree_estimated_errors = est
        return est

    def _largest_branch(self, node: _Node) -> Optional[tuple[Any, _Node]]:
        child_items = self._iter_child_items(node)
        if not child_items:
            return None
        return max(
            child_items,
            key=lambda item: (
                self._node_n_samples(item[1]),
                -1 if item[0] == "left" else 0 if item[0] == "right" else 1,
            ),
        )

    def _clear_split(self, node: _Node) -> None:
        node.is_leaf = True
        node.feature_index = None
        node.split_type = None
        node.threshold = None
        node.left = None
        node.right = None
        node.nominal_children = None
        node.nominal_child_probs = None
        node.nominal_default_child = None
        node.split_gain_ratio = None
        node.split_info_gain = None
        node.split_info_gain_adj = None
        node.split_intrinsic_value = None
        node.split_known_weight = None
        node.split_missing_weight = None
        self._invalidate_cached_metrics(node)

    def _prune_tree(self, node: _Node) -> _Node:
        """
        Confidence-based post-pruning (pessimistic pruning).
        """
        if node.is_leaf:
            return node

        if node.split_type == "nominal" and node.nominal_children:
            node.nominal_children = {
                edge: self._prune_tree(child)
                for edge, child in node.nominal_children.items()
            }
        else:
            if node.left is not None:
                node.left = self._prune_tree(node.left)
            if node.right is not None:
                node.right = self._prune_tree(node.right)

        self._invalidate_cached_metrics(node)
        leaf_est = self._node_estimated_errors_as_leaf(node)
        subtree_est = self._subtree_estimated_errors(node)

        child_items = self._iter_child_items(node)
        branch_est = float("inf")
        largest_branch_payload: Optional[tuple[_Node, np.ndarray, np.ndarray]] = None
        if self.enable_subtree_raising and child_items:
            largest = self._largest_branch(node)
            if largest is not None:
                largest_edge, largest_child = largest
                sib_idx_parts = []
                sib_w_parts = []
                for other_edge, other_child in child_items:
                    if other_edge == largest_edge:
                        continue
                    if other_child.train_indices is None or other_child.train_weights is None:
                        continue
                    sib_idx_parts.append(other_child.train_indices)
                    sib_w_parts.append(other_child.train_weights)
                incoming_idx, incoming_w = self._concat_weighted_parts(sib_idx_parts, sib_w_parts)
                branch_est = self._estimate_raise_cost(largest_child, incoming_idx, incoming_w)
                largest_branch_payload = (largest_child, incoming_idx, incoming_w)

        if leaf_est <= subtree_est + 0.1 and leaf_est <= branch_est + 0.1:
            self._clear_split(node)
            return node

        if largest_branch_payload is not None and branch_est <= subtree_est + 0.1:
            promoted, incoming_idx, incoming_w = largest_branch_payload
            promoted_copy = copy.deepcopy(promoted)
            if incoming_idx.size > 0:
                promoted_copy = self._augment_subtree_with_incoming(
                    promoted_copy,
                    incoming_idx,
                    incoming_w,
                )
            # The promoted subtree now represents the full parent node.
            promoted_copy.class_counts = node.class_counts.astype(np.float64, copy=True)
            promoted_copy.prediction = node.prediction
            promoted_copy.prediction_idx = node.prediction_idx
            if node.train_indices is not None and node.train_weights is not None:
                promoted_copy.train_indices = node.train_indices.copy()
                promoted_copy.train_weights = node.train_weights.copy()
            self._invalidate_cached_metrics(promoted_copy)
            # Re-run pruning on the promoted subtree using its final counts.
            # Without this second pass, subtree raising can preserve splits
            # that no longer survive the pessimistic pruning test after the
            # sibling mass has been folded into the promoted branch.
            return self._prune_tree(promoted_copy)

        return node

    def _cleanup_node_training_state(self, node: Optional[_Node]) -> None:
        if node is None:
            return
        node.train_indices = None
        node.train_weights = None
        for _, child in self._iter_child_items(node):
            self._cleanup_node_training_state(child)

    def _cleanup_after_fit(self) -> None:
        if self.root_ is not None:
            self._cleanup_node_training_state(self.root_)
        self._train_X_ = None
        self._train_y_encoded_ = None
        self._log2_lut_ = None
        self._relocate_values_cache_ = {}

    def _predict_batch(self, X: np.ndarray, node: _Node, indices: np.ndarray) -> np.ndarray:
        """
        Recursive vectorized batch prediction.

        This function processes multiple samples at once using boolean masks,
        which is about 3-5x faster than iterating sample by sample.

        Parameters
        ----------
        X : ndarray
            Full feature matrix.
        node : _Node
            Current node in the tree traversal.
        indices : ndarray
            Indices of the samples to predict at this node.

        Returns
        -------
        predictions : ndarray
            Predictions for the selected samples.
        """
        if indices.size == 0:
            return np.array([], dtype=np.int32)
        
        # Temporary array used to store predictions.
        predictions = np.empty(indices.size, dtype=np.int32)

        # Base case: leaf node.
        if node.is_leaf:
            predictions[:] = int(node.prediction_idx)
            return predictions

        x_feat = X[indices, node.feature_index]
        if node.split_type == "nominal":
            feat_arr = np.asarray(x_feat)
            missing_mask = self._feature_missing_mask(feat_arr)
            valid_mask = ~missing_mask
            handled_mask = np.zeros(indices.size, dtype=bool)

            if node.nominal_children:
                for edge, child in node.nominal_children.items():
                    branch_mask = np.zeros(indices.size, dtype=bool)
                    if np.any(valid_mask):
                        if edge == _NOMINAL_OTHER_BRANCH:
                            branch_mask[valid_mask] = ~handled_mask[valid_mask]
                        else:
                            branch_mask[valid_mask] = self._nominal_match_mask(feat_arr[valid_mask], edge)
                    if np.any(branch_mask):
                        handled_mask |= branch_mask
                        predictions[branch_mask] = self._predict_batch(X, child, indices[branch_mask])

            fallback_mask = ~handled_mask
            if np.any(fallback_mask):
                default_child = None
                if node.nominal_children is not None:
                    default_child = node.nominal_children.get(node.nominal_default_child)
                if default_child is None and node.nominal_children:
                    default_child = next(iter(node.nominal_children.values()))
                if default_child is not None:
                    predictions[fallback_mask] = self._predict_batch(X, default_child, indices[fallback_mask])
            return predictions

        numeric_feat = self._coerce_numeric_column(x_feat)
        left_mask = np.zeros(indices.size, dtype=bool)
        finite_mask = np.isfinite(numeric_feat)
        if np.any(finite_mask):
            left_mask[finite_mask] = numeric_feat[finite_mask] <= node.threshold
        if np.any(~finite_mask):
            left_mask[~finite_mask] = bool(node.missing_go_to_left)

        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if left_indices.size > 0 and node.left is not None:
            predictions[left_mask] = self._predict_batch(X, node.left, left_indices)
        if right_indices.size > 0 and node.right is not None:
            predictions[~left_mask] = self._predict_batch(X, node.right, right_indices)

        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class for each sample.
        """
        X = np.asarray(X)
        if not self._matrix_can_stay_numeric(X, bool(self._nominal_features_)):
            X = np.asarray(X, dtype=object)
        elif not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]
        
        if n_samples == 0:
            return np.array([])

        if self.enable_fractional_missing and (self._matrix_has_missing(X) or bool(self._nominal_features_)):
            # When missing values are present and fractional mode is active,
            # use a probability mixture to avoid hard routing of NaNs.
            proba = self.predict_proba(X)
            pred_idx = np.argmax(proba, axis=1)
            return self.classes_[pred_idx]

        # Vectorized prediction.
        indices = np.arange(n_samples, dtype=np.int32)
        predictions = self._predict_batch(X, self.root_, indices)

        # Convert to native type and avoid object arrays.
        return self.classes_[predictions]

    def _predict_proba_batch(
        self, 
        X: np.ndarray, 
        node: _Node, 
        indices: np.ndarray, 
        proba: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Recursive probability prediction (in place).

        Modifies the `proba` array directly to avoid unnecessary allocations.

        Parameters
        ----------
        X : ndarray
            Full feature matrix.
        node : _Node
            Current node.
        indices : ndarray
            Indices of samples to process.
        proba : ndarray
            Output array modified in place.
        """
        if indices.size == 0:
            return

        if weights is None:
            weights = np.ones(indices.size, dtype=np.float32)
        else:
            weights = weights.astype(np.float32, copy=False)
        
        # Base case: leaf node.
        if node.is_leaf:
            leaf_counts = node.class_counts
            if leaf_counts is not None:
                counts_sum = float(leaf_counts.sum())
                if counts_sum > 0:
                    if self.use_laplace:
                        dist = (
                            (leaf_counts + 1.0)
                            / (counts_sum + float(self.n_classes_))
                        ).astype(np.float32, copy=False)
                    else:
                        dist = (leaf_counts / counts_sum).astype(np.float32, copy=False)
                elif node.probability_counts is not None:
                    prob_counts = node.probability_counts.astype(np.float64, copy=False)
                    prob_sum = float(prob_counts.sum())
                    if prob_sum > 0.0:
                        if self.use_laplace:
                            dist = (
                                (prob_counts + 1.0)
                                / (prob_sum + float(self.n_classes_))
                            ).astype(np.float32, copy=False)
                        else:
                            dist = (prob_counts / prob_sum).astype(np.float32, copy=False)
                    elif node.prediction_idx is not None:
                        dist = np.zeros(self.n_classes_, dtype=np.float32)
                        dist[int(node.prediction_idx)] = 1.0
                    else:
                        dist = np.full(self.n_classes_, 1.0 / self.n_classes_, dtype=np.float32)
                elif node.prediction_idx is not None:
                    dist = np.zeros(self.n_classes_, dtype=np.float32)
                    dist[int(node.prediction_idx)] = 1.0
                else:
                    # Fallback: uniform distribution.
                    dist = np.full(self.n_classes_, 1.0 / self.n_classes_, dtype=np.float32)
            else:
                if node.prediction_idx is not None:
                    dist = np.zeros(self.n_classes_, dtype=np.float32)
                    dist[int(node.prediction_idx)] = 1.0
                else:
                    dist = np.full(self.n_classes_, 1.0 / self.n_classes_, dtype=np.float32)
            proba[indices] += weights[:, None] * dist[None, :]
            return
        
        x_feat = X[indices, node.feature_index]
        if node.split_type == "nominal":
            feat_arr = np.asarray(x_feat)
            missing_mask = self._feature_missing_mask(feat_arr)
            valid_mask = ~missing_mask
            handled_valid = np.zeros(int(np.sum(valid_mask)), dtype=bool)

            if np.any(valid_mask):
                known_idx = indices[valid_mask]
                known_w = weights[valid_mask]
                known_values = feat_arr[valid_mask]
                if node.nominal_children:
                    for edge, child in node.nominal_children.items():
                        if edge == _NOMINAL_OTHER_BRANCH:
                            branch_mask = ~handled_valid
                        else:
                            branch_mask = self._nominal_match_mask(known_values, edge)
                        if np.any(branch_mask):
                            handled_valid |= branch_mask
                            self._predict_proba_batch(
                                X,
                                child,
                                known_idx[branch_mask],
                                proba,
                                known_w[branch_mask],
                            )

                unseen_mask = ~handled_valid
                if np.any(unseen_mask):
                    fallback_idx = known_idx[unseen_mask]
                    fallback_w = known_w[unseen_mask]
                    if self.enable_fractional_missing and node.nominal_children:
                        for edge, child in node.nominal_children.items():
                            prob = float((node.nominal_child_probs or {}).get(edge, 0.0))
                            routed_w = fallback_w * prob
                            if np.any(routed_w > 0.0):
                                self._predict_proba_batch(X, child, fallback_idx, proba, routed_w)
                    else:
                        default_child = None
                        if node.nominal_children is not None:
                            default_child = node.nominal_children.get(node.nominal_default_child)
                        if default_child is None and node.nominal_children:
                            default_child = next(iter(node.nominal_children.values()))
                        if default_child is not None:
                            self._predict_proba_batch(X, default_child, fallback_idx, proba, fallback_w)

            if np.any(missing_mask):
                miss_idx = indices[missing_mask]
                miss_w = weights[missing_mask]
                if self.enable_fractional_missing and node.nominal_children:
                    for edge, child in node.nominal_children.items():
                        prob = float((node.nominal_child_probs or {}).get(edge, 0.0))
                        routed_w = miss_w * prob
                        if np.any(routed_w > 0.0):
                            self._predict_proba_batch(X, child, miss_idx, proba, routed_w)
                else:
                    default_child = None
                    if node.nominal_children is not None:
                        default_child = node.nominal_children.get(node.nominal_default_child)
                    if default_child is None and node.nominal_children:
                        default_child = next(iter(node.nominal_children.values()))
                    if default_child is not None:
                        self._predict_proba_batch(X, default_child, miss_idx, proba, miss_w)
            return

        numeric_feat = self._coerce_numeric_column(x_feat)
        finite_mask = np.isfinite(numeric_feat)
        missing_mask = ~finite_mask

        if np.any(finite_mask):
            known_idx = indices[finite_mask]
            known_w = weights[finite_mask]
            known_left_mask = numeric_feat[finite_mask] <= node.threshold
            left_indices = known_idx[known_left_mask]
            left_w = known_w[known_left_mask]
            right_indices = known_idx[~known_left_mask]
            right_w = known_w[~known_left_mask]
            if left_indices.size > 0 and node.left is not None:
                self._predict_proba_batch(X, node.left, left_indices, proba, left_w)
            if right_indices.size > 0 and node.right is not None:
                self._predict_proba_batch(X, node.right, right_indices, proba, right_w)

        if np.any(missing_mask):
            miss_idx = indices[missing_mask]
            miss_w = weights[missing_mask]
            if self.enable_fractional_missing:
                left_w = miss_w * float(node.left_prob)
                right_w = miss_w * float(1.0 - node.left_prob)
                if miss_idx.size > 0 and np.any(left_w > 0.0) and node.left is not None:
                    self._predict_proba_batch(X, node.left, miss_idx, proba, left_w)
                if miss_idx.size > 0 and np.any(right_w > 0.0) and node.right is not None:
                    self._predict_proba_batch(X, node.right, miss_idx, proba, right_w)
            else:
                if bool(node.missing_go_to_left):
                    if node.left is not None:
                        self._predict_proba_batch(X, node.left, miss_idx, proba, miss_w)
                else:
                    if node.right is not None:
                        self._predict_proba_batch(X, node.right, miss_idx, proba, miss_w)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Probabilities are based on the class distribution in the leaf where
        each sample lands.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probability of each class for each sample.
            `proba[i, j] = P(class=j | X[i])`.
        """
        X = np.asarray(X)
        if not self._matrix_can_stay_numeric(X, bool(self._nominal_features_)):
            X = np.asarray(X, dtype=object)
        elif not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float32, copy=False)
        n_samples = X.shape[0]
        
        # Initialize the probability array.
        proba = np.zeros((n_samples, self.n_classes_), dtype=np.float32)
        
        if n_samples == 0:
            return proba
        
        # Vectorized prediction.
        indices = np.arange(n_samples, dtype=np.int32)
        self._predict_proba_batch(X, self.root_, indices, proba, None)
        
        # Check rows with zero sum (should not happen, but keep a safe fallback).
        row_sums = proba.sum(axis=1)
        zero_rows = row_sums == 0
        if zero_rows.any():
            proba[zero_rows] = 1.0 / self.n_classes_
        
        return proba

    def _tree_stats_recursive(self, node: _Node, depth: int) -> dict[str, int]:
        if node is None:
            return {
                "node_count": 0,
                "leaf_count": 0,
                "internal_count": 0,
                "max_depth": max(depth - 1, 0),
                "numeric_split_count": 0,
                "nominal_split_count": 0,
            }
        if node.is_leaf:
            return {
                "node_count": 1,
                "leaf_count": 1,
                "internal_count": 0,
                "max_depth": depth,
                "numeric_split_count": 0,
                "nominal_split_count": 0,
            }

        child_stats = [self._tree_stats_recursive(child, depth + 1) for _, child in self._iter_child_items(node)]
        summary = {
            "node_count": 1,
            "leaf_count": 0,
            "internal_count": 1,
            "max_depth": depth,
            "numeric_split_count": 1 if node.split_type == "numeric" else 0,
            "nominal_split_count": 1 if node.split_type == "nominal" else 0,
        }
        for child_stat in child_stats:
            for key in ("node_count", "leaf_count", "internal_count", "numeric_split_count", "nominal_split_count"):
                summary[key] += int(child_stat[key])
            summary["max_depth"] = max(summary["max_depth"], int(child_stat["max_depth"]))
        return summary

    def get_tree_stats(self) -> dict[str, Any]:
        if self.root_ is None:
            return {
                "node_count": 0,
                "leaf_count": 0,
                "internal_count": 0,
                "max_depth": 0,
                "numeric_split_count": 0,
                "nominal_split_count": 0,
            }
        stats = self._tree_stats_recursive(self.root_, 0)
        stats["class_count"] = int(self.n_classes_ or 0)
        return stats

    @staticmethod
    def _format_export_edge_value(value: Any) -> str:
        value = _to_python_scalar(value)
        if isinstance(value, float):
            return f"{value:.15g}"
        return str(value)

    def _export_branch_condition(
        self,
        split_type: Optional[str],
        feature_index: Optional[int],
        threshold: Optional[float],
        edge: Any,
    ) -> Optional[str]:
        feature_name = self._feature_name(feature_index)
        if feature_name is None or split_type is None:
            return None
        if split_type == "numeric":
            if threshold is None:
                return None
            op = "<=" if edge == "left" else ">"
            return f"{feature_name} {op} {self._format_export_edge_value(float(threshold))}"
        if split_type == "nominal":
            return f"{feature_name} = {self._format_export_edge_value(edge)}"
        return None

    def _export_branch_label(
        self,
        split_type: Optional[str],
        threshold: Optional[float],
        edge: Any,
    ) -> Optional[str]:
        if split_type == "numeric":
            if threshold is None:
                return None
            op = "<=" if edge == "left" else ">"
            return f"{op} {self._format_export_edge_value(float(threshold))}"
        if split_type == "nominal":
            return f"= {self._format_export_edge_value(edge)}"
        return None

    def _export_node(
        self,
        node: _Node,
        depth: int,
        next_id: list[int],
        branch_condition: Optional[str] = None,
        path_conditions: Optional[list[str]] = None,
        child_index: Optional[int] = None,
    ) -> dict[str, Any]:
        node_id = int(next_id[0])
        next_id[0] += 1
        current_path = list(path_conditions or [])
        exported = {
            "node_id": node_id,
            "depth": int(depth),
            "child_index": None if child_index is None else int(child_index),
            "is_leaf": bool(node.is_leaf),
            "prediction": _to_jsonable(node.prediction),
            "prediction_idx": None if node.prediction_idx is None else int(node.prediction_idx),
            "n_samples": float(self._node_n_samples(node)),
            "branch_condition": branch_condition,
            "path_conditions": list(current_path),
            "class_counts": None
            if node.class_counts is None
            else [float(v) for v in np.asarray(node.class_counts, dtype=np.float64).tolist()],
            "probability_counts": None
            if node.probability_counts is None
            else [float(v) for v in np.asarray(node.probability_counts, dtype=np.float64).tolist()],
            "feature_index": None if node.feature_index is None else int(node.feature_index),
            "feature_name": self._feature_name(node.feature_index),
            "split_type": node.split_type,
            "threshold": None if node.threshold is None else float(node.threshold),
            "left_prob": float(node.left_prob),
            "missing_go_to_left": bool(node.missing_go_to_left),
            "split_gain_ratio": None if node.split_gain_ratio is None else float(node.split_gain_ratio),
            "split_info_gain": None if node.split_info_gain is None else float(node.split_info_gain),
            "split_info_gain_adj": None if node.split_info_gain_adj is None else float(node.split_info_gain_adj),
            "split_intrinsic_value": None if node.split_intrinsic_value is None else float(node.split_intrinsic_value),
            "split_known_weight": None if node.split_known_weight is None else float(node.split_known_weight),
            "split_missing_weight": None if node.split_missing_weight is None else float(node.split_missing_weight),
            "children": [],
        }
        if node.split_type == "nominal" and node.nominal_child_probs is not None:
            exported["nominal_child_probs"] = [
                {
                    "value": _to_jsonable(value),
                    "probability": float(prob),
                }
                for value, prob in node.nominal_child_probs.items()
            ]
            exported["nominal_default_child"] = _to_jsonable(node.nominal_default_child)

        for idx, (edge, child) in enumerate(self._iter_child_items(node)):
            child_condition = self._export_branch_condition(
                node.split_type,
                node.feature_index,
                node.threshold,
                edge,
            )
            child_path = current_path + ([child_condition] if child_condition is not None else [])
            exported["children"].append(
                {
                    "edge": _to_jsonable(edge),
                    "branch_label": self._export_branch_label(node.split_type, node.threshold, edge),
                    "condition_text": child_condition,
                    "child_index": int(idx),
                    "child": self._export_node(
                        child,
                        depth + 1,
                        next_id,
                        branch_condition=child_condition,
                        path_conditions=child_path,
                        child_index=idx,
                    ),
                }
            )
        return exported

    def export_tree(self) -> dict[str, Any]:
        if self.root_ is None:
            return {"root": None, "stats": self.get_tree_stats()}
        return {
            "root": self._export_node(self.root_, 0, [0]),
            "stats": self.get_tree_stats(),
            "classes": [_to_jsonable(v) for v in np.asarray(self.classes_, dtype=object).tolist()],
            "nominal_features": sorted(int(v) for v in self._nominal_features_),
            "feature_names": list(self._feature_names_ or []),
            "config": {
                "enable_pruning": bool(self.enable_pruning),
                "reduced_error_pruning": bool(self.reduced_error_pruning),
                "num_folds": int(self.num_folds),
                "confidence_factor": float(self.confidence_factor),
                "collapse_tree": bool(self.collapse_tree),
                "enable_subtree_raising": bool(self.enable_subtree_raising),
                "enable_fractional_missing": bool(self.enable_fractional_missing),
                "make_split_point_actual_value": bool(self.make_split_point_actual_value),
                "use_laplace": bool(self.use_laplace),
                "use_mdl_correction": bool(self.use_mdl_correction),
                "gain_prefilter_slack": float(self.gain_prefilter_slack),
                "binary_splits": bool(self.binary_splits),
                "cleanup": bool(self.cleanup),
            },
        }

    def get_split_debug_trace(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._split_debug_trace_))

    def iter_tree_nodes(self) -> list[dict[str, Any]]:
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
