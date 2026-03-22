from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, log_loss


BLOCK_THRESHOLDS: dict[str, dict[str, float]] = {
    "B": {
        "match_median_min": 0.99,
        "match_min_min": 0.97,
        "prob_median_max": 0.01,
        "prob_max_max": 0.03,
        "delta_balanced_accuracy_abs_max": 0.005,
        "delta_macro_f1_abs_max": 0.01,
        "delta_log_loss_abs_max": 0.02,
    },
    "C": {
        "match_median_min": 0.98,
        "match_min_min": 0.95,
        "prob_median_max": 0.02,
        "prob_max_max": 0.05,
        "delta_balanced_accuracy_abs_max": 0.01,
        "delta_macro_f1_abs_max": 0.015,
        "delta_log_loss_abs_max": 0.03,
    },
}

DEFAULT_DATASET_BLOCKS: dict[str, str] = {
    "NSL-KDD": "C",
    "CIC-IDS2017": "C",
    "UNSW-NB15": "C",
    "CIRA2020-AttNorm": "C",
    "CIRA2020-DoHNDoH": "C",
}

METRICS = ("balanced_accuracy", "macro_f1", "log_loss")


@dataclass(frozen=True)
class InstanceComparison:
    y_true: np.ndarray
    local_pred: np.ndarray
    weka_pred: np.ndarray
    local_proba: np.ndarray
    weka_proba: np.ndarray
    class_values: list[str]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _clip_proba(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(proba, dtype=np.float64), 1e-15, 1.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return clipped / row_sums


def load_instance_comparison(path: Path) -> InstanceComparison:
    frame = pd.read_csv(path)
    class_values = [c.removeprefix("local_p_") for c in frame.columns if c.startswith("local_p_")]
    local_cols = [f"local_p_{cls}" for cls in class_values]
    weka_cols = [f"weka_p_{cls}" for cls in class_values]
    return InstanceComparison(
        y_true=frame["actual"].astype(str).to_numpy(dtype=object),
        local_pred=frame["local_pred"].astype(str).to_numpy(dtype=object),
        weka_pred=frame["weka_pred"].astype(str).to_numpy(dtype=object),
        local_proba=frame[local_cols].to_numpy(dtype=np.float64),
        weka_proba=frame[weka_cols].to_numpy(dtype=np.float64),
        class_values=list(class_values),
    )


def compute_predictive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_values: list[str],
) -> dict[str, float]:
    recalls = []
    for cls in class_values:
        mask = y_true == cls
        support = int(np.sum(mask))
        if support == 0:
            continue
        recalls.append(float(np.sum(y_pred[mask] == cls) / support))
    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    return {
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=class_values, zero_division=0)),
        "log_loss": float(log_loss(y_true, _clip_proba(y_proba), labels=class_values)),
    }


def compute_mcnemar(y_true: np.ndarray, local_pred: np.ndarray, weka_pred: np.ndarray) -> dict[str, float]:
    local_correct = local_pred == y_true
    weka_correct = weka_pred == y_true
    b = int(np.sum(local_correct & (~weka_correct)))
    c = int(np.sum((~local_correct) & weka_correct))
    n = b + c
    p_value = float(binomtest(min(b, c), n=n, p=0.5).pvalue) if n > 0 else 1.0
    return {
        "b_local_only": float(b),
        "c_weka_only": float(c),
        "discordant_total": float(n),
        "p_value_raw": p_value,
    }


def paired_bootstrap_metric_deltas(
    comp: InstanceComparison,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 1,
) -> dict[str, dict[str, float]]:
    n = comp.y_true.shape[0]
    rng = np.random.RandomState(random_state)
    alpha = 1.0 - confidence_level

    local_base = compute_predictive_metrics(comp.y_true, comp.local_pred, comp.local_proba, comp.class_values)
    weka_base = compute_predictive_metrics(comp.y_true, comp.weka_pred, comp.weka_proba, comp.class_values)
    bootstrap_samples: dict[str, list[float]] = {metric: [] for metric in METRICS}

    for _ in range(int(n_bootstrap)):
        idx = rng.randint(0, n, size=n)
        y_true = comp.y_true[idx]
        local_metrics = compute_predictive_metrics(y_true, comp.local_pred[idx], comp.local_proba[idx], comp.class_values)
        weka_metrics = compute_predictive_metrics(y_true, comp.weka_pred[idx], comp.weka_proba[idx], comp.class_values)
        for metric in METRICS:
            bootstrap_samples[metric].append(float(local_metrics[metric] - weka_metrics[metric]))

    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        samples = np.asarray(bootstrap_samples[metric], dtype=np.float64)
        low = float(np.quantile(samples, alpha / 2.0))
        high = float(np.quantile(samples, 1.0 - alpha / 2.0))
        out[metric] = {
            "local": float(local_base[metric]),
            "weka": float(weka_base[metric]),
            "delta": float(local_base[metric] - weka_base[metric]),
            "ci_low": low,
            "ci_high": high,
        }
    return out


def holm_adjust(records: list[dict[str, Any]], p_key: str, group_keys: Iterable[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = {}
    for idx, record in enumerate(records):
        key = tuple(record.get(group_key) for group_key in group_keys)
        grouped.setdefault(key, []).append((idx, record))

    out = [dict(record) for record in records]
    for items in grouped.values():
        ranked = sorted(items, key=lambda item: float(item[1].get(p_key, 1.0)))
        m = len(ranked)
        adjusted = [0.0] * m
        for j, (_, record) in enumerate(ranked):
            adjusted[j] = min(1.0, (m - j) * float(record.get(p_key, 1.0)))
        for j in range(1, m):
            adjusted[j] = max(adjusted[j], adjusted[j - 1])
        for j, (idx, _) in enumerate(ranked):
            out[idx]["p_value_holm"] = adjusted[j]
            out[idx]["significant_holm_0_05"] = int(adjusted[j] < 0.05)
    return out


def evaluate_acceptance_run(
    summary_row: dict[str, str],
    block: str,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    out_dir = Path(summary_row["out_dir"])
    comp = load_instance_comparison(out_dir / "per_instance_comparison.csv")
    metrics = paired_bootstrap_metric_deltas(
        comp,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=int(summary_row.get("seed", "1") or 1),
    )
    mcnemar = compute_mcnemar(comp.y_true, comp.local_pred, comp.weka_pred)
    thresholds = BLOCK_THRESHOLDS[block]

    row: dict[str, Any] = {
        "dataset": summary_row["dataset"],
        "config": summary_row["config"],
        "seed": int(summary_row["seed"]),
        "block": block,
        "prediction_match_fraction": _safe_float(summary_row.get("prediction_match_fraction")),
        "probability_mean_abs_delta": _safe_float(summary_row.get("probability_mean_abs_delta")),
        "local_node_count": _safe_float(summary_row.get("local_node_count")),
        "weka_node_count": _safe_float(summary_row.get("weka_node_count")),
        **mcnemar,
    }
    for metric in METRICS:
        data = metrics[metric]
        margin = thresholds[f"delta_{metric}_abs_max"]
        row[f"local_{metric}"] = data["local"]
        row[f"weka_{metric}"] = data["weka"]
        row[f"delta_{metric}"] = data["delta"]
        row[f"delta_{metric}_ci_low"] = data["ci_low"]
        row[f"delta_{metric}_ci_high"] = data["ci_high"]
        row[f"ci_accept_{metric}"] = int(data["ci_low"] >= -margin and data["ci_high"] <= margin)
    return row


def aggregate_acceptance_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["dataset"]), str(record["config"]), str(record["block"]))
        grouped.setdefault(key, []).append(record)

    out: list[dict[str, Any]] = []
    for (dataset, config, block), rows in sorted(grouped.items()):
        thresholds = BLOCK_THRESHOLDS[block]

        def median_of(key: str) -> float:
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            return float(np.median(np.asarray(vals, dtype=np.float64)))

        def min_of(key: str) -> float:
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            return float(np.min(np.asarray(vals, dtype=np.float64)))

        def max_of(key: str) -> float:
            vals = [float(row[key]) for row in rows if row.get(key) is not None]
            return float(np.max(np.asarray(vals, dtype=np.float64)))

        agg = {
            "dataset": dataset,
            "config": config,
            "block": block,
            "seed_count": len(rows),
            "median_prediction_match_fraction": median_of("prediction_match_fraction"),
            "min_prediction_match_fraction": min_of("prediction_match_fraction"),
            "median_probability_mean_abs_delta": median_of("probability_mean_abs_delta"),
            "max_probability_mean_abs_delta": max_of("probability_mean_abs_delta"),
            "prediction_match_accept": int(
                median_of("prediction_match_fraction") >= thresholds["match_median_min"]
                and min_of("prediction_match_fraction") >= thresholds["match_min_min"]
            ),
            "probability_accept": int(
                median_of("probability_mean_abs_delta") <= thresholds["prob_median_max"]
                and max_of("probability_mean_abs_delta") <= thresholds["prob_max_max"]
            ),
            "mcnemar_significant_any_holm_0_05": int(any(int(row.get("significant_holm_0_05", 0)) for row in rows)),
        }
        metric_accept_flags = []
        for metric in METRICS:
            agg[f"median_abs_delta_{metric}"] = float(
                np.median(np.asarray([abs(float(row[f"delta_{metric}"])) for row in rows], dtype=np.float64))
            )
            agg[f"max_abs_delta_{metric}"] = max(abs(float(row[f"delta_{metric}"])) for row in rows)
            agg[f"ci_accept_rate_{metric}"] = float(
                sum(int(row[f"ci_accept_{metric}"]) for row in rows) / len(rows)
            )
            agg[f"all_ci_accept_{metric}"] = int(all(int(row[f"ci_accept_{metric}"]) for row in rows))
            metric_accept_flags.append(bool(agg[f"all_ci_accept_{metric}"]))
        agg["overall_accept"] = int(
            bool(agg["prediction_match_accept"])
            and bool(agg["probability_accept"])
            and all(metric_accept_flags)
        )
        out.append(agg)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
