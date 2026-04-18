"""Monitoring helpers for Workstream F.

Pure functions for calibration, score-distribution, and segment-level drift
monitoring used by ``weekly_model_monitoring_dag``. Kept numpy/pandas-only so
the module can be imported without the full MLflow/sklearn stack.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from source.common.monitoring_utils import compute_psi

# Minimum labelled rows required before a monitoring verdict is recorded.
MIN_LABELED_SAMPLES: int = int(os.getenv("MONITORING_MIN_SAMPLES", "1000"))

# Reliability bin breach threshold for calibration monitoring.
MAX_CALIBRATION_ERROR_THRESHOLD: float = float(
    os.getenv("MONITORING_MAX_CAL_ERROR", "0.10")
)

# Segment-level PSI breach threshold (matches feature-level threshold).
SEGMENT_PSI_THRESHOLD: float = float(os.getenv("MONITORING_SEGMENT_PSI", "0.2"))

# Status string written to ``processed.model_monitoring_results.status`` when
# the current cohort is too small for a trustworthy verdict.
INSUFFICIENT_DATA_STATUS: str = "insufficient_data"

# Status string for a fully-populated monitoring row (no breaches).
OK_STATUS: str = "ok"

# Status string used when at least one threshold breached.
BREACHED_STATUS: str = "breached"


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float64)


def _as_int_array(values: Iterable[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def compute_calibration_bins(
    y_true: Iterable[int],
    y_score: Iterable[float],
    n_bins: int = 10,
) -> list[dict]:
    """Return reliability-diagram bins for a binary classifier.

    Each bin dict has: ``bin_lower``, ``bin_upper``, ``mean_pred``,
    ``actual_rate``, ``count``. Uses fixed equal-width bins over ``[0, 1]``.
    Empty bins are emitted with ``count=0`` and ``mean_pred=actual_rate=None``
    so the output length is always ``n_bins``.
    """
    y_true_arr = _as_int_array(y_true)
    y_score_arr = _as_float_array(y_score)

    if y_true_arr.size != y_score_arr.size:
        raise ValueError(
            "compute_calibration_bins: y_true and y_score must have equal length"
        )
    if n_bins <= 0:
        raise ValueError("compute_calibration_bins: n_bins must be positive")

    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    # np.digitize: bin 1 for edges[0] <= x < edges[1], bin n_bins for top edge.
    # Clip scores into [0, 1] defensively.
    clipped = np.clip(y_score_arr, 0.0, 1.0)
    # Use right=False so 0.0 maps to bin 1 and 1.0 rolls into last bin via clamp.
    idx = np.clip(np.digitize(clipped, edges[1:-1], right=False), 0, n_bins - 1)

    bins: list[dict] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_lower": float(edges[b]),
                    "bin_upper": float(edges[b + 1]),
                    "mean_pred": None,
                    "actual_rate": None,
                    "count": 0,
                }
            )
            continue

        bins.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_pred": float(y_score_arr[mask].mean()),
                "actual_rate": float(y_true_arr[mask].mean()),
                "count": count,
            }
        )

    return bins


def compute_brier_score(
    y_true: Iterable[int], y_score: Iterable[float]
) -> float:
    """Compute the Brier score (mean squared error between labels and probs)."""
    y_true_arr = _as_int_array(y_true).astype(np.float64)
    y_score_arr = _as_float_array(y_score)
    if y_true_arr.size == 0:
        return float("nan")
    if y_true_arr.size != y_score_arr.size:
        raise ValueError(
            "compute_brier_score: y_true and y_score must have equal length"
        )
    return float(np.mean((y_score_arr - y_true_arr) ** 2))


def compute_max_calibration_error(bins: Sequence[Mapping[str, object]]) -> float:
    """Max absolute gap between mean_pred and actual_rate across non-empty bins."""
    gaps = []
    for b in bins:
        if b.get("count", 0) == 0:
            continue
        mean_pred = b.get("mean_pred")
        actual = b.get("actual_rate")
        if mean_pred is None or actual is None:
            continue
        gaps.append(abs(float(mean_pred) - float(actual)))
    return float(max(gaps)) if gaps else 0.0


def compute_score_distribution(
    y_score: Iterable[float], n_bins: int = 10
) -> list[dict]:
    """Return decile counts: ``[{bin_lower, bin_upper, count, pct}]``."""
    y_score_arr = _as_float_array(y_score)
    if n_bins <= 0:
        raise ValueError("compute_score_distribution: n_bins must be positive")

    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    total = max(int(y_score_arr.size), 1)
    clipped = np.clip(y_score_arr, 0.0, 1.0)
    counts, _ = np.histogram(clipped, bins=edges)

    return [
        {
            "bin_lower": float(edges[i]),
            "bin_upper": float(edges[i + 1]),
            "count": int(counts[i]),
            "pct": float(counts[i]) / total,
        }
        for i in range(n_bins)
    ]


def compute_segment_psi(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    segment_columns: Sequence[str],
    score_column: str = "score",
) -> dict[str, float]:
    """Compute PSI on ``score_column`` within each value of each segment column.

    Returns a flat dict keyed ``"{column}={value}"`` mapping to the PSI value.
    Segments that are missing entirely from either side receive a ``nan`` so
    downstream code can distinguish "no drift" (0.0) from "not comparable".
    """
    if score_column not in baseline.columns or score_column not in current.columns:
        raise ValueError(
            f"compute_segment_psi: '{score_column}' missing from baseline or current"
        )

    results: dict[str, float] = {}
    for col in segment_columns:
        if col not in baseline.columns or col not in current.columns:
            results[f"{col}=missing"] = float("nan")
            continue

        values = pd.unique(
            pd.concat([baseline[col], current[col]], ignore_index=True).dropna()
        )
        for value in values:
            baseline_slice = baseline.loc[baseline[col] == value, score_column].to_numpy(
                dtype=np.float64, copy=False
            )
            current_slice = current.loc[current[col] == value, score_column].to_numpy(
                dtype=np.float64, copy=False
            )
            key = f"{col}={value}"
            if baseline_slice.size < 2 or current_slice.size < 2:
                results[key] = float("nan")
                continue
            results[key] = compute_psi(baseline_slice, current_slice)

    return results


def has_insufficient_data(
    current_row_count: int,
    min_samples: int | None = None,
) -> bool:
    """Return True when the current cohort is below the monitoring guard."""
    threshold = int(min_samples if min_samples is not None else MIN_LABELED_SAMPLES)
    return int(current_row_count) < threshold


def segment_psi_breached(
    segment_psi: Mapping[str, float],
    threshold: float | None = None,
) -> list[str]:
    """Return the list of segment keys whose PSI exceeds the threshold."""
    limit = float(threshold if threshold is not None else SEGMENT_PSI_THRESHOLD)
    breached: list[str] = []
    for key, value in segment_psi.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v):
            continue
        if v > limit:
            breached.append(key)
    return breached


__all__ = [
    "MIN_LABELED_SAMPLES",
    "MAX_CALIBRATION_ERROR_THRESHOLD",
    "SEGMENT_PSI_THRESHOLD",
    "INSUFFICIENT_DATA_STATUS",
    "OK_STATUS",
    "BREACHED_STATUS",
    "compute_calibration_bins",
    "compute_brier_score",
    "compute_max_calibration_error",
    "compute_score_distribution",
    "compute_segment_psi",
    "has_insufficient_data",
    "segment_psi_breached",
]
