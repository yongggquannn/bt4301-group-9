"""Shared monitoring utilities: PSI computation and ROC AUC calculation."""

from __future__ import annotations

import numpy as np


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC for binary classification without sklearn dependency."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)

    if pos == 0 or neg == 0:
        raise ValueError("AUC needs both positive and negative classes.")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    distinct_idx = np.where(np.diff(s_sorted))[0]
    threshold_idx = np.r_[distinct_idx, y_sorted.size - 1]

    tpr = np.r_[0.0, tps[threshold_idx] / pos, 1.0]
    fpr = np.r_[0.0, fps[threshold_idx] / neg, 1.0]
    return float(np.trapz(tpr, fpr))


def compute_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between baseline and current distributions."""
    baseline = np.asarray(baseline, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]

    if baseline.size < 2 or current.size < 2:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(baseline, quantiles)
    edges = np.unique(edges)

    if edges.size < 2:
        return 0.0

    # Ensure all out-of-range current values are counted in tail bins.
    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf

    baseline_hist, _ = np.histogram(baseline, bins=edges)
    current_hist, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    baseline_pct = (baseline_hist / max(1, baseline_hist.sum())).astype(np.float64) + eps
    current_pct = (current_hist / max(1, current_hist.sum())).astype(np.float64) + eps

    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
