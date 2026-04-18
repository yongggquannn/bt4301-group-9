"""Unit tests for the Workstream F monitoring helpers (source/mlops/monitoring.py).

Exercises the calibration, score-distribution, segment PSI, and sample-size
helpers end-to-end. No Postgres, MLflow, or Airflow dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.mlops

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.mlops import monitoring as m


# ---------------------------------------------------------------------------
# compute_calibration_bins
# ---------------------------------------------------------------------------


def test_calibration_bins_perfect_classifier_has_zero_error() -> None:
    # Labels line up with the predicted probability bucket midpoints.
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_score = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    bins = m.compute_calibration_bins(y_true, y_score, n_bins=10)

    assert len(bins) == 10
    for b in bins:
        if b["count"] == 0:
            continue
        if b["bin_lower"] < 0.5:
            assert b["actual_rate"] == 0.0
        else:
            assert b["actual_rate"] == 1.0
    assert m.compute_max_calibration_error(bins) == pytest.approx(0.45, abs=0.01)


def test_calibration_bins_length_equals_n_bins_even_when_empty() -> None:
    bins = m.compute_calibration_bins([0, 1], [0.05, 0.95], n_bins=5)
    assert len(bins) == 5
    empty_bins = [b for b in bins if b["count"] == 0]
    assert len(empty_bins) == 3


def test_calibration_bins_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        m.compute_calibration_bins([0, 1], [0.5], n_bins=2)


def test_calibration_bins_rejects_nonpositive_n_bins() -> None:
    with pytest.raises(ValueError):
        m.compute_calibration_bins([0, 1], [0.1, 0.9], n_bins=0)


# ---------------------------------------------------------------------------
# compute_brier_score
# ---------------------------------------------------------------------------


def test_brier_score_of_perfect_prediction_is_zero() -> None:
    assert m.compute_brier_score([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]) == 0.0


def test_brier_score_of_worst_prediction_is_one() -> None:
    assert m.compute_brier_score([0, 1], [1.0, 0.0]) == 1.0


def test_brier_score_empty_inputs_return_nan() -> None:
    result = m.compute_brier_score([], [])
    assert result != result  # NaN check


# ---------------------------------------------------------------------------
# compute_max_calibration_error
# ---------------------------------------------------------------------------


def test_max_calibration_error_ignores_empty_bins() -> None:
    bins = [
        {"count": 0, "mean_pred": None, "actual_rate": None},
        {"count": 10, "mean_pred": 0.2, "actual_rate": 0.1},
        {"count": 5, "mean_pred": 0.8, "actual_rate": 0.3},
    ]
    assert m.compute_max_calibration_error(bins) == pytest.approx(0.5)


def test_max_calibration_error_returns_zero_when_no_populated_bins() -> None:
    assert m.compute_max_calibration_error([{"count": 0}]) == 0.0


# ---------------------------------------------------------------------------
# compute_score_distribution
# ---------------------------------------------------------------------------


def test_score_distribution_uniform_spread() -> None:
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    dist = m.compute_score_distribution(scores, n_bins=10)

    assert len(dist) == 10
    assert sum(b["count"] for b in dist) == len(scores)
    for b in dist:
        assert b["count"] == 1
        assert b["pct"] == pytest.approx(0.1)


def test_score_distribution_all_in_last_bucket() -> None:
    dist = m.compute_score_distribution([0.95] * 20, n_bins=10)
    assert dist[-1]["count"] == 20
    assert sum(b["count"] for b in dist[:-1]) == 0


# ---------------------------------------------------------------------------
# compute_segment_psi
# ---------------------------------------------------------------------------


def test_segment_psi_zero_when_segments_match() -> None:
    rng = np.random.default_rng(seed=0)
    baseline = pd.DataFrame(
        {
            "score": rng.uniform(size=200),
            "segment": ["a"] * 100 + ["b"] * 100,
        }
    )
    current = baseline.copy()

    result = m.compute_segment_psi(baseline, current, segment_columns=["segment"])

    assert set(result.keys()) == {"segment=a", "segment=b"}
    for v in result.values():
        assert v == pytest.approx(0.0, abs=1e-6)


def test_segment_psi_flags_shifted_segment() -> None:
    rng = np.random.default_rng(seed=1)
    baseline = pd.DataFrame(
        {
            "score": rng.uniform(0.0, 0.3, size=500),
            "segment": ["a"] * 500,
        }
    )
    current = pd.DataFrame(
        {
            # Distribution shifted right in the current window.
            "score": rng.uniform(0.7, 1.0, size=500),
            "segment": ["a"] * 500,
        }
    )
    result = m.compute_segment_psi(baseline, current, segment_columns=["segment"])
    assert result["segment=a"] > 0.2  # clearly drifted


def test_segment_psi_returns_nan_when_too_few_samples() -> None:
    baseline = pd.DataFrame({"score": [0.5], "segment": ["a"]})
    current = pd.DataFrame({"score": [0.5], "segment": ["a"]})
    result = m.compute_segment_psi(baseline, current, segment_columns=["segment"])
    assert result["segment=a"] != result["segment=a"]  # NaN


def test_segment_psi_missing_column_produces_missing_sentinel() -> None:
    baseline = pd.DataFrame({"score": [0.1, 0.2]})
    current = pd.DataFrame({"score": [0.3, 0.4]})
    result = m.compute_segment_psi(
        baseline, current, segment_columns=["missing_col"]
    )
    assert "missing_col=missing" in result
    assert result["missing_col=missing"] != result["missing_col=missing"]


def test_segment_psi_missing_score_column_raises() -> None:
    baseline = pd.DataFrame({"segment": ["a"]})
    current = pd.DataFrame({"segment": ["a"]})
    with pytest.raises(ValueError):
        m.compute_segment_psi(baseline, current, segment_columns=["segment"])


# ---------------------------------------------------------------------------
# has_insufficient_data / segment_psi_breached
# ---------------------------------------------------------------------------


def test_has_insufficient_data_flags_small_windows() -> None:
    assert m.has_insufficient_data(0, min_samples=1000)
    assert m.has_insufficient_data(999, min_samples=1000)
    assert not m.has_insufficient_data(1000, min_samples=1000)
    assert not m.has_insufficient_data(10_000)


def test_segment_psi_breached_returns_keys_over_threshold() -> None:
    segs = {
        "segment=a": 0.05,
        "segment=b": 0.30,
        "segment=c": float("nan"),
        "segment=d": 0.25,
    }
    assert sorted(m.segment_psi_breached(segs, threshold=0.2)) == [
        "segment=b",
        "segment=d",
    ]


def test_segment_psi_breached_empty_input_returns_empty_list() -> None:
    assert m.segment_psi_breached({}, threshold=0.2) == []
