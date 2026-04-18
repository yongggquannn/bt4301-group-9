"""Unit tests for the Workstream F business-metric promotion gate in
``source/mlops/register_model.py``.

We only import and test the pure helper — no MLflow client, no registry, no
network. The gate function is intentionally side-effect free for easy testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.mlops

pytest.importorskip("mlflow")  # register_model imports mlflow at module level
pytest.importorskip("numpy")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.mlops.register_model import (
    BUSINESS_METRIC_TOLERANCE,
    evaluate_business_metric_gate,
)


def test_challenger_improves_business_metric_passes_gate() -> None:
    allow, delta = evaluate_business_metric_gate(0.82, 0.78)
    assert allow is True
    assert delta == pytest.approx(0.04)


def test_challenger_within_tolerance_still_passes() -> None:
    # Challenger regressed by 0.005, within the 0.01 tolerance → allowed.
    allow, delta = evaluate_business_metric_gate(0.775, 0.78, tolerance=0.01)
    assert allow is True
    assert delta == pytest.approx(-0.005)


def test_challenger_beyond_tolerance_blocks_promotion() -> None:
    allow, delta = evaluate_business_metric_gate(0.65, 0.80, tolerance=0.01)
    assert allow is False
    assert delta == pytest.approx(-0.15)


def test_missing_challenger_metric_allows_promotion() -> None:
    allow, delta = evaluate_business_metric_gate(None, 0.80)
    assert allow is True
    assert delta is None


def test_missing_champion_metric_allows_promotion() -> None:
    allow, delta = evaluate_business_metric_gate(0.80, None)
    assert allow is True
    assert delta is None


def test_tolerance_absolute_value_handles_negative_input() -> None:
    # A negative tolerance should behave the same as its absolute value.
    allow, _ = evaluate_business_metric_gate(0.775, 0.78, tolerance=-0.01)
    assert allow is True


def test_default_tolerance_is_reasonable() -> None:
    # Sanity: the default tolerance is a small positive number.
    assert 0.0 < BUSINESS_METRIC_TOLERANCE < 0.1
