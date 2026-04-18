"""Unit tests for the Workstream E validation rule engine.

Uses a lightweight fake cursor so the rules can be exercised without a live
Postgres instance. Each rule's pass and fail path is covered at least once.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.dataops import validate_data as vd


# ---------------------------------------------------------------------------
# Fake cursor — scripted responses, no SQL execution.
# ---------------------------------------------------------------------------


class FakeCursor:
    """Queue-backed cursor stand-in: each execute() pops the next scripted row.

    `one` values feed fetchone(); `many` feed fetchall(). Scripts are popped
    left-to-right, so the test sets them up in the same order the rule queries.
    """

    def __init__(self, *, ones: list[Any] | None = None, manys: list[Any] | None = None) -> None:
        self._ones = list(ones or [])
        self._manys = list(manys or [])
        self.executed: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchone(self) -> Any:
        if not self._ones:
            raise AssertionError("FakeCursor.fetchone() called with no scripted rows left")
        return self._ones.pop(0)

    def fetchall(self) -> Any:
        if not self._manys:
            raise AssertionError("FakeCursor.fetchall() called with no scripted rows left")
        return self._manys.pop(0)


# ---------------------------------------------------------------------------
# RuleResult invariants
# ---------------------------------------------------------------------------


def test_rule_result_rejects_invalid_severity() -> None:
    with pytest.raises(ValueError):
        vd.RuleResult("r", "critical", vd.PASS)


def test_rule_result_rejects_invalid_status() -> None:
    with pytest.raises(ValueError):
        vd.RuleResult("r", vd.WARNING, "unknown")


# ---------------------------------------------------------------------------
# check_schema
# ---------------------------------------------------------------------------


def test_check_schema_passes_when_all_required_columns_present_with_matching_types() -> None:
    cur = FakeCursor(
        manys=[
            [("msno", "text"), ("is_churn", "integer"), ("extra", "text")],
        ]
    )
    cfg = {
        "table": "processed.customer_features",
        "required_columns": {"msno": "text", "is_churn": "integer"},
    }
    result = vd.check_schema(cur, cfg)

    assert result.status == vd.PASS
    assert result.severity == vd.BLOCKING
    assert result.detail["missing_columns"] == []
    assert result.detail["mismatched_types"] == []


def test_check_schema_flags_missing_and_mismatched_columns() -> None:
    cur = FakeCursor(manys=[[("msno", "text"), ("is_churn", "text")]])
    cfg = {
        "table": "processed.customer_features",
        "required_columns": {
            "msno": "text",
            "is_churn": "integer",  # type mismatch in actual schema
            "transaction_count": "integer",  # missing entirely
        },
    }
    result = vd.check_schema(cur, cfg)

    assert result.status == vd.FAIL
    assert result.detail["missing_columns"] == ["transaction_count"]
    assert result.detail["mismatched_types"] == [
        {"column": "is_churn", "expected": "integer", "actual": "text"}
    ]


# ---------------------------------------------------------------------------
# check_freshness
# ---------------------------------------------------------------------------


def test_check_freshness_passes_when_latest_within_window() -> None:
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    latest = now - timedelta(hours=3)
    cur = FakeCursor(ones=[(latest, now, 42)])
    cfg = {"table": "raw.members", "max_age_hours": 24}

    result = vd.check_freshness(cur, cfg)

    assert result.status == vd.PASS
    assert result.severity == vd.WARNING
    assert result.detail["age_hours"] == pytest.approx(3.0, rel=1e-6)


def test_check_freshness_fails_when_stale() -> None:
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    latest = now - timedelta(hours=48)
    cur = FakeCursor(ones=[(latest, now, 42)])
    cfg = {"table": "raw.members", "max_age_hours": 24}

    result = vd.check_freshness(cur, cfg)

    assert result.status == vd.FAIL
    assert result.detail["age_hours"] == pytest.approx(48.0, rel=1e-6)


def test_check_freshness_fails_on_empty_table() -> None:
    now = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    cur = FakeCursor(ones=[(None, now, 0)])
    cfg = {"table": "raw.members", "max_age_hours": 24}

    result = vd.check_freshness(cur, cfg)

    assert result.status == vd.FAIL
    assert result.detail["reason"] == "empty_table_or_null_timestamp"


# ---------------------------------------------------------------------------
# check_coverage
# ---------------------------------------------------------------------------


def test_check_coverage_passes_when_above_min_ratio() -> None:
    cur = FakeCursor(ones=[(1000, 990)])
    cfg = {
        "table": "processed.customer_features",
        "column": "transaction_count",
        "min_non_null_ratio": 0.95,
    }
    result = vd.check_coverage(cur, cfg)

    assert result.status == vd.PASS
    assert result.detail["non_null_ratio"] == 0.99


def test_check_coverage_fails_when_below_min_ratio() -> None:
    cur = FakeCursor(ones=[(1000, 500)])
    cfg = {
        "table": "processed.customer_features",
        "column": "transaction_count",
        "min_non_null_ratio": 0.95,
    }
    result = vd.check_coverage(cur, cfg)

    assert result.status == vd.FAIL


def test_check_coverage_fails_on_empty_table() -> None:
    cur = FakeCursor(ones=[(0, 0)])
    cfg = {
        "table": "processed.customer_features",
        "column": "transaction_count",
        "min_non_null_ratio": 0.95,
    }
    result = vd.check_coverage(cur, cfg)

    assert result.status == vd.FAIL
    assert result.detail["reason"] == "empty_table"


# ---------------------------------------------------------------------------
# check_distribution_column
# ---------------------------------------------------------------------------


def test_check_distribution_first_run_records_baseline() -> None:
    # First execute: AVG/STDDEV/COUNT. Second: baseline lookup (no prior row).
    cur = FakeCursor(ones=[(10.0, 2.0, 1000), None])
    cfg = {
        "table": "processed.customer_features",
        "column": "transaction_count",
        "deviation_threshold": 2.0,
    }
    result = vd.check_distribution_column(cur, cfg)

    assert result.status == vd.PASS
    assert result.detail["baseline"] is None
    assert result.detail["mean"] == pytest.approx(10.0)


def test_check_distribution_passes_when_within_threshold() -> None:
    baseline = {"mean": 10.0, "std": 2.0, "n": 1000}
    cur = FakeCursor(ones=[(11.0, 2.0, 1000), (json.dumps(baseline),)])
    cfg = {
        "table": "processed.customer_features",
        "column": "transaction_count",
        "deviation_threshold": 2.0,
    }
    result = vd.check_distribution_column(cur, cfg)

    assert result.status == vd.PASS
    # |11 - 10| / 2 = 0.5 ≤ 2
    assert result.detail["deviation"] == pytest.approx(0.5)


def test_check_distribution_fails_when_beyond_threshold() -> None:
    baseline = {"mean": 10.0, "std": 1.0, "n": 1000}
    cur = FakeCursor(ones=[(20.0, 1.0, 1000), (json.dumps(baseline),)])
    cfg = {
        "table": "processed.customer_features",
        "column": "transaction_count",
        "deviation_threshold": 2.0,
    }
    result = vd.check_distribution_column(cur, cfg)

    assert result.status == vd.FAIL
    # |20 - 10| / 1 = 10 > 2
    assert result.detail["deviation"] == pytest.approx(10.0)


def test_check_distribution_skips_when_no_rows() -> None:
    cur = FakeCursor(ones=[(None, None, 0)])
    cfg = {
        "table": "processed.customer_features",
        "column": "transaction_count",
    }
    result = vd.check_distribution_column(cur, cfg)

    assert result.status == vd.SKIP
    assert result.detail["reason"] == "no_values"


# ---------------------------------------------------------------------------
# Smoke rules
# ---------------------------------------------------------------------------


def test_smoke_row_count_positive_fails_on_empty_table() -> None:
    cur = FakeCursor(ones=[(0,)])
    cfg = {"rule_name": "row_count_positive", "table": "processed.customer_features"}
    result = vd.smoke_row_count_positive(cur, cfg)
    assert result.status == vd.FAIL


def test_smoke_row_count_match_fails_when_counts_differ() -> None:
    cur = FakeCursor(ones=[(100,), (95,)])
    cfg = {
        "rule_name": "row_count_matches_train",
        "left_table": "processed.customer_features",
        "right_table": "staging.train",
    }
    result = vd.smoke_row_count_match(cur, cfg)
    assert result.status == vd.FAIL
    assert result.detail["left_count"] == 100
    assert result.detail["right_count"] == 95


def test_smoke_value_in_set_fails_when_violators_present() -> None:
    cur = FakeCursor(ones=[(3,)])
    cfg = {
        "rule_name": "is_churn_binary",
        "table": "processed.customer_features",
        "column": "is_churn",
        "allowed_values": [0, 1],
    }
    result = vd.smoke_value_in_set(cur, cfg)
    assert result.status == vd.FAIL
    assert result.detail["violations"] == 3


def test_smoke_unique_column_passes_when_no_duplicates() -> None:
    cur = FakeCursor(ones=[(0,)])
    cfg = {
        "rule_name": "no_duplicate_msno",
        "table": "processed.customer_features",
        "column": "msno",
    }
    result = vd.smoke_unique_column(cur, cfg)
    assert result.status == vd.PASS


def test_smoke_numeric_range_detects_out_of_range_rows() -> None:
    cur = FakeCursor(ones=[(7,)])
    cfg = {
        "rule_name": "bd_in_plausible_range",
        "table": "processed.customer_features",
        "column": "bd",
        "min": 10,
        "max": 100,
        "severity": vd.WARNING,
    }
    result = vd.smoke_numeric_range(cur, cfg)
    assert result.status == vd.FAIL
    assert result.severity == vd.WARNING


def test_smoke_numeric_range_requires_min_or_max() -> None:
    cur = FakeCursor()
    with pytest.raises(ValueError):
        vd.smoke_numeric_range(cur, {"rule_name": "x", "table": "t", "column": "c"})


# ---------------------------------------------------------------------------
# Exit-code + orchestration
# ---------------------------------------------------------------------------


def test_resolve_exit_code_zero_when_no_blocking_failures() -> None:
    results = [
        vd.RuleResult("a", vd.BLOCKING, vd.PASS),
        vd.RuleResult("b", vd.WARNING, vd.FAIL),
        vd.RuleResult("c", vd.BLOCKING, vd.SKIP),
    ]
    assert vd.resolve_exit_code(results) == 0


def test_resolve_exit_code_nonzero_on_blocking_failure() -> None:
    results = [
        vd.RuleResult("a", vd.WARNING, vd.FAIL),
        vd.RuleResult("b", vd.BLOCKING, vd.FAIL),
    ]
    assert vd.resolve_exit_code(results) == 1


def test_run_all_rules_dispatches_each_section(monkeypatch) -> None:
    """Smoke-check the orchestrator hits every rule category exactly once."""
    calls: list[str] = []

    def fake_schema(cur, cfg):
        calls.append("schema")
        return vd.RuleResult("schema::t", vd.BLOCKING, vd.PASS)

    def fake_freshness(cur, cfg):
        calls.append("freshness")
        return vd.RuleResult("freshness::t", vd.WARNING, vd.PASS)

    def fake_coverage(cur, cfg):
        calls.append("coverage")
        return vd.RuleResult("coverage::t::c", vd.WARNING, vd.PASS)

    def fake_distribution(cur, cfg):
        calls.append(f"distribution::{cfg['column']}")
        return vd.RuleResult(f"distribution::{cfg['column']}", vd.WARNING, vd.PASS)

    monkeypatch.setattr(vd, "check_schema", fake_schema)
    monkeypatch.setattr(vd, "check_freshness", fake_freshness)
    monkeypatch.setattr(vd, "check_coverage", fake_coverage)
    monkeypatch.setattr(vd, "check_distribution_column", fake_distribution)

    cfg = {
        "schema": [{"table": "t"}],
        "freshness": [{"table": "t", "max_age_hours": 24}],
        "coverage": [{"table": "t", "column": "c", "min_non_null_ratio": 0.9}],
        "distribution": {"table": "t", "numeric_columns": ["x", "y"]},
        "smoke": [],
    }
    results = vd.run_all_rules(FakeCursor(), cfg)

    assert calls == ["schema", "freshness", "coverage", "distribution::x", "distribution::y"]
    assert len(results) == 5


def test_run_all_rules_records_unknown_smoke_kind_as_fail() -> None:
    cfg = {"smoke": [{"kind": "does_not_exist", "rule_name": "bad"}]}
    results = vd.run_all_rules(FakeCursor(), cfg)
    assert len(results) == 1
    assert results[0].status == vd.FAIL
    assert "unknown smoke kind" in results[0].detail["reason"]
