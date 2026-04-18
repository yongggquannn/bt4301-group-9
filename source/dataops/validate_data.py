"""Workstream E — YAML-driven data validation with severity and persistence.

Responsibilities
----------------
1. Schema validation — assert expected columns/types exist on critical tables.
2. Freshness checks — raw-table ``ingestion_timestamp`` must not be stale.
3. Coverage thresholds — per-column non-null ratio minima on feature tables.
4. Distribution drift — compare current numeric stats against the most recent
   persisted baseline in ``processed.validation_results.detail``.
5. Legacy smoke checks — row counts, binary targets, uniqueness, numeric ranges.

Each rule emits a :class:`RuleResult` tagged ``warning`` or ``blocking``. Results
are persisted to ``processed.validation_results`` and, if any blocking rule
fails, the script exits with ``SystemExit(1)`` so the Airflow task fails.

CLI
---
    python source/dataops/validate_data.py
    python source/dataops/validate_data.py --config docs/cleansing_rules.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Allow imports from project root when run standalone or via Airflow.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(PROJECT_ROOT))

from source.common.db import get_connection  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "docs" / "cleansing_rules.yaml"

PASS = "pass"
FAIL = "fail"
SKIP = "skip"

BLOCKING = "blocking"
WARNING = "warning"

VALID_SEVERITIES = {BLOCKING, WARNING}
VALID_STATUSES = {PASS, FAIL, SKIP}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleResult:
    """Single validation rule outcome — immutable, JSON-serialisable."""

    rule_name: str
    severity: str
    status: str
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(f"invalid severity: {self.severity}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {self.status}")


# ---------------------------------------------------------------------------
# Rule checks — all accept ``(cur, cfg)`` and return RuleResult(s)
# ---------------------------------------------------------------------------


def _qualified(name: str) -> tuple[str, str]:
    if "." not in name:
        raise ValueError(f"expected schema-qualified table, got {name!r}")
    schema, table = name.split(".", 1)
    return schema, table


def check_schema(cur, cfg: dict) -> RuleResult:
    """Assert every required column is present with a matching data_type."""
    table = cfg["table"]
    severity = cfg.get("severity", BLOCKING)
    required = cfg.get("required_columns", {}) or {}
    rule_name = f"schema::{table}"

    schema, tbl = _qualified(table)
    cur.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """,
        (schema, tbl),
    )
    actual = {row[0]: row[1] for row in cur.fetchall()}

    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    for col, expected_type in required.items():
        if col not in actual:
            missing.append(col)
            continue
        if actual[col].lower() != str(expected_type).lower():
            mismatched.append(
                {"column": col, "expected": expected_type, "actual": actual[col]}
            )

    status = PASS if not missing and not mismatched else FAIL
    detail = {
        "table": table,
        "missing_columns": missing,
        "mismatched_types": mismatched,
        "actual_column_count": len(actual),
    }
    return RuleResult(rule_name, severity, status, detail)


def check_freshness(cur, cfg: dict) -> RuleResult:
    """Freshness: max(timestamp_column) must be within max_age_hours."""
    table = cfg["table"]
    ts_col = cfg.get("timestamp_column", "ingestion_timestamp")
    max_age_hours = float(cfg["max_age_hours"])
    severity = cfg.get("severity", WARNING)
    rule_name = f"freshness::{table}"

    # NOTE: table + column names come from a trusted YAML config we author;
    # they are never user-supplied. psycopg2 does not support identifier
    # parameter substitution, so f-string interpolation is required here.
    cur.execute(
        f"SELECT MAX({ts_col}), NOW(), COUNT(*) FROM {table}"  # noqa: S608
    )
    latest, now, row_count = cur.fetchone()

    if row_count == 0 or latest is None:
        return RuleResult(
            rule_name,
            severity,
            FAIL,
            {
                "table": table,
                "reason": "empty_table_or_null_timestamp",
                "row_count": row_count,
            },
        )

    age_seconds = (now - latest).total_seconds()
    age_hours = age_seconds / 3600.0
    status = PASS if age_hours <= max_age_hours else FAIL
    return RuleResult(
        rule_name,
        severity,
        status,
        {
            "table": table,
            "latest": latest.isoformat(),
            "checked_at": now.isoformat(),
            "age_hours": round(age_hours, 4),
            "max_age_hours": max_age_hours,
            "row_count": row_count,
        },
    )


def check_coverage(cur, cfg: dict) -> RuleResult:
    """Coverage: non-null ratio for a column must meet the configured minimum."""
    table = cfg["table"]
    column = cfg["column"]
    min_ratio = float(cfg["min_non_null_ratio"])
    severity = cfg.get("severity", WARNING)
    rule_name = f"coverage::{table}::{column}"

    cur.execute(
        f"SELECT COUNT(*), COUNT({column}) FROM {table}"  # noqa: S608
    )
    total, non_null = cur.fetchone()

    if total == 0:
        return RuleResult(
            rule_name,
            severity,
            FAIL,
            {"table": table, "column": column, "reason": "empty_table"},
        )

    ratio = non_null / total
    status = PASS if ratio >= min_ratio else FAIL
    return RuleResult(
        rule_name,
        severity,
        status,
        {
            "table": table,
            "column": column,
            "non_null_ratio": round(ratio, 6),
            "min_non_null_ratio": min_ratio,
            "row_count": total,
            "non_null_count": non_null,
        },
    )


def _load_baseline(cur, rule_name: str) -> dict | None:
    """Pull the most recent prior PASS detail for a distribution rule, if any."""
    cur.execute(
        """
        SELECT detail
        FROM processed.validation_results
        WHERE rule_name = %s AND status = %s
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (rule_name, PASS),
    )
    row = cur.fetchone()
    if row is None:
        return None
    raw = row[0]
    # psycopg2 returns JSONB as dict; tolerate stringified JSON in tests.
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def check_distribution_column(cur, cfg: dict) -> RuleResult:
    """Compare current mean to the last persisted baseline for this column.

    First-ever run records the baseline and reports ``pass``. Later runs fail
    if ``|curr_mean - baseline_mean| / max(baseline_std, epsilon) > threshold``.
    """
    table = cfg["table"]
    column = cfg["column"]
    threshold = float(cfg.get("deviation_threshold", 2.0))
    severity = cfg.get("severity", WARNING)
    rule_name = f"distribution::{table}::{column}"

    cur.execute(
        f"SELECT AVG({column})::float8, STDDEV_POP({column})::float8, "  # noqa: S608
        f"COUNT({column}) FROM {table}"
    )
    curr_mean, curr_std, n = cur.fetchone()

    if n == 0 or curr_mean is None:
        return RuleResult(
            rule_name,
            severity,
            SKIP,
            {"table": table, "column": column, "reason": "no_values"},
        )

    baseline = _load_baseline(cur, rule_name)
    current_stats = {
        "table": table,
        "column": column,
        "mean": float(curr_mean),
        "std": float(curr_std) if curr_std is not None else 0.0,
        "n": int(n),
        "threshold": threshold,
    }

    if baseline is None:
        current_stats["baseline"] = None
        return RuleResult(rule_name, severity, PASS, current_stats)

    baseline_mean = float(baseline.get("mean", 0.0))
    baseline_std = float(baseline.get("std", 0.0))
    epsilon = 1e-9
    denom = baseline_std if baseline_std > epsilon else max(abs(baseline_mean), epsilon)
    deviation = abs(float(curr_mean) - baseline_mean) / denom

    status = PASS if deviation <= threshold else FAIL
    current_stats["baseline"] = {
        "mean": baseline_mean,
        "std": baseline_std,
        "n": baseline.get("n"),
    }
    current_stats["deviation"] = round(deviation, 6)
    return RuleResult(rule_name, severity, status, current_stats)


# ---------------------------------------------------------------------------
# Legacy smoke checks — each expressed as a standalone rule
# ---------------------------------------------------------------------------


def smoke_row_count_positive(cur, cfg: dict) -> RuleResult:
    table = cfg["table"]
    cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
    count = cur.fetchone()[0]
    return RuleResult(
        cfg["rule_name"],
        cfg.get("severity", BLOCKING),
        PASS if count > 0 else FAIL,
        {"table": table, "row_count": count},
    )


def smoke_row_count_match(cur, cfg: dict) -> RuleResult:
    left = cfg["left_table"]
    right = cfg["right_table"]
    cur.execute(f"SELECT COUNT(*) FROM {left}")  # noqa: S608
    left_n = cur.fetchone()[0]
    cur.execute(f"SELECT COUNT(*) FROM {right}")  # noqa: S608
    right_n = cur.fetchone()[0]
    return RuleResult(
        cfg["rule_name"],
        cfg.get("severity", BLOCKING),
        PASS if left_n == right_n else FAIL,
        {"left_table": left, "left_count": left_n, "right_table": right, "right_count": right_n},
    )


def smoke_value_in_set(cur, cfg: dict) -> RuleResult:
    table = cfg["table"]
    column = cfg["column"]
    allowed = cfg["allowed_values"]
    placeholders = ",".join(["%s"] * len(allowed))
    cur.execute(
        f"SELECT COUNT(*) FROM {table} "  # noqa: S608
        f"WHERE {column} IS NULL OR {column} NOT IN ({placeholders})",
        tuple(allowed),
    )
    violators = cur.fetchone()[0]
    return RuleResult(
        cfg["rule_name"],
        cfg.get("severity", BLOCKING),
        PASS if violators == 0 else FAIL,
        {"table": table, "column": column, "allowed_values": list(allowed), "violations": violators},
    )


def smoke_unique_column(cur, cfg: dict) -> RuleResult:
    table = cfg["table"]
    column = cfg["column"]
    cur.execute(
        f"SELECT COUNT(*) FROM ("  # noqa: S608
        f"  SELECT {column} FROM {table} "
        f"  GROUP BY {column} HAVING COUNT(*) > 1"
        f") dupes"
    )
    dup = cur.fetchone()[0]
    return RuleResult(
        cfg["rule_name"],
        cfg.get("severity", BLOCKING),
        PASS if dup == 0 else FAIL,
        {"table": table, "column": column, "duplicates": dup},
    )


def smoke_numeric_range(cur, cfg: dict) -> RuleResult:
    table = cfg["table"]
    column = cfg["column"]
    allow_null = cfg.get("allow_null", True)
    clauses: list[str] = []
    params: list[Any] = []
    if "min" in cfg:
        clauses.append(f"{column} < %s")
        params.append(cfg["min"])
    if "max" in cfg:
        clauses.append(f"{column} > %s")
        params.append(cfg["max"])
    if not clauses:
        raise ValueError("numeric_range rule requires at least one of min/max")

    null_clause = f"{column} IS NOT NULL AND " if allow_null else ""
    where = null_clause + "(" + " OR ".join(clauses) + ")"
    cur.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {where}",  # noqa: S608
        tuple(params),
    )
    violators = cur.fetchone()[0]
    return RuleResult(
        cfg["rule_name"],
        cfg.get("severity", BLOCKING),
        PASS if violators == 0 else FAIL,
        {
            "table": table,
            "column": column,
            "min": cfg.get("min"),
            "max": cfg.get("max"),
            "allow_null": allow_null,
            "violations": violators,
        },
    )


SMOKE_DISPATCH: dict[str, Callable[..., RuleResult]] = {
    "row_count_positive": smoke_row_count_positive,
    "row_count_match": smoke_row_count_match,
    "value_in_set": smoke_value_in_set,
    "unique_column": smoke_unique_column,
    "numeric_range": smoke_numeric_range,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def run_all_rules(cur, validation_cfg: dict) -> list[RuleResult]:
    """Execute every configured rule and collect results (no mutation of cfg)."""
    results: list[RuleResult] = []

    for cfg in validation_cfg.get("schema", []) or []:
        results.append(check_schema(cur, cfg))

    for cfg in validation_cfg.get("freshness", []) or []:
        results.append(check_freshness(cur, cfg))

    for cfg in validation_cfg.get("coverage", []) or []:
        results.append(check_coverage(cur, cfg))

    dist_cfg = validation_cfg.get("distribution") or {}
    if dist_cfg:
        table = dist_cfg["table"]
        severity = dist_cfg.get("severity", WARNING)
        threshold = dist_cfg.get("deviation_threshold", 2.0)
        for column in dist_cfg.get("numeric_columns", []) or []:
            results.append(
                check_distribution_column(
                    cur,
                    {
                        "table": table,
                        "column": column,
                        "severity": severity,
                        "deviation_threshold": threshold,
                    },
                )
            )

    for cfg in validation_cfg.get("smoke", []) or []:
        kind = cfg.get("kind")
        fn = SMOKE_DISPATCH.get(kind)
        if fn is None:
            results.append(
                RuleResult(
                    cfg.get("rule_name", f"smoke::{kind}"),
                    cfg.get("severity", WARNING),
                    FAIL,
                    {"reason": f"unknown smoke kind: {kind}"},
                )
            )
            continue
        results.append(fn(cur, cfg))

    return results


def resolve_exit_code(results: list[RuleResult]) -> int:
    """Return 1 iff any blocking rule failed, else 0."""
    return 1 if any(r.severity == BLOCKING and r.status == FAIL for r in results) else 0


def ensure_results_table(cur) -> None:
    """Create processed.validation_results if it doesn't exist yet.

    Mirrors the migration SQL so running this script against a fresh DB still
    works without manually applying migrations.
    """
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS processed.validation_results (
            result_id   BIGSERIAL   PRIMARY KEY,
            run_id      TEXT        NOT NULL,
            rule_name   TEXT        NOT NULL,
            severity    VARCHAR(16) NOT NULL,
            status      VARCHAR(16) NOT NULL,
            detail      JSONB,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_validation_results_run_id "
        "ON processed.validation_results (run_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_validation_results_created_at "
        "ON processed.validation_results (created_at DESC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_validation_results_rule_name "
        "ON processed.validation_results (rule_name, created_at DESC)"
    )


def persist_results(cur, run_id: str, results: list[RuleResult]) -> None:
    if not results:
        return
    rows = [
        (run_id, r.rule_name, r.severity, r.status, json.dumps(r.detail))
        for r in results
    ]
    cur.executemany(
        """
        INSERT INTO processed.validation_results
            (run_id, rule_name, severity, status, detail)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        """,
        rows,
    )


def _print_summary(run_id: str, results: list[RuleResult]) -> None:
    print(f"\nValidation run {run_id}: {len(results)} rule(s) evaluated")
    for r in results:
        marker = "PASS" if r.status == PASS else ("SKIP" if r.status == SKIP else "FAIL")
        print(f"  [{marker:<4}] [{r.severity:<8}] {r.rule_name}")
    blocking_fails = [r for r in results if r.severity == BLOCKING and r.status == FAIL]
    warning_fails = [r for r in results if r.severity == WARNING and r.status == FAIL]
    print(
        f"Summary: {len(results) - len(blocking_fails) - len(warning_fails)} ok, "
        f"{len(warning_fails)} warning fail(s), {len(blocking_fails)} blocking fail(s)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to cleansing_rules.yaml (default: docs/cleansing_rules.yaml)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    validation_cfg = cfg.get("validation") or {}
    if not validation_cfg:
        print(f"No 'validation:' section in {args.config}; nothing to do.")
        return 0

    run_id = os.getenv("AIRFLOW_CTX_DAG_RUN_ID") or str(uuid.uuid4())

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            ensure_results_table(cur)
            conn.commit()

            results = run_all_rules(cur, validation_cfg)

            persist_results(cur, run_id, results)
            conn.commit()
    finally:
        conn.close()

    _print_summary(run_id, results)

    exit_code = resolve_exit_code(results)
    if exit_code != 0:
        blocking_fails = [
            r for r in results if r.severity == BLOCKING and r.status == FAIL
        ]
        names = ", ".join(r.rule_name for r in blocking_fails)
        print(f"\nBlocking validation failure(s): {names}", file=sys.stderr)
        raise SystemExit(1)

    return 0


if __name__ == "__main__":
    main()
