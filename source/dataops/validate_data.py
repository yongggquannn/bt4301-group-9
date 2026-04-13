"""Data quality validation for processed.customer_features.

Runs a suite of checks and raises an exception if any critical check fails,
which stops the Airflow DAG to prevent bad data from propagating downstream.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(PROJECT_ROOT))

from source.common.db import get_connection


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed


def main() -> None:
    conn = get_connection()
    failures: list[str] = []

    try:
        with conn.cursor() as cur:
            # 1. Row count > 0
            cur.execute("SELECT COUNT(*) FROM processed.customer_features")
            feature_count = cur.fetchone()[0]
            if not _check("Row count > 0", feature_count > 0, f"{feature_count} rows"):
                failures.append(f"customer_features is empty ({feature_count} rows)")

            # 2. Row count matches raw.train
            cur.execute("SELECT COUNT(*) FROM raw.train")
            train_count = cur.fetchone()[0]
            matched = feature_count == train_count
            if not _check(
                "Row count matches raw.train",
                matched,
                f"features={feature_count}, train={train_count}",
            ):
                failures.append(
                    f"Row count mismatch: features={feature_count} vs train={train_count}"
                )

            # 3. No NULL msno
            cur.execute(
                "SELECT COUNT(*) FROM processed.customer_features WHERE msno IS NULL"
            )
            null_msno = cur.fetchone()[0]
            if not _check("No NULL msno", null_msno == 0, f"{null_msno} nulls"):
                failures.append(f"{null_msno} NULL msno values found")

            # 4. No NULL is_churn
            cur.execute(
                "SELECT COUNT(*) FROM processed.customer_features WHERE is_churn IS NULL"
            )
            null_churn = cur.fetchone()[0]
            if not _check("No NULL is_churn", null_churn == 0, f"{null_churn} nulls"):
                failures.append(f"{null_churn} NULL is_churn values found")

            # 5. is_churn only 0 or 1
            cur.execute(
                "SELECT COUNT(*) FROM processed.customer_features "
                "WHERE is_churn NOT IN (0, 1)"
            )
            bad_churn = cur.fetchone()[0]
            if not _check(
                "is_churn values in {0, 1}", bad_churn == 0, f"{bad_churn} invalid"
            ):
                failures.append(f"{bad_churn} is_churn values outside {{0, 1}}")

            # 6. No duplicate msno
            cur.execute(
                "SELECT COUNT(*) FROM ("
                "  SELECT msno FROM processed.customer_features "
                "  GROUP BY msno HAVING COUNT(*) > 1"
                ") dupes"
            )
            dup_count = cur.fetchone()[0]
            if not _check("No duplicate msno", dup_count == 0, f"{dup_count} duplicates"):
                failures.append(f"{dup_count} duplicate msno entries")

            # 7. Age (bd) within reasonable range (allow NULLs)
            cur.execute(
                "SELECT COUNT(*) FROM processed.customer_features "
                "WHERE bd IS NOT NULL AND (bd < 0 OR bd > 100)"
            )
            bad_age = cur.fetchone()[0]
            if not _check(
                "Age (bd) in [0, 100]", bad_age == 0, f"{bad_age} out of range"
            ):
                failures.append(f"{bad_age} age values outside [0, 100]")

            # 8. total_amount_paid non-negative (allow NULLs)
            cur.execute(
                "SELECT COUNT(*) FROM processed.customer_features "
                "WHERE total_amount_paid IS NOT NULL AND total_amount_paid < 0"
            )
            neg_paid = cur.fetchone()[0]
            if not _check(
                "total_amount_paid >= 0", neg_paid == 0, f"{neg_paid} negative"
            ):
                failures.append(f"{neg_paid} negative total_amount_paid values")

    finally:
        conn.close()

    print()
    if failures:
        summary = "; ".join(failures)
        raise ValueError(f"Data validation failed: {summary}")

    print("All data quality checks passed.")


if __name__ == "__main__":
    main()
