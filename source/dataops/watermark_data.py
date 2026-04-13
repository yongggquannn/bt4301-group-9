"""Watermark processed datasets with SHA-256 content hashes for audit trail.

Computes a deterministic hash over the processed.customer_features table
and records it in processed.data_watermarks for data provenance tracking.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# Allow imports from project root when run standalone or via Airflow.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(PROJECT_ROOT))

from source.common.db import get_connection


TABLE_NAME = "processed.customer_features"


def _compute_table_hash(cur) -> tuple[int, str]:
    """Return (row_count, sha256_hex) for the customer_features table.

    Hashes are computed over msno and is_churn columns ordered by msno
    so the result is deterministic regardless of physical row order.
    """
    cur.execute(
        "SELECT msno, is_churn FROM processed.customer_features ORDER BY msno"
    )
    rows = cur.fetchall()
    row_count = len(rows)

    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(f"{row[0]}|{row[1]}".encode())

    return row_count, hasher.hexdigest()


def main() -> None:
    pipeline_run_id = os.getenv("AIRFLOW_CTX_DAG_RUN_ID")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Ensure watermarks table exists (safe for first run before migrations).
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS processed.data_watermarks (
                    watermark_id    SERIAL      PRIMARY KEY,
                    table_name      TEXT        NOT NULL,
                    row_count       INTEGER    NOT NULL,
                    content_hash    TEXT        NOT NULL,
                    pipeline_run_id TEXT,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            conn.commit()

            row_count, content_hash = _compute_table_hash(cur)

            cur.execute(
                """
                INSERT INTO processed.data_watermarks
                    (table_name, row_count, content_hash, pipeline_run_id)
                VALUES (%s, %s, %s, %s)
                """,
                (TABLE_NAME, row_count, content_hash, pipeline_run_id),
            )
            conn.commit()

        print(f"Watermark recorded for {TABLE_NAME}:")
        print(f"  rows       = {row_count}")
        print(f"  hash       = {content_hash}")
        print(f"  run_id     = {pipeline_run_id}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
