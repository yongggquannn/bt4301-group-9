from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure source/dataops is importable regardless of Airflow worker cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg2

from feature_registry import FEATURE_SPECS

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "kkbox"),
    "user": os.getenv("POSTGRES_USER", "bt4301"),
    "password": os.getenv("POSTGRES_PASSWORD", "bt4301pass"),
}

TRUNCATE_SQL = "TRUNCATE TABLE processed.data_lineage;"

INSERT_SQL = """
INSERT INTO processed.data_lineage (
    feature_name,
    source_table,
    transformation_rule,
    created_at
) VALUES (%s, %s, %s, %s);
"""


def lineage_rows() -> list[tuple[str, str, str, datetime]]:
    created_at = datetime.now(timezone.utc)
    return [
        (
            spec.feature_name,
            spec.source_table,
            spec.transformation_rule,
            created_at,
        )
        for spec in FEATURE_SPECS
    ]


def main() -> None:
    rows = lineage_rows()
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        cur.execute(TRUNCATE_SQL)
        cur.executemany(INSERT_SQL, rows)
        conn.commit()
        print(f"Wrote {len(rows):,} rows to processed.data_lineage")
    except Exception as e:
        conn.rollback()
        print(f"ERROR: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
