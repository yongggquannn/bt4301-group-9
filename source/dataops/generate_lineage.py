from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure source/dataops is importable regardless of Airflow worker cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import psycopg2

from feature_registry import FEATURE_SPECS
from source.common.db import get_db_config

DB_CONFIG = get_db_config()

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
    print("=" * 60)
    print("Generating processed.data_lineage")
    print("=" * 60)

    rows = lineage_rows()
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("\n[1/2] Truncating existing lineage table...")
        cur.execute(TRUNCATE_SQL)
        print("[2/2] Inserting refreshed lineage rows...")
        cur.executemany(INSERT_SQL, rows)
        conn.commit()
        print(f"  Wrote {len(rows):,} rows to processed.data_lineage")
    except Exception as e:
        conn.rollback()
        print(f"\n  ERROR: {e}")
        raise
    finally:
        cur.close()
        conn.close()
        
    print("\n" + "=" * 60)
    print("Done. Lineage generated. Run run_eda.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
