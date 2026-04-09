"""
Loads raw KKBox CSV files into the raw.* PostgreSQL tables using COPY.

Run this AFTER `docker compose up -d` and BEFORE `cleanse_data.py`.

Usage:
    python source/dataops/load_raw_data.py

Environment variables (matches docker-compose.yml / .env):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import psycopg2

from source.common.db import get_db_config

DB_CONFIG = get_db_config()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# Map: (csv filename, target table, csv columns in order)
# ingestion_timestamp is excluded — it has a DEFAULT NOW() in the schema.
TABLES = [
    (
        "train_v2.csv",
        "raw.train",
        ["msno", "is_churn"],
    ),
    (
        "members_v3.csv",
        "raw.members",
        ["msno", "city", "bd", "gender", "registered_via", "registration_init_time"],
    ),
    (
        "transactions_v2.csv",
        "raw.transactions",
        [
            "msno", "payment_method_id", "payment_plan_days", "plan_list_price",
            "actual_amount_paid", "is_auto_renew", "transaction_date",
            "membership_expire_date", "is_cancel",
        ],
    ),
    (
        "user_logs_v2.csv",
        "raw.user_logs",
        ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"],
    ),
]


def load_table(cur, csv_file, table, columns):
    path = os.path.join(DATA_DIR, csv_file)
    col_list = ", ".join(columns)
    print(f"\n[{table}]")
    print(f"  Source: {csv_file}  ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")

    cur.execute(f"TRUNCATE TABLE {table};")

    copy_sql = f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')"
    with open(path, "r") as f:
        cur.copy_expert(copy_sql, f)

    cur.execute(f"SELECT COUNT(*) FROM {table};")
    count = cur.fetchone()[0]
    print(f"  Loaded: {count:,} rows")


def main():
    print("=" * 60)
    print("Loading raw KKBox CSVs into PostgreSQL")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        for csv_file, table, columns in TABLES:
            load_table(cur, csv_file, table, columns)
        conn.commit()
        print("\n  All tables loaded successfully.")
    except Exception as e:
        conn.rollback()
        print(f"\n  ERROR: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    print("\n" + "=" * 60)
    print("Done. Run cleanse_data.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
