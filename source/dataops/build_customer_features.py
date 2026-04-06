"""
US-4: Builds processed.customer_features from the four raw KKBox tables.

Run this script AFTER:
  1. Docker Postgres container is up   (docker compose up -d)
  2. Raw CSV data has been loaded into raw.*
  3. cleanse_data.py has been run to verify the raw tables are clean and ready.

Usage:
    python source/dataops/build_customer_features.py

Environment variables (matches docker-compose.yml / .env):
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import psycopg2

from source.common.db import get_db_config

DB_CONFIG = get_db_config()

DDL = """
CREATE TABLE IF NOT EXISTS processed.customer_features (
    msno                          TEXT          NOT NULL,
    is_churn                      SMALLINT      NOT NULL,
    city                          SMALLINT,
    bd                            SMALLINT,
    gender                        VARCHAR(10),
    registered_via                SMALLINT,
    registration_init_time        INT,
    transaction_count             INT,
    renewal_count                 INT,
    cancel_count                  INT,
    total_amount_paid             NUMERIC,
    avg_plan_days                 NUMERIC(8,2),
    latest_payment_method_id      SMALLINT,
    latest_is_auto_renew          SMALLINT,
    latest_membership_expire_date INT,
    num_active_days               INT,
    total_secs                    NUMERIC,
    avg_total_secs                NUMERIC(12,4),
    total_num_songs               INT,
    avg_num_songs                 NUMERIC(10,4),
    total_num_unq                 INT,
    avg_num_unq                   NUMERIC(10,4),
    feature_created_at            TIMESTAMPTZ   DEFAULT NOW(),
    PRIMARY KEY (msno)
);
"""

TRUNCATE_SQL = "TRUNCATE TABLE processed.customer_features;"

INSERT_SQL = """
INSERT INTO processed.customer_features (
    msno, is_churn,
    city, bd, gender, registered_via, registration_init_time,
    transaction_count, renewal_count, cancel_count,
    total_amount_paid, avg_plan_days,
    latest_payment_method_id, latest_is_auto_renew,
    latest_membership_expire_date,
    num_active_days, total_secs, avg_total_secs,
    total_num_songs, avg_num_songs, total_num_unq, avg_num_unq
)
WITH txn_ranked AS (
    SELECT
        msno,
        payment_method_id,
        payment_plan_days,
        actual_amount_paid,
        is_auto_renew,
        membership_expire_date,
        is_cancel,
        ROW_NUMBER() OVER (
            PARTITION BY msno
            ORDER BY transaction_date DESC
        ) AS rn
    FROM raw.transactions
),
txn_agg AS (
    SELECT
        msno,
        COUNT(*)                                               AS transaction_count,
        SUM(CASE WHEN is_cancel = 0 THEN 1 ELSE 0 END)        AS renewal_count,
        SUM(is_cancel)                                         AS cancel_count,
        SUM(actual_amount_paid)                                AS total_amount_paid,
        ROUND(AVG(payment_plan_days::NUMERIC), 2)              AS avg_plan_days,
        MAX(membership_expire_date)                            AS latest_membership_expire_date,
        MAX(CASE WHEN rn = 1 THEN payment_method_id END)       AS latest_payment_method_id,
        MAX(CASE WHEN rn = 1 THEN is_auto_renew END)           AS latest_is_auto_renew
    FROM txn_ranked
    GROUP BY msno
),
log_agg AS (
    SELECT
        msno,
        COUNT(DISTINCT date)                                              AS num_active_days,
        SUM(total_secs)                                                   AS total_secs,
        ROUND(AVG(total_secs), 4)                                         AS avg_total_secs,
        SUM(num_25 + num_50 + num_75 + num_985 + num_100)                 AS total_num_songs,
        ROUND(AVG((num_25 + num_50 + num_75 + num_985 + num_100)::NUMERIC), 4) AS avg_num_songs,
        SUM(num_unq)                                                      AS total_num_unq,
        ROUND(AVG(num_unq::NUMERIC), 4)                                   AS avg_num_unq
    FROM raw.user_logs
    GROUP BY msno
)
SELECT
    t.msno,
    t.is_churn,
    m.city,
    m.bd,
    m.gender,
    m.registered_via,
    m.registration_init_time,
    ta.transaction_count,
    ta.renewal_count,
    ta.cancel_count,
    ta.total_amount_paid,
    ta.avg_plan_days,
    ta.latest_payment_method_id,
    ta.latest_is_auto_renew,
    ta.latest_membership_expire_date,
    la.num_active_days,
    la.total_secs,
    la.avg_total_secs,
    la.total_num_songs,
    la.avg_num_songs,
    la.total_num_unq,
    la.avg_num_unq
FROM      raw.train        AS t
LEFT JOIN raw.members      AS m  ON t.msno = m.msno
LEFT JOIN txn_agg          AS ta ON t.msno = ta.msno
LEFT JOIN log_agg          AS la ON t.msno = la.msno;
"""


def validate(cur):
    checks_passed = True

    cur.execute("SELECT COUNT(*) FROM raw.train;")
    raw_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM processed.customer_features;")
    feat_count = cur.fetchone()[0]
    row_ok = raw_count == feat_count
    checks_passed = checks_passed and row_ok
    print(f"  raw.train rows        : {raw_count:,}")
    print(f"  customer_features rows: {feat_count:,}  {'PASS' if row_ok else 'FAIL'}")

    cur.execute("SELECT COUNT(*) FROM processed.customer_features WHERE msno IS NULL;")
    null_msno = cur.fetchone()[0]
    checks_passed = checks_passed and (null_msno == 0)
    print(f"  NULL msno             : {null_msno}  {'PASS' if null_msno == 0 else 'FAIL'}")

    cur.execute("SELECT COUNT(*) FROM processed.customer_features WHERE is_churn IS NULL;")
    null_churn = cur.fetchone()[0]
    checks_passed = checks_passed and (null_churn == 0)
    print(f"  NULL is_churn         : {null_churn}  {'PASS' if null_churn == 0 else 'FAIL'}")

    cur.execute("""
        SELECT
            ROUND(100.0 * SUM(CASE WHEN city IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1)
        FROM processed.customer_features;
    """)
    pct_demo = cur.fetchone()[0]
    print(f"  Demographics coverage : {pct_demo}%  (expected ~88%)")

    cur.execute("""
        SELECT
            ROUND(100.0 * SUM(CASE WHEN transaction_count IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1)
        FROM processed.customer_features;
    """)
    pct_txn = cur.fetchone()[0]
    print(f"  Transaction coverage  : {pct_txn}%")

    cur.execute("""
        SELECT
            ROUND(100.0 * SUM(CASE WHEN num_active_days IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1)
        FROM processed.customer_features;
    """)
    pct_log = cur.fetchone()[0]
    print(f"  User log coverage     : {pct_log}%")

    cur.execute("""
        SELECT
            ROUND(100.0 * AVG(is_churn::NUMERIC), 2) FROM raw.train;
    """)
    raw_churn_rate = cur.fetchone()[0]
    cur.execute("""
        SELECT
            ROUND(100.0 * AVG(is_churn::NUMERIC), 2) FROM processed.customer_features;
    """)
    feat_churn_rate = cur.fetchone()[0]
    print(f"  Churn rate (raw)      : {raw_churn_rate}%")
    print(f"  Churn rate (features) : {feat_churn_rate}%")

    return checks_passed


def main():
    print("=" * 60)
    print("US-4: Building processed.customer_features")
    print("=" * 60)

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("\n[1/3] Ensuring table DDL is applied...")
        cur.execute(DDL)
        conn.commit()
        print("  Table structure verified.")

        print("\n[2/3] Running transformation (TRUNCATE + INSERT)...")
        t0 = time.time()
        cur.execute(TRUNCATE_SQL)
        cur.execute(INSERT_SQL)
        conn.commit()
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        print("\n[3/3] Validation checks...")
        ok = validate(cur)

        if not ok:
            raise AssertionError("One or more critical validation checks FAILED. See above.")
        print("\n  All critical checks PASSED.")

    except Exception as e:
        conn.rollback()
        print(f"\n  ERROR: {e}")
        raise
    finally:
        cur.close()
        conn.close()

    print("\n" + "=" * 60)
    print("Done. processed.customer_features is ready. Run generate_lineage.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
