"""Shared database configuration for all pipeline scripts."""

from __future__ import annotations

import os


def get_db_config() -> dict:
    """Return a psycopg2-compatible connection-parameter dict."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", 5432)),
        "dbname": os.getenv("POSTGRES_DB", "kkbox"),
        "user": os.getenv("POSTGRES_USER", "bt4301"),
        "password": os.getenv("POSTGRES_PASSWORD", "bt4301pass"),
    }


def get_connection():
    """Open and return a new psycopg2 connection."""
    import psycopg2

    return psycopg2.connect(**get_db_config())


def get_current_snapshot_id(conn=None) -> str | None:
    """Return the UUID of the most recent completed feature snapshot, or None."""
    import psycopg2

    close_conn = False
    if conn is None:
        conn = psycopg2.connect(**get_db_config())
        close_conn = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT snapshot_id::text
                FROM processed.feature_snapshots
                WHERE status = 'completed'
                ORDER BY build_completed_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        if close_conn:
            conn.close()
