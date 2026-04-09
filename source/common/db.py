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
