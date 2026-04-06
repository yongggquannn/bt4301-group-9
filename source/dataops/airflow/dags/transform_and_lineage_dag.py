"""
Airflow DAG: transform_features -> track_lineage

Purpose:
    US-06 orchestration for feature build and lineage tracking.
    Scope is intentionally limited to two tasks for this user story.

How to use:
    1. Ensure Postgres is running and raw.* tables are already loaded.
    2. Ensure Airflow worker/scheduler uses the same Python env where project deps are installed.
    3. Set PROJECT_ROOT if your Airflow DAGs folder is outside this repository.
    4. Trigger DAG manually in Airflow UI.

Environment variables used by downstream scripts:
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

from airflow.decorators import dag, task

# If PROJECT_ROOT is not provided, resolve from this file location:
# source/dataops/airflow/dags/transform_and_lineage_dag.py -> repo root at parents[4]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(PROJECT_ROOT))

from source.common.dag_utils import run_python_script as _run_script

TRANSFORM_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "build_customer_features.py"
LINEAGE_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "generate_lineage.py"

run_python_script = partial(_run_script, cwd=PROJECT_ROOT)


@dag(
    dag_id="us6_transform_and_track_lineage",
    description="US-06: build processed.customer_features and populate processed.data_lineage",
    default_args={
        "owner": "dataops",
        "depends_on_past": False,
        "retries": 1,
    },
    start_date=datetime(2026, 1, 1),
    schedule=None,  # Manual trigger for sprint/demo use.
    catchup=False,
    tags=["bt4301", "dataops", "us6"],
) 
def us6_transform_and_track_lineage():
    @task(task_id="transform_features")
    def transform_features():
        run_python_script(TRANSFORM_SCRIPT)

    @task(task_id="track_lineage")
    def track_lineage():
        run_python_script(LINEAGE_SCRIPT)

    transform_features() >> track_lineage()


dag = us6_transform_and_track_lineage()
