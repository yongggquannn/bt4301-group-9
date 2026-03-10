"""
Airflow DAG: end-to-end DataOps orchestration.

Task chain:
    ingest_raw -> cleanse -> transform_features -> track_lineage -> trigger_eda

Notes:
    - `cleanse` and `trigger_eda` run project scripts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task


# source/dataops/airflow/dags/us8_dataops_e2e_dag.py -> repo root at parents[4]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[4]))
INGEST_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "load_raw_data.py"
CLEANSE_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "cleanse_data.py"
TRANSFORM_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "build_customer_features.py"
LINEAGE_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "generate_lineage.py"
EDA_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "run_eda.py"


def run_python_script(script_path: Path) -> None:
    """Run one project script and fail the task on non-zero exit code."""
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    subprocess.run(
        [sys.executable, "-B", str(script_path)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


@dag(
    dag_id="us8_dataops_e2e_pipeline",
    description="US-08 end-to-end DataOps orchestration DAG",
    default_args={
        "owner": "dataops",
        "depends_on_past": False,
        "retries": 1,
    },
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bt4301", "dataops", "us8"],
) 
def us8_dataops_e2e_pipeline():
    @task(task_id="ingest_raw")
    def ingest_raw():
        run_python_script(INGEST_SCRIPT)

    @task(task_id="cleanse")
    def cleanse():
        run_python_script(CLEANSE_SCRIPT)

    @task(task_id="transform_features")
    def transform_features():
        run_python_script(TRANSFORM_SCRIPT)

    @task(task_id="track_lineage")
    def track_lineage():
        run_python_script(LINEAGE_SCRIPT)

    @task(task_id="trigger_eda")
    def trigger_eda():
        run_python_script(EDA_SCRIPT)

    ingest_raw() >> cleanse() >> transform_features() >> track_lineage() >> trigger_eda()


dag = us8_dataops_e2e_pipeline()
