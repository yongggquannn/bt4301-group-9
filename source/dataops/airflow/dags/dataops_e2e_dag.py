"""
Airflow DAG: end-to-end DataOps orchestration.

Task chain:
    ingest_raw -> cleanse -> transform_features -> track_lineage -> trigger_eda -> generate_eda_images_report

Notes:
    - `cleanse`, `trigger_eda`, and `generate_eda_images_report` run project scripts.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

from airflow.decorators import dag, task

# source/dataops/airflow/dags/dataops_e2e_dag.py -> repo root at parents[4]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(PROJECT_ROOT))

from source.common.dag_utils import run_python_script as _run_script

INGEST_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "load_raw_data.py"
CLEANSE_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "cleanse_data.py"
TRANSFORM_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "build_customer_features.py"
LINEAGE_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "generate_lineage.py"
EDA_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "run_eda.py"
EDA_IMAGES_REPORT_SCRIPT = PROJECT_ROOT / "source" / "dataops" / "generate_eda_images_report.py"

run_python_script = partial(_run_script, cwd=PROJECT_ROOT)


@dag(
    dag_id="dataops_e2e_pipeline",
    description="End-to-end DataOps orchestration DAG",
    default_args={
        "owner": "dataops",
        "depends_on_past": False,
        "retries": 1,
    },
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bt4301", "dataops"],
)
def dataops_e2e_pipeline():
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

    @task(task_id="generate_eda_images_report")
    def generate_eda_images_report():
        run_python_script(EDA_IMAGES_REPORT_SCRIPT)

    ingest_raw() >> cleanse() >> transform_features() >> track_lineage() >> trigger_eda() >> generate_eda_images_report()


dag = dataops_e2e_pipeline()
