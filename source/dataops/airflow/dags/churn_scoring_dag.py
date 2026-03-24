"""
Airflow DAG: load_features -> load_production_model -> score -> write_predictions

Purpose:
    US: Churn scoring orchestration. Runs daily and writes churn probabilities
    to processed.churn_predictions.

How to use:
    1. Ensure Postgres is running and processed.customer_features is populated.
    2. Ensure Airflow worker/scheduler uses the same Python env where project deps are installed.
    3. Optionally set:
         - PROJECT_ROOT
         - MLFLOW_MODEL_URI (for loading a production model from MLflow)
         - CHURN_HIGH_THRESHOLD, CHURN_MEDIUM_THRESHOLD
    4. Trigger DAG manually, then switch schedule to daily for production.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


# source/dataops/airflow/dags/churn_scoring_dag.py -> repo root at parents[4]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[4]))
SCORING_SCRIPT = PROJECT_ROOT / "source" / "mlops" / "score_churn.py"


def run_scoring_step(step: str) -> None:
    """Run one scoring step by calling score_churn.py with a step argument."""
    if not SCORING_SCRIPT.exists():
        raise FileNotFoundError(f"Scoring script not found: {SCORING_SCRIPT}")

    cmd = [sys.executable, "-B", str(SCORING_SCRIPT), step]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="daily_churn_scoring",
    description="Daily churn scoring: load_features -> load_production_model -> score -> write_predictions",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["bt4301", "mlops", "churn_scoring"],
) as dag:
    load_features = PythonOperator(
        task_id="load_features",
        python_callable=run_scoring_step,
        op_kwargs={"step": "load_features"},
    )

    load_production_model = PythonOperator(
        task_id="load_production_model",
        python_callable=run_scoring_step,
        op_kwargs={"step": "load_production_model"},
    )

    score = PythonOperator(
        task_id="score",
        python_callable=run_scoring_step,
        op_kwargs={"step": "score"},
    )

    write_predictions = PythonOperator(
        task_id="write_predictions",
        python_callable=run_scoring_step,
        op_kwargs={"step": "write_predictions"},
    )

    load_features >> load_production_model >> score >> write_predictions

