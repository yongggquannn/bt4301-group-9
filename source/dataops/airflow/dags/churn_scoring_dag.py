"""
Airflow DAG: daily churn scoring orchestration.

Task chain:
    load_features -> load_production_model -> score -> write_predictions

Notes:
    - Each task calls score_churn.py with a step argument via subprocess.
    - Ensure Postgres is running and processed.customer_features is populated.
    - Optionally set:
        - PROJECT_ROOT
        - MLFLOW_MODEL_URI (for loading a production model from MLflow)
        - CHURN_HIGH_THRESHOLD, CHURN_MEDIUM_THRESHOLD
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task


# source/dataops/airflow/dags/churn_scoring_dag.py -> repo root at parents[4]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[4]))
SCORING_SCRIPT = PROJECT_ROOT / "source" / "mlops" / "score_churn.py"


def run_scoring_step(step: str) -> None:
    """Run one scoring step by calling score_churn.py with a step argument."""
    if not SCORING_SCRIPT.exists():
        raise FileNotFoundError(f"Scoring script not found: {SCORING_SCRIPT}")

    subprocess.run(
        [sys.executable, "-B", str(SCORING_SCRIPT), step],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


@dag(
    dag_id="daily_churn_scoring",
    description="Daily churn scoring: load_features -> load_production_model -> score -> write_predictions",
    default_args={
        "owner": "mlops",
        "depends_on_past": False,
        "retries": 1,
    },
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["bt4301", "mlops", "churn_scoring"],
)
def daily_churn_scoring():
    @task(task_id="load_features")
    def load_features():
        run_scoring_step("load_features")

    @task(task_id="load_production_model")
    def load_production_model():
        run_scoring_step("load_production_model")

    @task(task_id="score")
    def score():
        run_scoring_step("score")

    @task(task_id="write_predictions")
    def write_predictions():
        run_scoring_step("write_predictions")

    load_features() >> load_production_model() >> score() >> write_predictions()


dag = daily_churn_scoring()
