"""
Airflow DAG: automated retraining after monitoring drift alerts.

Task chain:
    check_drift_results -> retrain_if_needed -> evaluate -> register

Notes:
    - Reads the latest row from processed.model_monitoring_results.
    - Retrains only when max_psi > 0.2 or auc_delta > 0.05.
    - Uses champion-challenger registration logic for final promotion.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2
from airflow.operators.bash import BashOperator
from airflow.decorators import dag, task
from psycopg2.extras import RealDictCursor


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACT_DIR = PROJECT_ROOT / "docs" / "artifacts"
FEATURE_SELECTION_SCRIPT = PROJECT_ROOT / "source" / "mlops" / "feature_selection.py"
IMBALANCE_SCRIPT = PROJECT_ROOT / "source" / "mlops" / "train_us18_class_imbalance.py"
TRAIN_SCRIPT = PROJECT_ROOT / "source" / "mlops" / "train_model.py"
REGISTER_SCRIPT = PROJECT_ROOT / "source" / "mlops" / "register_model.py"

US10_BEST_MODEL_PATH = ARTIFACT_DIR / "us10_best_model.json"
US20_REGISTRY_PATH = ARTIFACT_DIR / "us20_champion_challenger_registry.json"
US21_EVALUATION_PATH = ARTIFACT_DIR / "us21_retraining_evaluation.json"
US21_DECISION_PATH = ARTIFACT_DIR / "us21_retraining_decision.json"

PSI_THRESHOLD = float(os.getenv("MONITORING_PSI_THRESHOLD", "0.2"))
AUC_DELTA_THRESHOLD = float(os.getenv("MONITORING_AUC_DELTA_THRESHOLD", "0.05"))
PROMOTION_THRESHOLD = float(os.getenv("CHAMPION_CHALLENGER_THRESHOLD", "0.0"))


from source.common.db import get_connection as get_pg_conn


@dag(
    dag_id="us21_automated_retraining",
    description="US-21 retraining DAG: check drift -> retrain -> evaluate -> register",
    default_args={
        "owner": "mlops",
        "depends_on_past": False,
        "retries": 0,
    },
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["bt4301", "mlops", "retraining", "us21"],
)
def us21_automated_retraining():
    @task.branch(task_id="check_drift_results")
    def check_drift_results() -> str:
        with get_pg_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        monitored_at,
                        baseline_auc,
                        current_auc,
                        auc_delta,
                        max_psi,
                        breached,
                        breached_reasons,
                        baseline_row_count,
                        current_row_count,
                        psi_by_feature
                    FROM processed.model_monitoring_results
                    ORDER BY monitored_at DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()

        if row is None:
            return "skip_retraining"

        max_psi = float(row["max_psi"] or 0.0)
        auc_delta = float(row["auc_delta"]) if row["auc_delta"] is not None else None
        should_retrain = max_psi > PSI_THRESHOLD or (
            auc_delta is not None and auc_delta > AUC_DELTA_THRESHOLD
        )

        payload = {
            "checked_at": datetime.utcnow().isoformat() + "Z",
            "latest_monitoring_row": {
                "monitored_at": row["monitored_at"].isoformat() if row["monitored_at"] else None,
                "baseline_auc": float(row["baseline_auc"]) if row["baseline_auc"] is not None else None,
                "current_auc": float(row["current_auc"]) if row["current_auc"] is not None else None,
                "auc_delta": auc_delta,
                "max_psi": max_psi,
                "breached": bool(row["breached"]),
                "breached_reasons": row["breached_reasons"],
                "baseline_row_count": int(row["baseline_row_count"] or 0),
                "current_row_count": int(row["current_row_count"] or 0),
                "psi_by_feature": row["psi_by_feature"],
            },
            "psi_threshold": PSI_THRESHOLD,
            "auc_delta_threshold": AUC_DELTA_THRESHOLD,
            "should_retrain": should_retrain,
        }
        US21_DECISION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return "retrain_if_needed" if should_retrain else "skip_retraining"

    retrain = BashOperator(
        task_id="retrain_if_needed",
        cwd=str(PROJECT_ROOT),
        env={
            "PROJECT_ROOT": str(PROJECT_ROOT),
            "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "127.0.0.1"),
            "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
            "POSTGRES_DB": os.getenv("POSTGRES_DB", "kkbox"),
            "POSTGRES_USER": os.getenv("POSTGRES_USER", "bt4301"),
            "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "bt4301pass"),
            "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"),
        },
        bash_command=(
            "python -B source/mlops/feature_selection.py && "
            "python -B source/mlops/train_us18_class_imbalance.py && "
            "python -B source/mlops/train_model.py"
        ),
    )

    @task(task_id="evaluate")
    def evaluate() -> dict[str, Any]:
        if not US10_BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{US10_BEST_MODEL_PATH} not found after retraining. "
                "Expected train_model.py to produce it."
            )

        best_payload = json.loads(US10_BEST_MODEL_PATH.read_text(encoding="utf-8"))
        retrain_meta = {
            "retrained_at": datetime.utcnow().isoformat() + "Z",
            "steps_run": [
                FEATURE_SELECTION_SCRIPT.name,
                IMBALANCE_SCRIPT.name,
                TRAIN_SCRIPT.name,
            ],
        }
        evaluation = {
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "retrain_meta": retrain_meta,
            "best_model": best_payload["best_model"],
            "run_id": best_payload["run_id"],
            "metrics": best_payload["metrics"],
            "imbalance_strategy": best_payload.get("imbalance_strategy"),
        }
        US21_EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        return evaluation

    @task(task_id="register")
    def register(evaluation: dict[str, Any]) -> None:
        os.environ["MLFLOW_TRACKING_URI"] = os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5001",
        )
        if not REGISTER_SCRIPT.exists():
            raise FileNotFoundError(f"Script not found: {REGISTER_SCRIPT}")
        result = subprocess.run(
            [sys.executable, "-B", str(REGISTER_SCRIPT),
             "--promotion-threshold", str(PROMOTION_THRESHOLD)],
            capture_output=True, text=True, check=False,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"register_model.py failed (exit {result.returncode}):\n{result.stderr}"
            )

        registry_payload = {}
        if US20_REGISTRY_PATH.exists():
            registry_payload = json.loads(US20_REGISTRY_PATH.read_text(encoding="utf-8"))

        decision_payload = {
            "registered_at": datetime.utcnow().isoformat() + "Z",
            "promotion_threshold": PROMOTION_THRESHOLD,
            "evaluation": evaluation,
            "registry_decision": registry_payload,
        }
        US21_DECISION_PATH.write_text(json.dumps(decision_payload, indent=2), encoding="utf-8")

    @task(task_id="skip_retraining")
    def skip_retraining() -> None:
        payload = {
            "checked_at": datetime.utcnow().isoformat() + "Z",
            "should_retrain": False,
            "reason": "Latest monitoring result did not exceed PSI/AUC delta thresholds.",
            "psi_threshold": PSI_THRESHOLD,
            "auc_delta_threshold": AUC_DELTA_THRESHOLD,
        }
        US21_DECISION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("Retraining skipped because latest monitoring result did not breach thresholds.")

    branch = check_drift_results()
    evaluated = evaluate()
    registered = register(evaluated)
    skipped = skip_retraining()

    branch >> [retrain, skipped]
    retrain >> evaluated >> registered


dag = us21_automated_retraining()
