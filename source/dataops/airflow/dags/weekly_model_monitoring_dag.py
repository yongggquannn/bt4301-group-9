"""
Airflow DAG: weekly model monitoring for drift and degradation.

Task chain:
    compute_and_log_monitoring_metrics -> evaluate_thresholds
        -> log_monitoring_alert -> trigger_retraining
        -> no_alert_needed

Notes:
    - Computes PSI for top 5 features.
    - Compares current AUC vs baseline and logs delta to PostgreSQL.
    - Automatically triggers the US-21 retraining DAG when PSI > 0.2 or
      AUC delta > 0.05.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from psycopg2.extras import RealDictCursor


DEFAULT_TOP_5_FEATURES = [
    "transaction_count",
    "total_amount_paid",
    "avg_plan_days",
    "num_active_days",
    "avg_total_secs",
]

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(PROJECT_ROOT))
PERM_IMPORTANCE_PATH = PROJECT_ROOT / "docs" / "artifacts" / "permutation_importance.csv"
BEST_MODEL_PATH = PROJECT_ROOT / "docs" / "artifacts" / "best_model.json"
REGISTRY_EVIDENCE_PATH = PROJECT_ROOT / "docs" / "artifacts" / "model_registry.json"


from source.common.db import get_connection as get_pg_conn
from source.common.monitoring_utils import compute_psi, roc_auc_binary
from source.mlops.monitoring import (
    MAX_CALIBRATION_ERROR_THRESHOLD,
    MIN_LABELED_SAMPLES,
    SEGMENT_PSI_THRESHOLD,
    compute_brier_score,
    compute_calibration_bins,
    compute_max_calibration_error,
    compute_score_distribution,
    segment_psi_breached,
)

# Monitoring DAG segments — derived at query time from customer_features columns.
SEGMENT_COLUMNS = ("latest_is_auto_renew", "has_cancel_history")


def stable_bucket(key: str) -> int:
    # Stable hash bucket for cohort fallback split.
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100


def resolve_top_monitor_features(conn) -> list[str]:
    if not PERM_IMPORTANCE_PATH.exists():
        return DEFAULT_TOP_5_FEATURES

    importance_df = pd.read_csv(PERM_IMPORTANCE_PATH)
    if "feature" not in importance_df.columns:
        return DEFAULT_TOP_5_FEATURES

    if "perm_importance_mean" in importance_df.columns:
        importance_df = importance_df.sort_values(
            "perm_importance_mean",
            ascending=False,
            kind="mergesort",
        )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'processed'
              AND table_name = 'customer_features'
            """
        )
        feature_table_cols = {row[0] for row in cur.fetchall()}

    excluded = {"msno", "is_churn", "feature_created_at"}
    candidates = []
    for feature in importance_df["feature"].tolist():
        if feature in excluded:
            continue
        if feature not in feature_table_cols:
            continue
        candidates.append(feature)

    top_5 = candidates[:5]
    return top_5 if len(top_5) == 5 else DEFAULT_TOP_5_FEATURES


def load_baseline_auc_reference() -> tuple[float | None, str]:
    for path, source_name in [
        (REGISTRY_EVIDENCE_PATH, "mlflow_registry_production"),
        (BEST_MODEL_PATH, "best_model_artifact"),
    ]:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            if "roc_auc" in metrics:
                return float(metrics["roc_auc"]), source_name
            if "roc_auc" in payload:
                return float(payload["roc_auc"]), source_name
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None, "historical_baseline_window"


@dag(
    dag_id="weekly_model_monitoring",
    description="Weekly monitoring DAG: PSI + AUC degradation checks with automatic retraining trigger",
    default_args={
        "owner": "mlops",
        "depends_on_past": False,
        "retries": 1,
    },
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["bt4301", "mlops", "monitoring"],
)
def weekly_model_monitoring():
    @task(task_id="compute_and_log_monitoring_metrics")
    def compute_and_log_monitoring_metrics() -> dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        current_start = now_utc - timedelta(days=7)
        baseline_start = now_utc - timedelta(days=90)
        min_rows = 50

        _SAFE_COLUMN_RE = re.compile(r"^[a-zA-Z0-9_]+$")

        with get_pg_conn() as conn:
            top_5_features = resolve_top_monitor_features(conn)
            for feat in top_5_features:
                if not _SAFE_COLUMN_RE.match(feat):
                    raise ValueError(f"Unsafe feature name rejected: {feat!r}")
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        EXISTS (
                            SELECT 1
                            FROM information_schema.tables
                            WHERE table_schema = 'processed'
                              AND table_name = 'churn_predictions'
                        ) AS has_processed_preds,
                        EXISTS (
                            SELECT 1
                            FROM information_schema.tables
                            WHERE table_schema = 'predictions'
                              AND table_name = 'churn_predictions'
                        ) AS has_predictions_preds
                    """
                )
                table_flags = cur.fetchone()
                if table_flags["has_processed_preds"]:
                    predictions_table = "processed.churn_predictions"
                elif table_flags["has_predictions_preds"]:
                    predictions_table = "predictions.churn_predictions"
                else:
                    raise ValueError(
                        "No predictions table found. Expected processed.churn_predictions "
                        "or predictions.churn_predictions."
                    )

                cur.execute(
                    f"""
                    SELECT
                        p.customer_id AS msno,
                        p.scored_at,
                        p.churn_probability::float8 AS score,
                        t.is_churn::int AS label,
                        COALESCE(cf.latest_is_auto_renew, 0)::int AS latest_is_auto_renew,
                        CASE WHEN COALESCE(cf.cancel_count, 0) > 0 THEN 1 ELSE 0 END
                            AS has_cancel_history,
                        {", ".join([f"cf.{f}::float8 AS {f}" for f in top_5_features])}
                    FROM {predictions_table} p
                    JOIN processed.customer_features cf
                      ON cf.msno = p.customer_id
                    JOIN raw.train t
                      ON t.msno = p.customer_id
                    WHERE p.churn_probability IS NOT NULL
                    """
                )
                rows = cur.fetchall()

            if not rows:
                raise ValueError(
                    "No joined rows found. Ensure predictions.churn_predictions and raw.train are populated."
                )

            msno = np.array([r["msno"] for r in rows], dtype=object)
            scored_at = np.array([r["scored_at"] for r in rows], dtype=object)
            y_true = np.array([r["label"] for r in rows], dtype=np.int64)
            y_score = np.array([r["score"] for r in rows], dtype=np.float64)

            time_current_mask = np.array(
                [ts is not None and ts >= current_start for ts in scored_at], dtype=bool
            )
            time_baseline_mask = np.array(
                [ts is not None and baseline_start <= ts < current_start for ts in scored_at], dtype=bool
            )

            use_hash_fallback = (
                np.sum(time_current_mask) < min_rows or np.sum(time_baseline_mask) < min_rows
            )
            if use_hash_fallback:
                buckets = np.array([stable_bucket(str(k)) for k in msno], dtype=np.int64)
                baseline_mask = buckets < 70
                current_mask = buckets >= 70
                cohort_strategy = "hash_fallback"
                baseline_window_start = baseline_start
                baseline_window_end = current_start
                current_window_start = current_start
                current_window_end = now_utc
            else:
                baseline_mask = time_baseline_mask
                current_mask = time_current_mask
                cohort_strategy = "time_window"
                baseline_window_start = baseline_start
                baseline_window_end = current_start
                current_window_start = current_start
                current_window_end = now_utc

            baseline_n = int(np.sum(baseline_mask))
            current_n = int(np.sum(current_mask))
            baseline_pos = int(np.sum(y_true[baseline_mask] == 1))
            baseline_neg = int(np.sum(y_true[baseline_mask] == 0))
            current_pos = int(np.sum(y_true[current_mask] == 1))
            current_neg = int(np.sum(y_true[current_mask] == 0))

            baseline_auc, baseline_auc_source = load_baseline_auc_reference()
            current_auc = None
            auc_delta = None

            psi_by_feature: dict[str, float] = {}
            for feat in top_5_features:
                values = np.array([r[feat] for r in rows], dtype=np.float64)
                psi_by_feature[feat] = compute_psi(values[baseline_mask], values[current_mask])

            finite_psi = [v for v in psi_by_feature.values() if np.isfinite(v)]
            max_psi = float(max(finite_psi) if finite_psi else 0.0)

            breached_reasons = []
            if baseline_auc is None and (baseline_pos == 0 or baseline_neg == 0):
                breached_reasons.append(
                    "Insufficient baseline class coverage for AUC and no production baseline artifact found."
                )
            if baseline_auc is None and baseline_pos > 0 and baseline_neg > 0:
                baseline_auc = roc_auc_binary(y_true[baseline_mask], y_score[baseline_mask])
                baseline_auc_source = "historical_baseline_window"
            if current_pos == 0 or current_neg == 0:
                breached_reasons.append(
                    "Insufficient current class coverage for AUC (requires both churn/non-churn)."
                )

            if baseline_auc is not None and current_pos > 0 and current_neg > 0:
                current_auc = roc_auc_binary(y_true[current_mask], y_score[current_mask])
                auc_delta = baseline_auc - current_auc

            if max_psi > 0.2:
                breached_reasons.append(f"PSI threshold breached: max_psi={max_psi:.4f} (>0.2)")
            if auc_delta is not None and auc_delta > 0.05:
                breached_reasons.append(
                    f"AUC degradation threshold breached: delta={auc_delta:.4f} (>0.05)"
                )

            # --- Workstream F: calibration + segment PSI + sample-size guard ---
            current_y_true = y_true[current_mask]
            current_y_score = y_score[current_mask]
            calibration_bins: list[dict] = []
            brier_score: float | None = None
            max_cal_err: float | None = None
            score_distribution: list[dict] = []
            segment_psi: dict[str, float] = {}

            labeled_sample_count = int(current_y_true.size)
            insufficient = labeled_sample_count < MIN_LABELED_SAMPLES

            if labeled_sample_count > 0:
                try:
                    calibration_bins = list(
                        compute_calibration_bins(current_y_true, current_y_score, n_bins=10)
                    )
                    brier_score = compute_brier_score(current_y_true, current_y_score)
                    max_cal_err = compute_max_calibration_error(calibration_bins)
                    score_distribution = list(compute_score_distribution(current_y_score))
                except ValueError as exc:
                    breached_reasons.append(f"Calibration computation failed: {exc}")

                # Per-segment PSI — compare current vs baseline for each segment key.
                segment_df_baseline = pd.DataFrame(
                    {
                        "score": y_score[baseline_mask],
                        **{
                            col: np.array([r[col] for r in rows])[baseline_mask]
                            for col in SEGMENT_COLUMNS
                        },
                    }
                )
                segment_df_current = pd.DataFrame(
                    {
                        "score": current_y_score,
                        **{
                            col: np.array([r[col] for r in rows])[current_mask]
                            for col in SEGMENT_COLUMNS
                        },
                    }
                )
                try:
                    from source.mlops.monitoring import compute_segment_psi

                    segment_psi = compute_segment_psi(
                        segment_df_baseline,
                        segment_df_current,
                        segment_columns=list(SEGMENT_COLUMNS),
                        score_column="score",
                    )
                except ValueError as exc:
                    breached_reasons.append(f"Segment PSI failed: {exc}")

            if insufficient:
                status = "insufficient_data"
                # Don't flip breached just because we saw a single PSI number;
                # we don't have enough labels to trust the verdict yet.
                breached_reasons.append(
                    f"Insufficient labelled samples in current window: "
                    f"{labeled_sample_count} < {MIN_LABELED_SAMPLES}"
                )
                breached = False
            else:
                if max_cal_err is not None and max_cal_err > MAX_CALIBRATION_ERROR_THRESHOLD:
                    breached_reasons.append(
                        f"Calibration error exceeds threshold: "
                        f"{max_cal_err:.4f} > {MAX_CALIBRATION_ERROR_THRESHOLD}"
                    )
                bad_segments = segment_psi_breached(segment_psi, SEGMENT_PSI_THRESHOLD)
                if bad_segments:
                    breached_reasons.append(
                        f"Segment PSI breached ({SEGMENT_PSI_THRESHOLD}): "
                        f"{', '.join(bad_segments)}"
                    )
                breached = len(breached_reasons) > 0
                status = "breached" if breached else "ok"

            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS processed.model_monitoring_results (
                        result_id               BIGSERIAL     PRIMARY KEY,
                        monitored_at            TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
                        baseline_window_start   TIMESTAMPTZ   NOT NULL,
                        baseline_window_end     TIMESTAMPTZ   NOT NULL,
                        current_window_start    TIMESTAMPTZ   NOT NULL,
                        current_window_end      TIMESTAMPTZ   NOT NULL,
                        baseline_auc            NUMERIC(8,6),
                        current_auc             NUMERIC(8,6),
                        auc_delta               NUMERIC(8,6),
                        max_psi                 NUMERIC(10,6) NOT NULL,
                        breached                BOOLEAN       NOT NULL DEFAULT FALSE,
                        breached_reasons        TEXT,
                        psi_by_feature          JSONB         NOT NULL,
                        baseline_auc_source     TEXT          NOT NULL DEFAULT 'historical_baseline_window',
                        cohort_strategy         TEXT          NOT NULL DEFAULT 'time_window',
                        baseline_row_count      INT           NOT NULL DEFAULT 0,
                        current_row_count       INT           NOT NULL DEFAULT 0
                    )
                    """
                )
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results ALTER COLUMN baseline_auc DROP NOT NULL"
                )
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results ALTER COLUMN current_auc DROP NOT NULL"
                )
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results ALTER COLUMN auc_delta DROP NOT NULL"
                )
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results "
                    "ADD COLUMN IF NOT EXISTS baseline_auc_source TEXT NOT NULL DEFAULT 'historical_baseline_window'"
                )
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results "
                    "ADD COLUMN IF NOT EXISTS cohort_strategy TEXT NOT NULL DEFAULT 'time_window'"
                )
                cur.execute(
                    "UPDATE processed.model_monitoring_results "
                    "SET baseline_auc_source = 'historical_baseline_window' "
                    "WHERE baseline_auc_source IS NULL"
                )
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results "
                    "ADD COLUMN IF NOT EXISTS baseline_row_count INT NOT NULL DEFAULT 0"
                )
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results "
                    "ADD COLUMN IF NOT EXISTS current_row_count INT NOT NULL DEFAULT 0"
                )
                # Workstream F: ensure the richer monitoring columns exist on
                # long-lived DBs that were created before the F migration.
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results "
                    "ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ok'"
                )
                for col, typ in [
                    ("brier_score", "NUMERIC(10,6)"),
                    ("max_calibration_error", "NUMERIC(10,6)"),
                    ("calibration_bins", "JSONB"),
                    ("score_distribution", "JSONB"),
                    ("segment_psi", "JSONB"),
                ]:
                    cur.execute(
                        f"ALTER TABLE processed.model_monitoring_results "
                        f"ADD COLUMN IF NOT EXISTS {col} {typ}"
                    )
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results "
                    "ADD COLUMN IF NOT EXISTS labeled_sample_count INT NOT NULL DEFAULT 0"
                )
                cur.execute(
                    "ALTER TABLE processed.model_monitoring_results "
                    "ADD COLUMN IF NOT EXISTS min_sample_threshold INT NOT NULL DEFAULT 0"
                )
                cur.execute(
                    """
                    INSERT INTO processed.model_monitoring_results (
                        baseline_window_start,
                        baseline_window_end,
                        current_window_start,
                        current_window_end,
                        baseline_auc,
                        current_auc,
                        auc_delta,
                        max_psi,
                        breached,
                        breached_reasons,
                        psi_by_feature,
                        baseline_auc_source,
                        cohort_strategy,
                        baseline_row_count,
                        current_row_count,
                        status,
                        brier_score,
                        max_calibration_error,
                        calibration_bins,
                        score_distribution,
                        segment_psi,
                        labeled_sample_count,
                        min_sample_threshold
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s,
                            %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s)
                    """,
                    (
                        baseline_window_start,
                        baseline_window_end,
                        current_window_start,
                        current_window_end,
                        baseline_auc,
                        current_auc,
                        auc_delta,
                        max_psi,
                        breached,
                        "; ".join(breached_reasons) if breached_reasons else None,
                        json.dumps(psi_by_feature),
                        baseline_auc_source,
                        cohort_strategy,
                        baseline_n,
                        current_n,
                        status,
                        brier_score,
                        max_cal_err,
                        json.dumps(calibration_bins),
                        json.dumps(score_distribution),
                        json.dumps(segment_psi),
                        labeled_sample_count,
                        MIN_LABELED_SAMPLES,
                    ),
                )

            conn.commit()

        return {
            "status": status,
            "breached": breached,
            "breached_reasons": breached_reasons,
            "predictions_table": predictions_table,
            "top_5_features": top_5_features,
            "baseline_auc": baseline_auc,
            "baseline_auc_source": baseline_auc_source,
            "current_auc": current_auc,
            "auc_delta": auc_delta,
            "max_psi": max_psi,
            "psi_by_feature": psi_by_feature,
            "cohort_strategy": cohort_strategy,
            "baseline_row_count": baseline_n,
            "current_row_count": current_n,
            "labeled_sample_count": labeled_sample_count,
            "min_sample_threshold": MIN_LABELED_SAMPLES,
            "brier_score": brier_score,
            "max_calibration_error": max_cal_err,
            "segment_psi": segment_psi,
        }

    @task.branch(task_id="evaluate_thresholds")
    def evaluate_thresholds(metrics: dict[str, Any]) -> str:
        return "log_monitoring_alert" if metrics["breached"] else "no_alert_needed"


    @task(task_id="log_monitoring_alert")
    def log_monitoring_alert(metrics: dict[str, Any]) -> None:
        print(
            "Model monitoring alert triggered; launching automated retraining.",
            {
                "auc_delta": metrics["auc_delta"],
                "max_psi": metrics["max_psi"],
                "breached_reasons": metrics["breached_reasons"],
                "predictions_table": metrics["predictions_table"],
                "top_5_features": metrics["top_5_features"],
                "baseline_auc_source": metrics["baseline_auc_source"],
                "cohort_strategy": metrics["cohort_strategy"],
                "baseline_row_count": metrics["baseline_row_count"],
                "current_row_count": metrics["current_row_count"],
            },
        )

    @task(task_id="no_alert_needed")
    def no_alert_needed(metrics: dict[str, Any]) -> None:
        print(
            "Monitoring passed without alert.",
            {
                "auc_delta": metrics["auc_delta"],
                "max_psi": metrics["max_psi"],
                "predictions_table": metrics["predictions_table"],
                "top_5_features": metrics["top_5_features"],
                "baseline_auc_source": metrics["baseline_auc_source"],
                "cohort_strategy": metrics["cohort_strategy"],
                "baseline_row_count": metrics["baseline_row_count"],
                "current_row_count": metrics["current_row_count"],
            },
        )

    metrics = compute_and_log_monitoring_metrics()
    branch = evaluate_thresholds(metrics)
    alert = log_monitoring_alert(metrics)
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="automated_retraining",
        wait_for_completion=False,
        reset_dag_run=False,
    )
    no_alert = no_alert_needed(metrics)

    branch >> [alert, no_alert]
    alert >> trigger_retraining


dag = weekly_model_monitoring()
