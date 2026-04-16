from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg2
from mlflow.models import get_model_info
from psycopg2.extras import execute_values

from source.common.db import get_current_snapshot_id, get_db_config

DB_CONFIG = get_db_config()

DEFAULT_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

SCORING_DIR = Path("data") / "scoring"
SCORING_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = SCORING_DIR / "features.parquet"
MODEL_METADATA_PATH = SCORING_DIR / "production_model.json"
SCORES_PATH = SCORING_DIR / "scores.parquet"

HIGH_THRESHOLD = float(os.getenv("CHURN_HIGH_THRESHOLD", "0.7"))
MEDIUM_THRESHOLD = float(os.getenv("CHURN_MEDIUM_THRESHOLD", "0.4"))

_REGISTRY_URI = "models:/KKBox-Churn-Classifier/Production"


@dataclass(frozen=True)
class ScoringArtifacts:
    features_path: Path = FEATURES_PATH
    model_metadata_path: Path = MODEL_METADATA_PATH
    scores_path: Path = SCORES_PATH


@dataclass(frozen=True)
class ProductionModelMetadata:
    model_uri: str
    input_columns: list[str]
    input_signature: dict[str, Any]
    model_name: str | None = None
    model_version: str | None = None


def _connect():
    return psycopg2.connect(**DB_CONFIG)


def _load_feature_table() -> pd.DataFrame:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'processed'
                      AND table_name = 'customer_features'
                );
                """
            )
            exists = bool(cur.fetchone()[0])

        if not exists:
            raise RuntimeError(
                "Missing feature store table `processed.customer_features`.\n"
                "This typically means Postgres schemas/tables were not initialized or the feature build has not run.\n"
                "Run, in order:\n"
                "  1) docker compose up -d\n"
                "  2) python source/dataops/load_raw_data.py\n"
                "  3) python source/dataops/build_customer_features.py\n"
                "Then re-run the Airflow DAG."
            )

        return pd.read_sql(
            """
            SELECT *
            FROM processed.customer_features
            ORDER BY msno
            """,
            conn,
        )


def airflow_load_features(artifacts: ScoringArtifacts | None = None, **_) -> None:
    artifacts = artifacts or ScoringArtifacts()
    df = _load_feature_table()
    df.to_parquet(artifacts.features_path, index=False)
    print(f"[load_features] Saved {len(df):,} rows to {artifacts.features_path}")


def _resolve_model_uri() -> str:
    return os.getenv("MLFLOW_MODEL_URI", _REGISTRY_URI)


def _load_model_info(model_uri: str):
    try:
        return get_model_info(model_uri)
    except Exception as exc:
        raise RuntimeError(
            "[load_production_model] Failed to resolve the production model at "
            f"{model_uri}. Ensure train_model.py and register_model.py completed "
            "successfully, or set MLFLOW_MODEL_URI to a valid MLflow model URI."
        ) from exc


def _load_serving_model(model_uri: str):
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise RuntimeError(
            "[load_production_model] Failed to load the production model from "
            f"{model_uri}. Ensure the MLflow tracking server is reachable and the "
            "registered model artifact exists."
        ) from exc

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model does not expose predict_proba(). "
            "Expected a fitted sklearn Pipeline logged by train_model.py."
        )
    return model


def _extract_input_columns(signature) -> list[str]:
    if signature is None or signature.inputs is None:
        raise RuntimeError(
            "Production model is missing an MLflow input signature. "
            "Retrain and re-register the model so scoring can enforce the serving schema."
        )

    input_columns = [str(name) for name in signature.inputs.input_names()]
    if not input_columns:
        raise RuntimeError(
            "Production model signature does not define named input columns. "
            "Retrain and re-register the model with a DataFrame input signature."
        )
    return input_columns


def _build_model_metadata(model_uri: str, model_info) -> ProductionModelMetadata:
    input_columns = _extract_input_columns(model_info.signature)
    return ProductionModelMetadata(
        model_uri=model_uri,
        input_columns=input_columns,
        input_signature=model_info.signature.to_dict(),
        model_name=getattr(model_info, "name", None),
        model_version=(
            str(model_info.registered_model_version)
            if getattr(model_info, "registered_model_version", None) is not None
            else None
        ),
    )


def _write_model_metadata(metadata: ProductionModelMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.__dict__, indent=2), encoding="utf-8")


def _load_model_metadata(path: Path) -> ProductionModelMetadata:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ProductionModelMetadata(**payload)


def airflow_load_production_model(artifacts: ScoringArtifacts | None = None, **_) -> None:
    artifacts = artifacts or ScoringArtifacts()

    model_uri = _resolve_model_uri()
    model_info = _load_model_info(model_uri)
    _load_serving_model(model_uri)
    metadata = _build_model_metadata(model_uri, model_info)
    _write_model_metadata(metadata, artifacts.model_metadata_path)

    version_suffix = f" version={metadata.model_version}" if metadata.model_version else ""
    print(
        "[load_production_model] Resolved model "
        f"{metadata.model_uri}{version_suffix} with inputs {metadata.input_columns} "
        f"-> {artifacts.model_metadata_path}"
    )


def _prepare_model_inputs(df: pd.DataFrame, input_columns: list[str]) -> pd.DataFrame:
    missing = [column for column in input_columns if column not in df.columns]
    if missing:
        raise RuntimeError(
            "Feature table is missing required production-model columns: "
            f"{missing}. Rebuild the feature table before scoring."
        )
    return df.loc[:, input_columns].copy()


def _predict_positive_class_proba(model, features: pd.DataFrame) -> np.ndarray:
    probs = np.asarray(model.predict_proba(features))
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise RuntimeError(
            "predict_proba returned an unexpected shape. "
            "Expected binary classification probabilities with two columns."
        )
    return probs[:, 1]


def _assign_risk_tier(prob: float) -> str:
    if prob >= HIGH_THRESHOLD:
        return "High"
    if prob >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def airflow_score(artifacts: ScoringArtifacts | None = None, **_) -> None:
    artifacts = artifacts or ScoringArtifacts()

    if not artifacts.features_path.exists():
        raise FileNotFoundError("Features parquet not found. Run load_features first.")
    if not artifacts.model_metadata_path.exists():
        raise FileNotFoundError(
            "Production model metadata not found. Run load_production_model first."
        )

    df = pd.read_parquet(artifacts.features_path)
    metadata = _load_model_metadata(artifacts.model_metadata_path)
    model = _load_serving_model(metadata.model_uri)
    model_inputs = _prepare_model_inputs(df, metadata.input_columns)
    probs = _predict_positive_class_proba(model, model_inputs)

    scores = pd.DataFrame(
        {
            "customer_id": df["msno"],
            "churn_probability": probs,
        }
    )
    scores["risk_tier"] = scores["churn_probability"].apply(_assign_risk_tier)
    scores.to_parquet(artifacts.scores_path, index=False)
    print(
        f"[score] Scored {len(scores):,} customers with {metadata.model_uri} "
        f"-> {artifacts.scores_path}"
    )


def airflow_write_predictions(artifacts: ScoringArtifacts | None = None, **_) -> None:
    artifacts = artifacts or ScoringArtifacts()

    if not artifacts.scores_path.exists():
        raise FileNotFoundError("Scores parquet not found. Run score first.")

    scores = pd.read_parquet(artifacts.scores_path)
    scored_at = datetime.now(timezone.utc)
    scores["scored_at"] = scored_at

    ddl = """
    CREATE TABLE IF NOT EXISTS processed.churn_predictions (
        customer_id         TEXT        NOT NULL,
        churn_probability   NUMERIC     NOT NULL,
        risk_tier           VARCHAR(16) NOT NULL,
        scored_at           TIMESTAMPTZ NOT NULL,
        shap_values         JSONB,
        feature_snapshot_id UUID,
        PRIMARY KEY (customer_id, scored_at)
    );
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(
                "ALTER TABLE processed.churn_predictions "
                "ADD COLUMN IF NOT EXISTS feature_snapshot_id UUID"
            )

        snapshot_id = get_current_snapshot_id(conn)

        rows = list(
            scores[["customer_id", "churn_probability", "risk_tier", "scored_at"]].itertuples(
                index=False, name=None
            )
        )
        rows = [(*row, snapshot_id) for row in rows]

        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO processed.churn_predictions (
                    customer_id, churn_probability, risk_tier, scored_at,
                    feature_snapshot_id
                )
                VALUES %s
                """,
                rows,
            )
        conn.commit()

    print(
        f"[write_predictions] Wrote {len(rows):,} rows into processed.churn_predictions"
        f" (feature_snapshot={snapshot_id})"
    )


def main(step: str | None = None) -> None:
    """
    CLI wrapper to support Airflow and local testing.

    Usage:
        python source/mlops/score_churn.py load_features
        python source/mlops/score_churn.py load_production_model
        python source/mlops/score_churn.py score
        python source/mlops/score_churn.py write_predictions
        python source/mlops/score_churn.py all
    """
    step = step or (sys.argv[1] if len(sys.argv) > 1 else "all")
    artifacts = ScoringArtifacts()

    if step == "load_features":
        airflow_load_features(artifacts)
    elif step == "load_production_model":
        airflow_load_production_model(artifacts)
    elif step == "score":
        airflow_score(artifacts)
    elif step == "write_predictions":
        airflow_write_predictions(artifacts)
    elif step == "all":
        airflow_load_features(artifacts)
        airflow_load_production_model(artifacts)
        airflow_score(artifacts)
        airflow_write_predictions(artifacts)
    else:
        raise SystemExit(f"Unknown step: {step}")


if __name__ == "__main__":
    main()
