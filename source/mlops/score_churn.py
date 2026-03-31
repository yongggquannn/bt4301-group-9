from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "kkbox"),
    "user": os.getenv("POSTGRES_USER", "bt4301"),
    "password": os.getenv("POSTGRES_PASSWORD", "bt4301pass"),
}

SCORING_DIR = Path("data") / "scoring"
SCORING_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = SCORING_DIR / "features.parquet"
MODEL_PATH = SCORING_DIR / "production_model.pkl"
SCORES_PATH = SCORING_DIR / "scores.parquet"

HIGH_THRESHOLD = float(os.getenv("CHURN_HIGH_THRESHOLD", "0.7"))
MEDIUM_THRESHOLD = float(os.getenv("CHURN_MEDIUM_THRESHOLD", "0.4"))


@dataclass
class ScoringArtifacts:
    features_path: Path = FEATURES_PATH
    model_path: Path = MODEL_PATH
    scores_path: Path = SCORES_PATH


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


def _train_fallback_model(df: pd.DataFrame) -> Pipeline:
    if "is_churn" not in df.columns:
        raise ValueError("Expected 'is_churn' column in feature table.")

    y = df["is_churn"].astype(int)
    X = df.drop(columns=[c for c in ["msno", "is_churn", "feature_created_at"] if c in df.columns])

    # Build a preprocessing pipeline that can handle both numeric and categorical columns.
    # This fallback exists only when MLFLOW_MODEL_URI is not set.
    categorical_cols = [c for c in X.columns if X[c].dtype == object]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Keep the fallback model small so Airflow scoring DAG runs quickly
    # when no MLflow production model is provided.
    clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)
    return pipe


_REGISTRY_URI = "models:/KKBox-Churn-Classifier/Production"


def _load_model_from_mlflow(df: pd.DataFrame) -> object:
    model_uri = os.getenv("MLFLOW_MODEL_URI")

    if not model_uri:
        # Try the Model Registry (US-12) before falling back to a scratch model.
        try:
            print(
                f"[load_production_model] MLFLOW_MODEL_URI not set; "
                f"trying registry at {_REGISTRY_URI}"
            )
            return mlflow.pyfunc.load_model(_REGISTRY_URI)
        except mlflow.exceptions.MlflowException as exc:
            print(
                f"[load_production_model] Registry model not available ({exc}); "
                "training a simple fallback RandomForest model instead."
            )
            return _train_fallback_model(df)

    print(f"[load_production_model] Loading production model from MLflow: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def airflow_load_production_model(artifacts: ScoringArtifacts | None = None, **_) -> None:
    artifacts = artifacts or ScoringArtifacts()

    if not artifacts.features_path.exists():
        raise FileNotFoundError(
            f"Features file not found at {artifacts.features_path}. "
            "Run load_features task first."
        )

    df = pd.read_parquet(artifacts.features_path)
    model = _load_model_from_mlflow(df)

    import joblib

    joblib.dump(model, artifacts.model_path)
    print(f"[load_production_model] Saved model to {artifacts.model_path}")


def _predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    # sklearn Pipeline
    if hasattr(model, "predict_proba"):
        X = df.drop(columns=[c for c in ["msno", "is_churn", "feature_created_at"] if c in df.columns])
        return model.predict_proba(X)[:, 1]

    # mlflow.pyfunc generic model
    preds = model.predict(df)
    preds = np.asarray(preds).reshape(-1)
    return preds


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
    if not artifacts.model_path.exists():
        raise FileNotFoundError("Model pickle not found. Run load_production_model first.")

    df = pd.read_parquet(artifacts.features_path)

    import joblib

    model = joblib.load(artifacts.model_path)
    probs = _predict_proba(model, df)

    scores = pd.DataFrame(
        {
            "customer_id": df["msno"],
            "churn_probability": probs,
        }
    )
    scores["risk_tier"] = scores["churn_probability"].apply(_assign_risk_tier)
    scores.to_parquet(artifacts.scores_path, index=False)
    print(f"[score] Scored {len(scores):,} customers -> {artifacts.scores_path}")


def airflow_write_predictions(artifacts: ScoringArtifacts | None = None, **_) -> None:
    artifacts = artifacts or ScoringArtifacts()

    if not artifacts.scores_path.exists():
        raise FileNotFoundError("Scores parquet not found. Run score first.")

    scores = pd.read_parquet(artifacts.scores_path)
    scored_at = datetime.now(timezone.utc)
    scores["scored_at"] = scored_at

    rows = list(
        scores[["customer_id", "churn_probability", "risk_tier", "scored_at"]].itertuples(
            index=False, name=None
        )
    )

    ddl = """
    CREATE TABLE IF NOT EXISTS processed.churn_predictions (
        customer_id       TEXT        NOT NULL,
        churn_probability NUMERIC     NOT NULL,
        risk_tier         VARCHAR(16) NOT NULL,
        scored_at         TIMESTAMPTZ NOT NULL,
        shap_values       JSONB,
        PRIMARY KEY (customer_id, scored_at)
    );
    """

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
            execute_values(
                cur,
                """
                INSERT INTO processed.churn_predictions (
                    customer_id, churn_probability, risk_tier, scored_at
                )
                VALUES %s
                """,
                rows,
            )
        conn.commit()

    print(f"[write_predictions] Wrote {len(rows):,} rows into processed.churn_predictions")


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

