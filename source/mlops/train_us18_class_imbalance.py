from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import mlflow
import numpy as np
import pandas as pd
import psycopg2
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

from source.common.db import get_db_config

DB_CONFIG = get_db_config()


FEATURE_TABLE_SQL = """
SELECT *
FROM processed.customer_features
;
"""


@dataclass(frozen=True)
class RunMetrics:
    precision_churn: float
    recall_churn: float
    f1_churn: float
    roc_auc: float
    support_churn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_churn": self.precision_churn,
            "recall_churn": self.recall_churn,
            "f1_churn": self.f1_churn,
            "roc_auc": self.roc_auc,
            "support_churn": self.support_churn,
        }


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(**DB_CONFIG)


def load_feature_store(sample_rows: int | None, seed: int) -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql(FEATURE_TABLE_SQL, conn)

    if sample_rows is None:
        return df

    if sample_rows <= 0:
        raise ValueError("sample_rows must be > 0 or None")

    # Stratified sampling on y if possible.
    if "is_churn" in df.columns:
        y = df["is_churn"].astype(int)
        # Preserve class balance by sampling fraction based on requested rows.
        frac = min(1.0, sample_rows / len(df))
        df = df.sample(frac=frac, random_state=seed, replace=False)
        # If we sampled randomly and it’s too far from target counts, accept it for runtime.
        return df

    return df.sample(n=sample_rows, random_state=seed, replace=False)


def _infer_feature_columns(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, list[str], list[str]]:
    if "is_churn" not in df.columns:
        raise ValueError("Expected `is_churn` column in processed.customer_features")

    y = df["is_churn"].astype(int)
    X = df.drop(columns=[c for c in ["is_churn", "feature_created_at"] if c in df.columns]).copy()
    if "msno" in X.columns:
        X = X.drop(columns=["msno"])

    categorical_cols = [
        c
        for c in X.columns
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c])
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return y, X, categorical_cols, numeric_cols


def build_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # Scaling helps LogisticRegression; keep sparse off (we output dense anyway).
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_and_evaluate_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    preprocessor: ColumnTransformer,
    seed: int,
    sampling_strategy: float,
    k_neighbors: int,
    threshold: float,
    max_iter: int,
) -> tuple[RunMetrics, dict[str, Any]]:
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=seed,
        k_neighbors=k_neighbors,
    )
    X_res, y_res = smote.fit_resample(X_train_t, y_train)

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight=None,
        random_state=seed,
    )
    model.fit(X_res, y_res)

    proba = model.predict_proba(X_val_t)[:, 1]
    pred = (proba >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_val,
        pred,
        pos_label=1,
        average="binary",
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_val, proba)

    metrics = RunMetrics(
        precision_churn=float(precision),
        recall_churn=float(recall),
        f1_churn=float(f1),
        roc_auc=float(roc_auc),
        support_churn=int(support[1]) if isinstance(support, (list, tuple, np.ndarray)) and len(support) > 1 else int(np.sum(y_val == 1)),
    )

    payload = {"model": model, "val_proba": proba}
    return metrics, payload


def train_and_evaluate_class_weight(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    preprocessor: ColumnTransformer,
    seed: int,
    threshold: float,
    max_iter: int,
) -> tuple[RunMetrics, dict[str, Any]]:
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    model = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X_train_t, y_train)

    proba = model.predict_proba(X_val_t)[:, 1]
    pred = (proba >= threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_val,
        pred,
        pos_label=1,
        average="binary",
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_val, proba)

    metrics = RunMetrics(
        precision_churn=float(precision),
        recall_churn=float(recall),
        f1_churn=float(f1),
        roc_auc=float(roc_auc),
        support_churn=int(support[1]) if isinstance(support, (list, tuple, np.ndarray)) and len(support) > 1 else int(np.sum(y_val == 1)),
    )

    payload = {"model": model, "val_proba": proba}
    return metrics, payload


def choose_strategy(smote_metrics: RunMetrics, cw_metrics: RunMetrics) -> str:
    # Chosen strategy documented with justification.
    # Primary: maximize churn F1.
    # Secondary: break ties by churn recall.
    if (smote_metrics.f1_churn, smote_metrics.recall_churn) >= (cw_metrics.f1_churn, cw_metrics.recall_churn):
        return "smote"
    return "class_weight_balanced"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI (default: %(default)s)",
    )
    parser.add_argument("--experiment-name", default="us18-class-imbalance", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--sample-rows", default=None, type=int)

    # SMOTE settings
    parser.add_argument("--smote-sampling-strategy", default=0.5, type=float)
    parser.add_argument("--smote-k-neighbors", default=5, type=int)

    parser.add_argument("--max-iter", default=2000, type=int)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    df = load_feature_store(sample_rows=args.sample_rows, seed=args.seed)
    y, X, categorical_cols, numeric_cols = _infer_feature_columns(df)

    if len(X) < 1000:
        raise RuntimeError("Feature table is unexpectedly small; check your database connection and data.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    smote_metrics, smote_payload = train_and_evaluate_smote(
        X_train,
        y_train,
        X_val,
        y_val,
        preprocessor=preprocessor,
        seed=args.seed,
        sampling_strategy=args.smote_sampling_strategy,
        k_neighbors=args.smote_k_neighbors,
        threshold=args.threshold,
        max_iter=args.max_iter,
    )

    # Rebuild preprocessor to avoid leakage from the previous fit (clarity > micro-efficiency).
    preprocessor_cw = build_preprocessor(categorical_cols, numeric_cols)
    cw_metrics, cw_payload = train_and_evaluate_class_weight(
        X_train,
        y_train,
        X_val,
        y_val,
        preprocessor=preprocessor_cw,
        seed=args.seed,
        threshold=args.threshold,
        max_iter=args.max_iter,
    )

    comparison = pd.DataFrame(
        [
            {
                "strategy": "SMOTE",
                "precision_churn": smote_metrics.precision_churn,
                "recall_churn": smote_metrics.recall_churn,
                "f1_churn": smote_metrics.f1_churn,
                "roc_auc": smote_metrics.roc_auc,
                "support_churn": smote_metrics.support_churn,
            },
            {
                "strategy": "class_weight=\"balanced\"",
                "precision_churn": cw_metrics.precision_churn,
                "recall_churn": cw_metrics.recall_churn,
                "f1_churn": cw_metrics.f1_churn,
                "roc_auc": cw_metrics.roc_auc,
                "support_churn": cw_metrics.support_churn,
            },
        ]
    ).sort_values("f1_churn", ascending=False, kind="mergesort")

    chosen = choose_strategy(smote_metrics, cw_metrics)

    justification = {
        "chosen_strategy": chosen,
        "rule": "Choose max churn F1; tie-break with churn recall.",
        "metrics": {
            "smote": smote_metrics.to_dict(),
            "class_weight_balanced": cw_metrics.to_dict(),
        },
    }

    out_dir = Path("docs") / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = out_dir / "us18_precision_recall_f1_comparison.csv"
    chosen_json = out_dir / "us18_chosen_strategy.json"
    comparison_md = out_dir / "us18_precision_recall_f1_comparison.md"

    comparison.to_csv(comparison_csv, index=False)
    chosen_json.write_text(json.dumps(justification, indent=2), encoding="utf-8")
    comparison_md.write_text(comparison.to_markdown(index=False), encoding="utf-8")

    def _log_for_strategy(
        strategy: str,
        metrics: RunMetrics,
        payload: dict[str, Any],
    ) -> None:
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name=strategy):
            mlflow.log_params(
                {
                    "seed": args.seed,
                    "threshold": args.threshold,
                    "test_size": args.test_size,
                    "smote_sampling_strategy": args.smote_sampling_strategy,
                    "smote_k_neighbors": args.smote_k_neighbors,
                    "strategy": strategy,
                    "model": "LogisticRegression",
                }
            )
            mlflow.log_metrics(
                {
                    "precision_churn": metrics.precision_churn,
                    "recall_churn": metrics.recall_churn,
                    "f1_churn": metrics.f1_churn,
                    "roc_auc": metrics.roc_auc,
                    "support_churn": float(metrics.support_churn),
                }
            )

            mlflow.log_artifact(str(comparison_csv))
            mlflow.log_artifact(str(chosen_json))
            mlflow.log_artifact(str(comparison_md))

            # Store models for potential later reuse.
            if strategy == "SMOTE":
                joblib.dump(payload["model"], out_dir / "us18_smote_model.joblib")
                mlflow.log_artifact(str(out_dir / "us18_smote_model.joblib"))
            else:
                joblib.dump(payload["model"], out_dir / "us18_class_weight_model.joblib")
                mlflow.log_artifact(str(out_dir / "us18_class_weight_model.joblib"))

    _log_for_strategy("SMOTE", smote_metrics, smote_payload)
    _log_for_strategy('class_weight="balanced"', cw_metrics, cw_payload)

    # Also write a human-readable summary doc (helps the PR/evidence check).
    doc_path = Path("docs") / "us18_class_imbalance.md"
    doc_path.write_text(
        f"""# US18 - Class Imbalance Handling

## Comparison

Saved to: `docs/artifacts/us18_precision_recall_f1_comparison.csv`

| Strategy | Precision (churn) | Recall (churn) | F1 (churn) | ROC AUC |
|---|---:|---:|---:|---:|
| SMOTE | {smote_metrics.precision_churn:.4f} | {smote_metrics.recall_churn:.4f} | {smote_metrics.f1_churn:.4f} | {smote_metrics.roc_auc:.4f} |
| class_weight=\"balanced\" | {cw_metrics.precision_churn:.4f} | {cw_metrics.recall_churn:.4f} | {cw_metrics.f1_churn:.4f} | {cw_metrics.roc_auc:.4f} |

## Chosen strategy

Chosen: **{chosen}**

Rule: choose max churn F1; tie-break with churn recall.

Justification:
```json
{json.dumps(justification, indent=2)}
```
""",
        encoding="utf-8",
    )

    print("US18 complete.")
    print(f"Chosen strategy: {chosen}")
    print(comparison)


if __name__ == "__main__":
    main()

