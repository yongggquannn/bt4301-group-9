from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlflow
import numpy as np
import pandas as pd
import psycopg2
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
DEFAULT_N_JOBS = int(os.getenv("FEATURE_SELECTION_N_JOBS", "1"))


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected_features: list[str]
    importance_table: pd.DataFrame
    dropped_due_to_correlation: list[str]


DEFAULT_DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "kkbox"),
    "user": os.getenv("POSTGRES_USER", "bt4301"),
    "password": os.getenv("POSTGRES_PASSWORD", "bt4301pass"),
}


def load_feature_store(db_config: dict = DEFAULT_DB_CONFIG) -> pd.DataFrame:
    conn = psycopg2.connect(**db_config)
    try:
        return pd.read_sql(
            """
            SELECT *
            FROM processed.customer_features
            ORDER BY msno
            """,
            conn,
        )
    finally:
        conn.close()


def _infer_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in ["gender"] if c in df.columns]
    categorical += [
        c
        for c in [
            "city",
            "registered_via",
            "latest_payment_method_id",
            "latest_is_auto_renew",
        ]
        if c in df.columns
    ]
    categorical = sorted(set(categorical))

    excluded = {"msno", "is_churn", "feature_created_at"}
    numeric = [
        c
        for c in df.columns
        if c not in excluded and c not in categorical and pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric = sorted(set(numeric))
    return categorical, numeric


def _build_pipeline(
    categorical_cols: list[str],
    numeric_cols: list[str],
    seed: int,
    n_jobs: int,
) -> Pipeline:
    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def _correlation_filter(
    df: pd.DataFrame,
    numeric_cols: list[str],
    importances: dict[str, float],
    threshold: float,
) -> tuple[list[str], list[str]]:
    if not numeric_cols:
        return [], []

    corr = df[numeric_cols].corr(numeric_only=True).abs()
    kept: list[str] = []
    dropped: list[str] = []

    # Greedy: process by descending importance (keep most useful first)
    ordered = sorted(numeric_cols, key=lambda c: importances.get(c, -np.inf), reverse=True)
    for col in ordered:
        if col in dropped:
            continue
        ok = True
        for k in kept:
            if pd.notna(corr.loc[col, k]) and corr.loc[col, k] >= threshold:
                ok = False
                break
        if ok:
            kept.append(col)
        else:
            dropped.append(col)

    return kept, dropped


def run_feature_selection(
    df: pd.DataFrame,
    *,
    experiment_name: str = "feature-selection",
    run_name: str = "permutation-importance",
    seed: int = 42,
    test_size: float = 0.2,
    min_features: int = 8,
    importance_threshold: float = 0.0,
    correlation_threshold: float = 0.9,
    n_repeats: int = 8,
    n_jobs: int = DEFAULT_N_JOBS,
) -> FeatureSelectionResult:
    if "is_churn" not in df.columns:
        raise ValueError("Expected target column 'is_churn' in feature store.")

    df = df.copy()
    y = df["is_churn"].astype(int)
    X = df.drop(columns=[c for c in ["is_churn", "feature_created_at"] if c in df.columns])
    if "msno" in X.columns:
        X = X.drop(columns=["msno"])

    categorical_cols, numeric_cols = _infer_columns(df)
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in numeric_cols if c in X.columns]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    pipe = _build_pipeline(categorical_cols, numeric_cols, seed=seed, n_jobs=n_jobs)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "seed": seed,
                "test_size": test_size,
                "min_features": min_features,
                "importance_threshold": importance_threshold,
                "correlation_threshold": correlation_threshold,
                "permutation_n_repeats": n_repeats,
                "n_jobs": n_jobs,
                "model": "RandomForestClassifier",
            }
        )

        pipe.fit(X_train, y_train)
        val_proba = pipe.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        mlflow.log_metric("val_auc", float(val_auc))

        # Model-based importance (impurity-based) on transformed feature space
        model_importance_path = None
        try:
            pre = pipe.named_steps["pre"]
            clf = pipe.named_steps["clf"]
            feature_names = pre.get_feature_names_out()
            model_importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "model_importance": getattr(clf, "feature_importances_"),
                }
            ).sort_values("model_importance", ascending=False, kind="mergesort")
            model_importance_path = (Path("docs") / "artifacts" / "model_feature_importance.csv")
        except Exception:
            model_importance_df = None

        perm = permutation_importance(
            pipe,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=seed,
            scoring="roc_auc",
            n_jobs=n_jobs,
        )

        importance_df = pd.DataFrame(
            {
                "feature": list(X_val.columns),
                "perm_importance_mean": perm.importances_mean,
                "perm_importance_std": perm.importances_std,
            }
        ).sort_values("perm_importance_mean", ascending=False, kind="mergesort")

        importances = {
            r["feature"]: float(r["perm_importance_mean"])
            for _, r in importance_df.iterrows()
        }

        # Base selection: threshold + ensure at least min_features
        threshold_selected = [
            f for f in importance_df["feature"].tolist() if importances.get(f, -np.inf) > importance_threshold
        ]
        if len(threshold_selected) < min_features:
            threshold_selected = importance_df["feature"].head(min_features).tolist()

        # Correlation filter on numeric only, applied after base selection
        numeric_selected = [c for c in threshold_selected if c in numeric_cols]
        kept_numeric, dropped_corr = _correlation_filter(
            X_train,
            numeric_selected,
            importances=importances,
            threshold=correlation_threshold,
        )
        selected = [c for c in threshold_selected if c not in numeric_cols] + kept_numeric
        selected = list(dict.fromkeys(selected))  # stable de-dupe

        out_dir = Path("docs") / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)

        features_path = out_dir / "final_feature_set.json"
        features_csv_path = out_dir / "final_feature_set.csv"
        importance_path = out_dir / "permutation_importance.csv"
        if model_importance_df is not None:
            model_importance_path = out_dir / "model_feature_importance.csv"

        features_payload = {
            "selected_features": selected,
            "selected_feature_count": len(selected),
            "selection_method": {
                "importance": "Permutation importance (ROC AUC drop) + model impurity importance (RandomForestClassifier)",
                "rule": f"Keep features with perm_importance_mean > {importance_threshold} (min {min_features}), then drop numeric features with abs(corr) >= {correlation_threshold} keeping higher-importance feature.",
            },
            "dropped_due_to_correlation": dropped_corr,
        }

        features_path.write_text(json.dumps(features_payload, indent=2), encoding="utf-8")
        pd.DataFrame({"feature": selected}).to_csv(features_csv_path, index=False)
        importance_df.to_csv(importance_path, index=False)
        if model_importance_df is not None:
            model_importance_df.to_csv(model_importance_path, index=False)

        mlflow.log_artifact(str(features_path))
        mlflow.log_artifact(str(features_csv_path))
        mlflow.log_artifact(str(importance_path))
        if model_importance_df is not None:
            mlflow.log_artifact(str(model_importance_path))

        mlflow.log_metrics(
            {
                "selected_feature_count": float(len(selected)),
                "dropped_corr_count": float(len(dropped_corr)),
            }
        )

    return FeatureSelectionResult(
        selected_features=selected,
        importance_table=importance_df,
        dropped_due_to_correlation=dropped_corr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature selection with permutation importance",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI (default: %(default)s)",
    )
    parser.add_argument(
        "--n-jobs",
        default=DEFAULT_N_JOBS,
        type=int,
        help="Parallel workers for feature selection (default: %(default)s)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    df = load_feature_store()
    res = run_feature_selection(df, n_jobs=args.n_jobs)
    print(f"Selected {len(res.selected_features)} features:")
    for f in res.selected_features:
        print(f"  - {f}")


if __name__ == "__main__":
    main()

