"""US-19 SHAP Explainability.

Generate SHAP values for the production churn model, produce summary and
waterfall plots, persist per-prediction explanations to the database, and
log artifacts to MLflow.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg2
import shap
from psycopg2.extras import execute_values
from xgboost import XGBClassifier

from train_model import (
    CATEGORICAL_FEATURES,
    _split_columns,
    build_preprocessor,
    load_selected_features,
)

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
ARTIFACT_DIR = _PROJECT_ROOT / "docs" / "artifacts"
DEFAULT_TRACKING_URI = "http://localhost:5001"

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "kkbox"),
    "user": os.getenv("POSTGRES_USER", "bt4301"),
    "password": os.getenv("POSTGRES_PASSWORD", "bt4301pass"),
}

_REGISTRY_URI = "models:/KKBox-Churn-Classifier/Production"
_LOCAL_MODEL_DIR = _PROJECT_ROOT / "data" / "scoring"
EXPERIMENT_NAME = "KKBox Churn"

# Allowlist of valid feature column names (prevents SQL injection).
_KNOWN_FEATURE_COLUMNS = frozenset(
    {
        "msno",
        "city",
        "bd",
        "gender",
        "registered_via",
        "registration_init_time",
        "transaction_count",
        "renewal_count",
        "cancel_count",
        "total_amount_paid",
        "avg_plan_days",
        "latest_payment_method_id",
        "latest_is_auto_renew",
        "latest_membership_expire_date",
        "num_active_days",
        "total_secs",
        "avg_total_secs",
        "total_num_songs",
        "avg_num_songs",
        "total_num_unq",
        "avg_num_unq",
    }
)


@dataclass(frozen=True)
class SampleCustomer:
    """A single customer chosen for a waterfall plot."""

    customer_id: str
    tier: str
    churn_probability: float
    index: int  # row position in the aligned dataframe


def _connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(**DB_CONFIG)


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------

def load_production_model() -> tuple[Any, str, Any | None]:
    """Load the raw classifier from MLflow or local fallback.

    Returns ``(model, model_type, embedded_preprocessor)`` where
    *model_type* is ``"xgboost"`` or ``"other"`` and
    *embedded_preprocessor* is ``None`` when the model was loaded
    standalone (MLflow registry) or the Pipeline's own preprocessor
    when loaded from a local fallback Pipeline.
    """
    embedded_preprocessor = None
    try:
        logger.info("Loading model from MLflow registry: %s", _REGISTRY_URI)
        model = mlflow.sklearn.load_model(_REGISTRY_URI)
    except mlflow.exceptions.MlflowException:
        logger.warning(
            "MLflow registry unavailable; trying local fallback at %s",
            _LOCAL_MODEL_DIR,
        )
        import joblib

        model_path = _LOCAL_MODEL_DIR / "production_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model found at {model_path}. "
                "Run the scoring pipeline first (python source/mlops/score_churn.py all)."
            )
        model = joblib.load(model_path)

        # If loaded via joblib from score_churn the object may be an
        # sklearn Pipeline or an mlflow.pyfunc wrapper.  Extract the
        # classifier and keep the embedded preprocessor for SHAP.
        if hasattr(model, "named_steps"):
            embedded_preprocessor = model.named_steps.get("pre", None)
            model = model.named_steps.get("clf", model)
        if hasattr(model, "_model_impl"):
            model = model._model_impl

    model_type = "xgboost" if isinstance(model, XGBClassifier) else "other"
    logger.info("Loaded model type: %s (%s)", type(model).__name__, model_type)
    return model, model_type, embedded_preprocessor


# ---------------------------------------------------------------------------
# 2. Load features + predictions
# ---------------------------------------------------------------------------

def load_features_and_predictions(
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature data and latest predictions, joined on customer ID."""
    # Validate feature names against allowlist.
    unknown = set(selected_features) - _KNOWN_FEATURE_COLUMNS
    if unknown:
        raise ValueError(f"Unknown feature columns rejected: {unknown}")

    cols = ", ".join(["msno"] + selected_features)
    feature_sql = f"SELECT {cols} FROM processed.customer_features"

    pred_sql = """
        SELECT DISTINCT ON (customer_id)
               customer_id, churn_probability, risk_tier, scored_at
        FROM processed.churn_predictions
        ORDER BY customer_id, scored_at DESC
    """

    with _connect() as conn:
        features_df = pd.read_sql(feature_sql, conn)
        preds_df = pd.read_sql(pred_sql, conn)

    if preds_df.empty:
        raise RuntimeError(
            "No predictions found in processed.churn_predictions. "
            "Run the scoring pipeline first."
        )

    merged = preds_df.merge(
        features_df, left_on="customer_id", right_on="msno", how="inner",
    )
    if merged.empty:
        raise RuntimeError(
            "No matching rows after joining predictions with features. "
            "Ensure both tables are populated."
        )

    # Separate back into features-only and predictions-only DataFrames,
    # aligned by row position.
    feat_out = merged[selected_features].reset_index(drop=True)
    pred_out = merged[["customer_id", "churn_probability", "risk_tier", "scored_at"]].reset_index(
        drop=True
    )
    logger.info("Loaded %d customers with predictions and features.", len(feat_out))
    return feat_out, pred_out


# ---------------------------------------------------------------------------
# 3. Build SHAP explainer
# ---------------------------------------------------------------------------

def build_explainer(
    model: Any,
    model_type: str,
    X_transformed: np.ndarray,
) -> shap.Explainer:
    """Create an appropriate SHAP explainer for the model type."""
    if model_type == "xgboost":
        return shap.TreeExplainer(model)
    return shap.KernelExplainer(
        model.predict_proba, shap.sample(X_transformed, 100),
    )


# ---------------------------------------------------------------------------
# 4. Compute SHAP values
# ---------------------------------------------------------------------------

def compute_shap_values(
    explainer: shap.Explainer,
    X_transformed: np.ndarray,
    feature_names: list[str],
) -> shap.Explanation:
    """Compute SHAP values and return an Explanation for churn (class 1)."""
    raw = explainer(X_transformed)

    # For binary classification TreeExplainer returns shape
    # (n_samples, n_features, 2).  Select class-1 slice.
    if raw.values.ndim == 3:
        vals = raw.values[:, :, 1]
        base = raw.base_values[:, 1] if raw.base_values.ndim == 2 else raw.base_values
    else:
        vals = raw.values
        base = raw.base_values

    return shap.Explanation(
        values=vals,
        base_values=base,
        data=X_transformed,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# 5. Aggregate one-hot SHAP back to original features
# ---------------------------------------------------------------------------

def aggregate_onehot_shap(
    shap_explanation: shap.Explanation,
    transformed_names: list[str],
    original_features: list[str],
    cat_features: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Sum one-hot encoded SHAP columns back to original feature names.

    Returns ``(aggregated_matrix, original_feature_names)`` where the
    matrix has shape ``(n_samples, len(original_features))``.
    """
    vals = shap_explanation.values  # (n_samples, n_transformed)
    agg = np.zeros((vals.shape[0], len(original_features)), dtype=np.float64)

    for j, tname in enumerate(transformed_names):
        matched = False
        for i, orig in enumerate(original_features):
            if orig in cat_features:
                # One-hot columns are named like "gender_male" or just "gender".
                if tname == orig or tname.startswith(f"{orig}_"):
                    agg[:, i] += vals[:, j]
                    matched = True
                    break
            else:
                if tname == orig:
                    agg[:, i] += vals[:, j]
                    matched = True
                    break
        if not matched:
            logger.debug("Transformed column %r not mapped — skipping.", tname)

    return agg, original_features


# ---------------------------------------------------------------------------
# 6. Pick sample customers
# ---------------------------------------------------------------------------

def pick_sample_customers(predictions_df: pd.DataFrame) -> list[SampleCustomer]:
    """Select one customer per risk tier for waterfall plots."""
    samples: list[SampleCustomer] = []

    for tier, pick_fn in [
        ("High", lambda g: g["churn_probability"].idxmax()),
        ("Medium", lambda g: (g["churn_probability"] - g["churn_probability"].median()).abs().idxmin()),
        ("Low", lambda g: g["churn_probability"].idxmin()),
    ]:
        group = predictions_df[predictions_df["risk_tier"] == tier]
        if group.empty:
            logger.warning("No customers in %s tier — skipping waterfall.", tier)
            continue
        idx = pick_fn(group)
        row = predictions_df.loc[idx]
        samples.append(
            SampleCustomer(
                customer_id=str(row["customer_id"]),
                tier=tier.lower(),
                churn_probability=float(row["churn_probability"]),
                index=int(idx),
            )
        )

    return samples


# ---------------------------------------------------------------------------
# 7. Summary plot
# ---------------------------------------------------------------------------

def plot_summary(shap_explanation: shap.Explanation) -> Path:
    """Generate a SHAP summary (beeswarm) plot and save to disk."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "us19_shap_summary.png"

    shap.summary_plot(shap_explanation, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Summary plot saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# 8. Waterfall plot
# ---------------------------------------------------------------------------

def plot_waterfall(shap_explanation: shap.Explanation, sample: SampleCustomer) -> Path:
    """Generate a SHAP waterfall plot for one customer."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"us19_shap_waterfall_{sample.tier}.png"

    shap.waterfall_plot(shap_explanation[sample.index], show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(
        "Waterfall plot (%s tier, customer=%s) saved to %s",
        sample.tier, sample.customer_id, path,
    )
    return path


# ---------------------------------------------------------------------------
# 9. Format SHAP for DB
# ---------------------------------------------------------------------------

def format_shap_for_db(
    shap_row: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-K SHAP values as a list of dicts for JSONB storage."""
    pairs = list(zip(feature_names, shap_row.tolist()))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [
        {"feature": name, "shap_value": round(float(val), 6)}
        for name, val in pairs[:top_k]
    ]


# ---------------------------------------------------------------------------
# 10. Update predictions with SHAP values
# ---------------------------------------------------------------------------

def update_predictions_with_shap(
    shap_matrix: np.ndarray,
    feature_names: list[str],
    predictions_df: pd.DataFrame,
    top_k: int = 5,
) -> int:
    """Batch-update ``shap_values`` in ``processed.churn_predictions``."""
    rows: list[tuple[str, str, str]] = []
    for i in range(len(predictions_df)):
        cid = str(predictions_df.iloc[i]["customer_id"])
        scored = str(predictions_df.iloc[i]["scored_at"])
        payload = json.dumps(format_shap_for_db(shap_matrix[i], feature_names, top_k))
        rows.append((payload, cid, scored))

    updated = 0
    batch_size = 1000

    with _connect() as conn:
        with conn.cursor() as cur:
            # Ensure column exists (idempotent).
            cur.execute(
                "ALTER TABLE processed.churn_predictions "
                "ADD COLUMN IF NOT EXISTS shap_values JSONB;"
            )
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                execute_values(
                    cur,
                    """
                    UPDATE processed.churn_predictions AS cp
                    SET shap_values = v.shap_values::jsonb
                    FROM (VALUES %s) AS v(shap_values, customer_id, scored_at)
                    WHERE cp.customer_id = v.customer_id
                      AND cp.scored_at = v.scored_at::timestamptz
                    """,
                    batch,
                )
                updated += cur.rowcount
        conn.commit()

    logger.info("Updated %d prediction rows with SHAP values.", updated)
    return updated


# ---------------------------------------------------------------------------
# 11. Log to MLflow
# ---------------------------------------------------------------------------

def log_to_mlflow(artifact_paths: list[Path]) -> str:
    """Create an MLflow run and log all artifact PNGs."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="US-19 SHAP Explainability") as run:
        for path in artifact_paths:
            mlflow.log_artifact(str(path))
        run_id = run.info.run_id

    logger.info("MLflow run logged: %s", run_id)
    return run_id


# ---------------------------------------------------------------------------
# 12. Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="US-19: SHAP Explainability")
    parser.add_argument(
        "--tracking-uri", default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top SHAP features to store per prediction (default: 5)",
    )
    parser.add_argument(
        "--skip-db-update", action="store_true",
        help="Skip writing SHAP values back to the database",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load selected features ---
    selected_features = load_selected_features()
    logger.info("Selected features (%d): %s", len(selected_features), selected_features)

    cat_cols, num_cols = _split_columns(selected_features)

    # --- Load model ---
    model, model_type, embedded_preprocessor = load_production_model()

    # --- Load data ---
    features_df, predictions_df = load_features_and_predictions(selected_features)

    # --- Preprocess features ---
    # raw_feature_data holds the untransformed feature values matching
    # selected_features — used as ``data`` in the aggregated shap.Explanation.
    if embedded_preprocessor is not None:
        # Local fallback Pipeline: the preprocessor is already fitted and
        # expects the full feature set (all columns from customer_features).
        # We need to load all features, not just the selected 12.
        logger.info("Using embedded preprocessor from fallback Pipeline.")
        with _connect() as conn:
            all_features_df = pd.read_sql(
                "SELECT * FROM processed.customer_features ORDER BY msno", conn,
            )
        # Align with predictions by customer_id.
        merged = predictions_df.merge(
            all_features_df, left_on="customer_id", right_on="msno", how="inner",
        )
        drop_cols = ["msno", "is_churn", "feature_created_at",
                     "customer_id", "churn_probability", "risk_tier", "scored_at"]
        X_for_transform = merged.drop(
            columns=[c for c in drop_cols if c in merged.columns],
        )
        X_transformed = embedded_preprocessor.transform(X_for_transform)
        transformed_names = list(embedded_preprocessor.get_feature_names_out())
        # For the fallback model, features are not the selected 12 so we
        # use the transformer's output names directly (no one-hot aggregation).
        cat_cols = [c for c in X_for_transform.columns if X_for_transform[c].dtype == object]
        selected_features = list(X_for_transform.columns)
        raw_feature_data = X_for_transform.values
    else:
        # Production model from MLflow: rebuild preprocessor on selected features.
        preprocessor = build_preprocessor(cat_cols, num_cols)
        X_transformed = preprocessor.fit_transform(features_df)
        transformed_names = list(preprocessor.get_feature_names_out())
        raw_feature_data = features_df.values

    # --- SHAP ---
    explainer = build_explainer(model, model_type, X_transformed)
    shap_explanation = compute_shap_values(explainer, X_transformed, transformed_names)

    # --- Aggregate one-hot back to original features ---
    agg_matrix, agg_names = aggregate_onehot_shap(
        shap_explanation, transformed_names, selected_features, cat_cols,
    )
    agg_explanation = shap.Explanation(
        values=agg_matrix,
        base_values=shap_explanation.base_values,
        data=raw_feature_data,
        feature_names=agg_names,
    )

    # --- Plots ---
    artifact_paths: list[Path] = []

    summary_path = plot_summary(agg_explanation)
    artifact_paths.append(summary_path)

    samples = pick_sample_customers(predictions_df)
    for sample in samples:
        wf_path = plot_waterfall(agg_explanation, sample)
        artifact_paths.append(wf_path)

    # --- DB update ---
    if not args.skip_db_update:
        updated = update_predictions_with_shap(
            agg_matrix, agg_names, predictions_df, args.top_k,
        )
        logger.info("Database update complete: %d rows.", updated)
    else:
        logger.info("Skipping database update (--skip-db-update).")

    # --- MLflow ---
    run_id = log_to_mlflow(artifact_paths)

    logger.info("US-19 SHAP explainability complete. MLflow run: %s", run_id)
    logger.info("Artifacts: %s", [str(p) for p in artifact_paths])


if __name__ == "__main__":
    main()
