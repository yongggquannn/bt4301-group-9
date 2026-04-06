"""US-15 Model governance artifacts: model card, fairness tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "artifacts"


def describe_training_date_range(df: pd.DataFrame) -> str:
    """Summarize the feature snapshot date range used for training."""
    if "feature_created_at" not in df.columns:
        return "Not available (feature_created_at not loaded)."
    ts = pd.to_datetime(df["feature_created_at"], errors="coerce").dropna()
    if ts.empty:
        return "Not available (feature_created_at empty)."
    return f"{ts.min().date().isoformat()} to {ts.max().date().isoformat()}"


def age_band_from_bd(series: pd.Series) -> pd.Series:
    """Bucket ages into coarse bands for descriptive fairness reporting."""
    s = pd.to_numeric(series, errors="coerce")
    bands = pd.Series("Unknown", index=series.index, dtype="object")
    bands[(s >= 18) & (s <= 24)] = "18-24"
    bands[(s >= 25) & (s <= 34)] = "25-34"
    bands[(s >= 35) & (s <= 44)] = "35-44"
    bands[(s >= 45) & (s <= 54)] = "45-54"
    bands[s >= 55] = "55+"
    return bands


def build_group_fairness_table(
    governance_val: pd.DataFrame,
    y_val: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float,
    group_col: str,
) -> pd.DataFrame:
    """Compute churn-rate summaries by gender or age band on validation data."""
    work = pd.DataFrame(index=governance_val.index)
    if group_col == "age_band":
        if "bd" in governance_val.columns:
            work[group_col] = age_band_from_bd(governance_val["bd"])
        else:
            work[group_col] = "Unknown"
    elif group_col == "gender":
        if "gender" in governance_val.columns:
            work[group_col] = governance_val["gender"].astype("object").fillna("Unknown")
            work[group_col] = work[group_col].replace("", "Unknown")
        else:
            work[group_col] = "Unknown"
    else:
        raise ValueError(f"Unsupported group column: {group_col}")

    work["is_churn"] = np.asarray(y_val)
    work["pred_churn"] = (np.asarray(y_proba) >= threshold).astype(int)

    out = (
        work.groupby(group_col, dropna=False)
        .agg(
            sample_size=("is_churn", "size"),
            churn_rate=("is_churn", "mean"),
            predicted_churn_rate=("pred_churn", "mean"),
        )
        .reset_index()
    )
    out["churn_rate"] = out["churn_rate"].astype(float)
    out["predicted_churn_rate"] = out["predicted_churn_rate"].astype(float)
    return out.sort_values(group_col, kind="mergesort").reset_index(drop=True)


def write_us15_governance_artifacts(
    *,
    model_name: str,
    selected_features: list[str],
    full_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    governance_val: pd.DataFrame,
    y_val: np.ndarray,
    y_proba: np.ndarray,
    metrics: dict[str, float],
    threshold: float,
) -> dict[str, Any]:
    """Create model-card and fairness artifacts for the chosen production model."""
    feature_types = pd.DataFrame(
        {
            "feature_name": selected_features,
            "dtype": [str(feature_frame[c].dtype) for c in selected_features],
        }
    )
    fairness_gender = build_group_fairness_table(
        governance_val,
        y_val,
        y_proba,
        threshold=threshold,
        group_col="gender",
    )
    fairness_age_band = build_group_fairness_table(
        governance_val,
        y_val,
        y_proba,
        threshold=threshold,
        group_col="age_band",
    )

    model_card_path = ARTIFACT_DIR / "us15_model_card.md"
    feature_types_path = ARTIFACT_DIR / "us15_feature_list_types.csv"
    fairness_gender_path = ARTIFACT_DIR / "us15_fairness_gender.csv"
    fairness_age_band_path = ARTIFACT_DIR / "us15_fairness_age_band.csv"

    feature_types.to_csv(feature_types_path, index=False)
    fairness_gender.to_csv(fairness_gender_path, index=False)
    fairness_age_band.to_csv(fairness_age_band_path, index=False)

    known_limitations = [
        "Fairness analysis here is descriptive and should not be interpreted as causal bias attribution.",
        "Age-band analysis depends on the quality and availability of `bd` values.",
        "Threshold-based classification changes both error tradeoffs and group-level predicted churn rates.",
        "Model performance can drift as customer behavior and subscription patterns evolve.",
    ]

    model_card_path.write_text(
        "\n".join(
            [
                "# Model Card - Production Churn Model",
                "",
                "## Training Data",
                "- Source table: `processed.customer_features`",
                f"- Rows used: {len(full_df):,}",
                "- Label: `is_churn`",
                f"- Data date range: {describe_training_date_range(full_df)}",
                "",
                "## Feature List and Types",
                feature_types.to_markdown(index=False),
                "",
                "## Validation Performance",
                f"- Production model: `{model_name}`",
                f"- Precision (churn=1): {metrics['precision_churn']:.4f}",
                f"- Recall (churn=1): {metrics['recall_churn']:.4f}",
                f"- F1 (churn=1): {metrics['f1_churn']:.4f}",
                f"- ROC AUC: {metrics['roc_auc']:.4f}",
                "",
                "## Known Limitations and Bias Considerations",
                *[f"- {item}" for item in known_limitations],
                "",
                "## Fairness Analysis: Churn Rate by Gender",
                fairness_gender.to_markdown(index=False),
                "",
                "## Fairness Analysis: Churn Rate by Age Band",
                fairness_age_band.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "artifact_paths": [
            model_card_path,
            feature_types_path,
            fairness_gender_path,
            fairness_age_band_path,
        ],
        "fairness_gender_gap": (
            float(fairness_gender["churn_rate"].max() - fairness_gender["churn_rate"].min())
            if not fairness_gender.empty
            else 0.0
        ),
        "fairness_age_band_gap": (
            float(fairness_age_band["churn_rate"].max() - fairness_age_band["churn_rate"].min())
            if not fairness_age_band.empty
            else 0.0
        ),
    }
