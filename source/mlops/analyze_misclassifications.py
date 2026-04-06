"""US-22 Misclassification analysis on validation predictions.

This script trains a single churn model, evaluates on validation data, and
generates artifacts comparing misclassified vs correctly classified samples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
ARTIFACT_DIR = PROJECT_ROOT / "docs" / "artifacts"
FEATURE_SET_PATH = ARTIFACT_DIR / "final_feature_set.json"
IMBALANCE_STRATEGY_PATH = ARTIFACT_DIR / "us18_chosen_strategy.json"

from source.common.db import get_db_config
from train_model import (
    CATEGORICAL_FEATURES,
    _split_columns,
    build_preprocessor as _build_preprocessor,
    load_imbalance_strategy,
    load_selected_features,
)

DB_CONFIG = get_db_config()

MODEL_CHOICES = ("logistic_regression", "xgboost", "mlp")


def load_feature_store(selected_features: list[str], sample_rows: int | None, seed: int) -> pd.DataFrame:
    cols = selected_features + ["is_churn"]
    sql = f"SELECT {', '.join(cols)} FROM processed.customer_features"
    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(sql, conn)

    if sample_rows is not None:
        if sample_rows <= 0:
            raise ValueError("sample_rows must be > 0 or None")
        frac = min(1.0, sample_rows / len(df))
        df = df.sample(frac=frac, random_state=seed, replace=False)
    return df


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    cat_cols, num_cols = _split_columns(features)
    return _build_preprocessor(cat_cols, num_cols)


def make_model(model_name: str, seed: int, pos_weight: float, strategy: str) -> tuple[Any, bool]:
    class_weight = "balanced" if strategy == "class_weight_balanced" else None
    if model_name == "logistic_regression":
        return (
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                class_weight=class_weight,
                random_state=seed,
            ),
            False,
        )
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError(
                "XGBoost is unavailable in this environment. "
                "Install OpenMP (`brew install libomp`) and reinstall xgboost, "
                "or run with `--model logistic_regression` / `--model mlp`."
            )
        return (
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight if strategy == "class_weight_balanced" else 1.0,
                eval_metric="logloss",
                random_state=seed,
            ),
            False,
        )
    return (
        MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        ),
        strategy == "class_weight_balanced",
    )


def fit_and_predict(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    preprocessor: ColumnTransformer,
    strategy: str,
    seed: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    if strategy == "smote":
        smote = SMOTE(sampling_strategy=0.5, random_state=seed, k_neighbors=5)
        X_fit, y_fit = smote.fit_resample(X_train_t, y_train.values)
        sample_weights = None
    else:
        X_fit, y_fit = X_train_t, y_train.values
        sample_weights = compute_sample_weight("balanced", y_fit)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    pos_weight = neg_count / max(pos_count, 1)
    model, needs_sample_weight = make_model(model_name, seed, pos_weight, strategy)

    if needs_sample_weight and sample_weights is not None:
        model.fit(X_fit, y_fit, sample_weight=sample_weights)
    else:
        model.fit(X_fit, y_fit)

    y_prob = model.predict_proba(X_val_t)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def summarize_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def build_case_frame(
    X_val: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    df = X_val.copy()
    df["actual"] = y_true
    df["predicted"] = y_pred
    df["predicted_probability"] = y_prob
    df["is_misclassified"] = (df["actual"] != df["predicted"]).astype(int)
    df["error_type"] = "correct"
    df.loc[(df["actual"] == 1) & (df["predicted"] == 0), "error_type"] = "FN"
    df.loc[(df["actual"] == 0) & (df["predicted"] == 1), "error_type"] = "FP"
    return df


def compare_numeric_distribution(df: pd.DataFrame, feature: str) -> dict[str, Any]:
    mis = pd.to_numeric(df.loc[df["is_misclassified"] == 1, feature], errors="coerce").dropna()
    cor = pd.to_numeric(df.loc[df["is_misclassified"] == 0, feature], errors="coerce").dropna()
    if len(mis) == 0 or len(cor) == 0:
        return {}

    mis_mean = float(mis.mean())
    cor_mean = float(cor.mean())
    pooled = float(np.sqrt((mis.var(ddof=1) + cor.var(ddof=1)) / 2)) if (len(mis) > 1 and len(cor) > 1) else 0.0
    std_diff = (mis_mean - cor_mean) / pooled if pooled > 0 else 0.0

    return {
        "feature": feature,
        "type": "numeric",
        "misclassified_mean": mis_mean,
        "correct_mean": cor_mean,
        "misclassified_median": float(mis.median()),
        "correct_median": float(cor.median()),
        "std_mean_diff": float(std_diff),
    }


def compare_categorical_distribution(df: pd.DataFrame, feature: str) -> dict[str, Any]:
    mis = df.loc[df["is_misclassified"] == 1, feature].astype(str)
    cor = df.loc[df["is_misclassified"] == 0, feature].astype(str)
    if len(mis) == 0 or len(cor) == 0:
        return {}

    mis_share = mis.value_counts(normalize=True)
    cor_share = cor.value_counts(normalize=True)
    cats = sorted(set(mis_share.index).union(cor_share.index))

    rows: list[dict[str, Any]] = []
    for cat in cats:
        m = float(mis_share.get(cat, 0.0))
        c = float(cor_share.get(cat, 0.0))
        rows.append(
            {
                "feature": feature,
                "type": "categorical",
                "category": cat,
                "misclassified_share": m,
                "correct_share": c,
                "share_diff": m - c,
                "lift": (m / c) if c > 0 else np.nan,
            }
        )
    best = max(rows, key=lambda r: abs(r["share_diff"]))
    return best


def create_distribution_plots(df: pd.DataFrame, numeric_features: list[str], output_path: Path) -> None:
    plot_features = numeric_features[:4]
    if not plot_features:
        return

    nrows = len(plot_features)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 3.2 * nrows))
    if nrows == 1:
        axes = [axes]

    for ax, feature in zip(axes, plot_features):
        sns.histplot(
            data=df,
            x=feature,
            hue="is_misclassified",
            bins=30,
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(f"{feature}: misclassified (1) vs correct (0)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def derive_actionable_insights(
    numeric_rows: list[dict[str, Any]],
    categorical_rows: list[dict[str, Any]],
    confusion: dict[str, int],
) -> list[str]:
    insights: list[str] = []

    top_numeric = sorted(numeric_rows, key=lambda r: abs(r["std_mean_diff"]), reverse=True)[:2]
    for row in top_numeric:
        direction = "higher" if row["misclassified_mean"] > row["correct_mean"] else "lower"
        insights.append(
            f"`{row['feature']}` is materially {direction} among misclassified cases "
            f"(misclassified mean={row['misclassified_mean']:.2f}, correct mean={row['correct_mean']:.2f}). "
            f"Action: add bucketized/interaction features for `{row['feature']}` and re-tune threshold."
        )

    if categorical_rows:
        top_cat = max(categorical_rows, key=lambda r: abs(r["share_diff"]))
        insights.append(
            f"Category `{top_cat['feature']}={top_cat['category']}` is over-represented in misclassifications "
            f"(misclassified share={top_cat['misclassified_share']:.1%}, correct share={top_cat['correct_share']:.1%}). "
            f"Action: introduce category-target interactions or category-specific calibration."
        )

    if confusion["fn"] > confusion["fp"]:
        insights.append(
            "False negatives exceed false positives. Action: lower decision threshold to catch more true churners "
            "and monitor precision trade-off."
        )
    else:
        insights.append(
            "False positives are at least as frequent as false negatives. Action: increase decision threshold or "
            "add precision-oriented features to reduce false alarms."
        )

    return insights[: max(2, len(insights))]


def write_report(
    output_path: Path,
    model_name: str,
    threshold: float,
    confusion: dict[str, int],
    numeric_path: Path,
    categorical_path: Path,
    insights: list[str],
    plot_path: Path,
) -> None:
    lines = [
        "# US-22 Misclassification Analysis",
        "",
        "## Validation confusion matrix breakdown",
        "",
        f"- Model: `{model_name}`",
        f"- Decision threshold: `{threshold}`",
        f"- TN: {confusion['tn']}, FP: {confusion['fp']}, FN: {confusion['fn']}, TP: {confusion['tp']}",
        f"- Missed churn (FN): **{confusion['fn']}**",
        f"- False alarm (FP): **{confusion['fp']}**",
        "",
        "## Feature distribution comparison (misclassified vs correctly classified)",
        "",
        f"- Numeric summary: `docs/artifacts/{numeric_path.name}`",
        f"- Categorical summary: `docs/artifacts/{categorical_path.name}`",
        f"- Plot: `docs/artifacts/{plot_path.name}`",
        "",
        "## Actionable insights",
        "",
    ]

    for idx, insight in enumerate(insights, start=1):
        lines.append(f"{idx}. {insight}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="US-22 misclassification analysis")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="xgboost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sample-rows", type=int, default=None)
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    selected_features = load_selected_features()
    strategy = load_imbalance_strategy()
    df = load_feature_store(selected_features, args.sample_rows, args.seed)
    if len(df) < 100:
        raise RuntimeError(f"Only {len(df)} rows loaded; check DB data/setup.")

    y = df["is_churn"].astype(int)
    X = df.drop(columns=["is_churn"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    preprocessor = build_preprocessor(selected_features)
    y_pred, y_prob = fit_and_predict(
        model_name=args.model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        preprocessor=preprocessor,
        strategy=strategy,
        seed=args.seed,
        threshold=args.threshold,
    )

    confusion = summarize_confusion(y_val.values, y_pred)
    case_df = build_case_frame(X_val, y_val.values, y_pred, y_prob)

    numeric_features = [f for f in selected_features if f not in CATEGORICAL_FEATURES]
    categorical_features = [f for f in selected_features if f in CATEGORICAL_FEATURES]

    numeric_rows = [compare_numeric_distribution(case_df, f) for f in numeric_features]
    numeric_rows = [r for r in numeric_rows if r]
    categorical_rows = [compare_categorical_distribution(case_df, f) for f in categorical_features]
    categorical_rows = [r for r in categorical_rows if r]

    numeric_df = pd.DataFrame(numeric_rows)
    if not numeric_df.empty:
        numeric_df = numeric_df.sort_values("std_mean_diff", key=np.abs, ascending=False)
    categorical_df = pd.DataFrame(categorical_rows)
    if not categorical_df.empty:
        categorical_df = categorical_df.sort_values("share_diff", key=np.abs, ascending=False)

    insights = derive_actionable_insights(numeric_rows, categorical_rows, confusion)

    misclassified_cases_path = ARTIFACT_DIR / "us22_misclassified_cases.csv"
    numeric_summary_path = ARTIFACT_DIR / "us22_numeric_distribution_comparison.csv"
    categorical_summary_path = ARTIFACT_DIR / "us22_categorical_distribution_comparison.csv"
    confusion_json_path = ARTIFACT_DIR / "us22_confusion_breakdown.json"
    plot_path = ARTIFACT_DIR / "us22_feature_distribution_plot.png"
    report_path = PROJECT_ROOT / "docs" / "us22_misclassification_analysis.md"

    case_df.to_csv(misclassified_cases_path, index=False)
    numeric_df.to_csv(numeric_summary_path, index=False)
    categorical_df.to_csv(categorical_summary_path, index=False)
    confusion_json_path.write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    create_distribution_plots(case_df, numeric_features, plot_path)

    write_report(
        output_path=report_path,
        model_name=args.model,
        threshold=args.threshold,
        confusion=confusion,
        numeric_path=numeric_summary_path,
        categorical_path=categorical_summary_path,
        insights=insights,
        plot_path=plot_path,
    )

    print("US-22 analysis complete.")
    print(f"Report: {report_path}")
    print(f"FN={confusion['fn']} FP={confusion['fp']}")


if __name__ == "__main__":
    main()
