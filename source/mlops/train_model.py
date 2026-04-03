"""US-10 Model Training and Comparison.

Train three model types (Logistic Regression, XGBoost, Neural Network)
under the MLflow experiment ``KKBox Churn``.  Each run logs hyper-parameters,
classification metrics, a confusion-matrix heatmap, and (for XGBoost) a
feature-importance bar chart.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt 
import mlflow 
import mlflow.sklearn 
import numpy as np 
import pandas as pd 
import psycopg2 
import seaborn as sns 
from imblearn.over_sampling import SMOTE 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import ( 
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.utils.class_weight import compute_sample_weight 
from xgboost import XGBClassifier 

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
ARTIFACT_DIR = _PROJECT_ROOT / "docs" / "artifacts"
FEATURE_SET_PATH = ARTIFACT_DIR / "final_feature_set.json"
IMBALANCE_STRATEGY_PATH = ARTIFACT_DIR / "us18_chosen_strategy.json"

EXPERIMENT_NAME = "KKBox Churn"

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "dbname": os.getenv("POSTGRES_DB", "kkbox"),
    "user": os.getenv("POSTGRES_USER", "bt4301"),
    "password": os.getenv("POSTGRES_PASSWORD", "bt4301pass"),
}

CATEGORICAL_FEATURES = frozenset(
    {
        "gender",
        "city",
        "registered_via",
        "latest_payment_method_id",
        "latest_is_auto_renew",
    }
)

# Allowlist of valid column names for SQL queries (prevents injection).
_KNOWN_COLUMNS = frozenset(
    {
        "msno",
        "is_churn",
        "feature_created_at",
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

MODEL_SLUG = {
    "Logistic Regression": "logistic_regression",
    "XGBoost": "xgboost",
    "Neural Network (MLP)": "mlp",
}

IMBALANCE_SMOTE = "smote"
IMBALANCE_CLASS_WEIGHT = "class_weight_balanced"


@dataclass(frozen=True)
class ModelConfig:
    """Declarative specification for a single model run."""

    name: str
    model_factory: Callable[[], Any]
    hyperparams: dict[str, Any]
    has_feature_importance: bool
    needs_sample_weight: bool


@dataclass(frozen=True)
class RunResult:
    """Metrics and artifact paths produced by a single training run."""

    model_name: str
    precision_churn: float
    recall_churn: float
    f1_churn: float
    roc_auc: float
    run_id: str
    confusion_matrix_path: Path
    feature_importance_path: Path | None
    y_proba: np.ndarray
    y_val: np.ndarray


@dataclass(frozen=True)
class TrainOutput:
    """All outputs from a single train-and-evaluate cycle."""

    metrics: dict[str, float]
    model: Any
    feature_names: list[str]
    y_pred: np.ndarray
    y_proba: np.ndarray
    y_val: np.ndarray


def _validate_columns(columns: list[str]) -> None:
    """Reject column names not in the known allowlist."""
    unknown = set(columns) - _KNOWN_COLUMNS
    if unknown:
        raise ValueError(f"Unknown column names rejected: {unknown}")


def load_selected_features() -> list[str]:
    """Read the feature list produced by US-11 feature selection."""
    with open(FEATURE_SET_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["selected_features"]


def load_imbalance_strategy() -> str:
    """Read the chosen class-imbalance strategy from US-18.

    Falls back to ``class_weight_balanced`` when the artifact is missing.
    """
    if not IMBALANCE_STRATEGY_PATH.exists():
        logger.warning(
            "%s not found — falling back to %s",
            IMBALANCE_STRATEGY_PATH,
            IMBALANCE_CLASS_WEIGHT,
        )
        return IMBALANCE_CLASS_WEIGHT

    with open(IMBALANCE_STRATEGY_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("chosen_strategy", IMBALANCE_CLASS_WEIGHT)


def load_feature_store(
    selected_features: list[str],
    sample_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    """Load only the selected features + target from the DB."""
    governance_cols = [c for c in ["gender", "bd", "feature_created_at"] if c not in selected_features]
    all_cols = selected_features + governance_cols + ["is_churn"]
    _validate_columns(all_cols)
    columns = ", ".join(all_cols)
    sql = f"SELECT {columns} FROM processed.customer_features"

    with psycopg2.connect(**DB_CONFIG) as conn:
        df = pd.read_sql(sql, conn)

    if sample_rows is not None:
        if sample_rows <= 0:
            raise ValueError("sample_rows must be > 0 or None")
        frac = min(1.0, sample_rows / len(df))
        df = df.sample(frac=frac, random_state=seed, replace=False)

    return df


def _split_columns(
    features: list[str],
) -> tuple[list[str], list[str]]:
    """Split selected features into categorical / numeric lists."""
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    num = [f for f in features if f not in CATEGORICAL_FEATURES]
    return cat, num


def build_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
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


def apply_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply the chosen imbalance strategy to transformed training data.

    Returns ``(X_resampled, y_resampled, sample_weights)``.
    *sample_weights* is ``None`` when SMOTE is used (resampled data already
    balanced) and populated when the strategy is ``class_weight_balanced``
    (for models that need explicit sample weights, e.g. MLP).
    """
    if strategy == IMBALANCE_SMOTE:
        smote = SMOTE(
            sampling_strategy=0.5,
            random_state=seed,
            k_neighbors=5,
        )
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res, None

    weights = compute_sample_weight("balanced", y)
    return X, y, weights


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average="binary", zero_division=0,
    )
    roc_auc = roc_auc_score(y_true, y_proba)
    return {
        "precision_churn": float(precision),
        "recall_churn": float(recall),
        "f1_churn": float(f1),
        "roc_auc": float(roc_auc),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> Path:
    slug = MODEL_SLUG.get(model_name, model_name.lower().replace(" ", "_"))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    path = ARTIFACT_DIR / f"us10_confusion_matrix_{slug}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
) -> Path:
    slug = MODEL_SLUG.get(model_name, model_name.lower().replace(" ", "_"))
    importances = model.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, max(6, len(feature_names) * 0.35)))
    ax.barh(np.array(feature_names)[idx], importances[idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    path = ARTIFACT_DIR / f"us10_feature_importance_{slug}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def train_and_evaluate(
    config: ModelConfig,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    strategy: str,
    seed: int,
    threshold: float,
) -> TrainOutput:
    """Preprocess, handle imbalance, fit, and evaluate a model."""
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    feature_names_out = list(preprocessor.get_feature_names_out())

    X_res, y_res, sample_weights = apply_imbalance(
        X_train_t, y_train.values, strategy, seed,
    )

    model = config.model_factory()
    if config.needs_sample_weight and sample_weights is not None:
        model.fit(X_res, y_res, sample_weight=sample_weights)
    else:
        model.fit(X_res, y_res)

    y_proba = model.predict_proba(X_val_t)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = compute_metrics(y_val.values, y_pred, y_proba)
    return TrainOutput(
        metrics=metrics,
        model=model,
        feature_names=feature_names_out,
        y_pred=y_pred,
        y_proba=y_proba,
        y_val=y_val.values,
    )


def log_mlflow_run(
    experiment_name: str,
    model_name: str,
    model: Any,
    hyperparams: dict[str, Any],
    metrics: dict[str, float],
    artifact_paths: list[Path],
    strategy: str,
    seed: int,
    test_size: float,
) -> str:
    """Create one MLflow run and return its run_id."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(
            {
                **hyperparams,
                "model_type": model_name,
                "imbalance_strategy": strategy,
                "seed": seed,
                "test_size": test_size,
            }
        )
        mlflow.log_metrics(metrics)

        for path in artifact_paths:
            mlflow.log_artifact(str(path))

        mlflow.sklearn.log_model(model, artifact_path="model")

        return run.info.run_id


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


def build_model_configs(
    strategy: str,
    seed: int,
    pos_weight: float,
) -> list[ModelConfig]:
    cw = "balanced" if strategy == IMBALANCE_CLASS_WEIGHT else None
    xgb_spw = pos_weight if strategy == IMBALANCE_CLASS_WEIGHT else 1.0

    return [
        ModelConfig(
            name="Logistic Regression",
            model_factory=partial(
                LogisticRegression,
                C=1.0,
                max_iter=2000,
                solver="lbfgs",
                class_weight=cw,
                random_state=seed,
            ),
            hyperparams={
                "C": 1.0,
                "max_iter": 2000,
                "solver": "lbfgs",
                "class_weight": str(cw),
            },
            has_feature_importance=False,
            needs_sample_weight=False,
        ),
        ModelConfig(
            name="XGBoost",
            model_factory=partial(
                XGBClassifier,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=xgb_spw,
                eval_metric="logloss",
                random_state=seed,
            ),
            hyperparams={
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": xgb_spw,
            },
            has_feature_importance=True,
            needs_sample_weight=False,
        ),
        ModelConfig(
            name="Neural Network (MLP)",
            model_factory=partial(
                MLPClassifier,
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            ),
            hyperparams={
                "hidden_layer_sizes": "(128, 64, 32)",
                "activation": "relu",
                "solver": "adam",
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.1,
            },
            has_feature_importance=False,
            needs_sample_weight=(strategy == IMBALANCE_CLASS_WEIGHT),
        ),
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="US-10: Train & compare churn models",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI (default: %(default)s)",
    )
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--sample-rows", default=None, type=int)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "xgboost", "mlp"],
        help="Subset of models to train (for dev iteration)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    selected_features = load_selected_features()
    strategy = load_imbalance_strategy()
    logger.info(
        "Selected features (%d): %s", len(selected_features), selected_features,
    )
    logger.info("Imbalance strategy: %s", strategy)

    df = load_feature_store(selected_features, args.sample_rows, args.seed)
    if len(df) < 100:
        raise RuntimeError(
            f"Only {len(df)} rows loaded — check DB connection and data."
        )

    y = df["is_churn"].astype(int)
    X = df[selected_features].copy()
    governance_cols = [c for c in ["gender", "bd"] if c in df.columns]
    governance_df = df[governance_cols].copy()

    cat_cols, num_cols = _split_columns(selected_features)
    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in num_cols if c in X.columns]

    X_train, X_val, _governance_train, governance_val, y_train, y_val = train_test_split(
        X, governance_df, y, test_size=args.test_size, random_state=args.seed, stratify=y,
    )

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    pos_weight = neg_count / max(pos_count, 1)

    all_configs = build_model_configs(strategy, args.seed, pos_weight)

    slug_to_config = {MODEL_SLUG[c.name]: c for c in all_configs}
    configs = [slug_to_config[m] for m in args.models if m in slug_to_config]
    if not configs:
        raise ValueError(f"No matching models for --models {args.models}")

    results: list[RunResult] = []

    for config in configs:
        logger.info("Training: %s", config.name)

        preprocessor = build_preprocessor(cat_cols, num_cols)

        output = train_and_evaluate(
            config=config,
            preprocessor=preprocessor,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            strategy=strategy,
            seed=args.seed,
            threshold=args.threshold,
        )

        cm_path = plot_confusion_matrix(
            output.y_val, output.y_pred, config.name,
        )
        artifact_paths: list[Path] = [cm_path]

        fi_path: Path | None = None
        if config.has_feature_importance:
            fi_path = plot_feature_importance(
                output.model, output.feature_names, config.name,
            )
            artifact_paths.append(fi_path)

        run_id = log_mlflow_run(
            experiment_name=args.experiment_name,
            model_name=config.name,
            model=output.model,
            hyperparams=config.hyperparams,
            metrics=output.metrics,
            artifact_paths=artifact_paths,
            strategy=strategy,
            seed=args.seed,
            test_size=args.test_size,
        )

        results.append(
            RunResult(
                model_name=config.name,
                precision_churn=output.metrics["precision_churn"],
                recall_churn=output.metrics["recall_churn"],
                f1_churn=output.metrics["f1_churn"],
                roc_auc=output.metrics["roc_auc"],
                run_id=run_id,
                confusion_matrix_path=cm_path,
                feature_importance_path=fi_path,
                y_proba=output.y_proba,
                y_val=output.y_val,
            )
        )

        logger.info(
            "  %s  AUC-ROC=%.4f  F1=%.4f  run_id=%s",
            config.name,
            output.metrics["roc_auc"],
            output.metrics["f1_churn"],
            run_id,
        )

    comparison = pd.DataFrame(
        [
            {
                "model": r.model_name,
                "precision_churn": r.precision_churn,
                "recall_churn": r.recall_churn,
                "f1_churn": r.f1_churn,
                "roc_auc": r.roc_auc,
                "run_id": r.run_id,
            }
            for r in results
        ]
    ).sort_values("roc_auc", ascending=False, kind="mergesort")

    best = max(results, key=lambda r: r.roc_auc)

    comparison_csv = ARTIFACT_DIR / "us10_model_comparison.csv"
    comparison_md = ARTIFACT_DIR / "us10_model_comparison.md"
    best_json = ARTIFACT_DIR / "us10_best_model.json"

    comparison.to_csv(comparison_csv, index=False)
    comparison_md.write_text(
        comparison.to_markdown(index=False), encoding="utf-8",
    )
    best_json.write_text(
        json.dumps(
            {
                "best_model": best.model_name,
                "run_id": best.run_id,
                "metrics": {
                    "precision_churn": best.precision_churn,
                    "recall_churn": best.recall_churn,
                    "f1_churn": best.f1_churn,
                    "roc_auc": best.roc_auc,
                },
                "imbalance_strategy": strategy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    governance = write_us15_governance_artifacts(
        model_name=best.model_name,
        selected_features=selected_features,
        full_df=df,
        feature_frame=X,
        governance_val=governance_val,
        y_val=best.y_val,
        y_proba=best.y_proba,
        metrics={
            "precision_churn": best.precision_churn,
            "recall_churn": best.recall_churn,
            "f1_churn": best.f1_churn,
            "roc_auc": best.roc_auc,
        },
        threshold=args.threshold,
    )

    with mlflow.start_run(run_id=best.run_id):
        mlflow.set_tags(
            {
                "us15_model_governance": "true",
                "production_candidate": "true",
            }
        )
        mlflow.log_metrics(
            {
                "fairness_gender_churn_rate_gap": governance["fairness_gender_gap"],
                "fairness_age_band_churn_rate_gap": governance["fairness_age_band_gap"],
            }
        )
        for path in governance["artifact_paths"]:
            mlflow.log_artifact(str(path))

    logger.info("Model Comparison (sorted by AUC-ROC):\n%s", comparison.to_string(index=False))
    logger.info("Best model: %s (AUC-ROC: %.4f)", best.model_name, best.roc_auc)
    logger.info("Artifacts saved to: %s", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
