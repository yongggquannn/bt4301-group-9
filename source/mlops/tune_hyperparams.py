"""US-17 Hyperparameter Tuning with Optuna.

Run an Optuna study to find optimal XGBoost hyperparameters for the
KKBox churn model.  Each trial is logged as a nested MLflow run under
a single parent run.  Generates:
  - best hyperparameters JSON artifact
  - Optuna optimisation-history plot (PNG)
  - parameter-importance plot (PNG)
  - improvement summary (Markdown)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Re-use helpers from the Sprint-2 training script (same directory).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKING_URI = "http://localhost:5001"
sys.path.insert(0, str(_SCRIPT_DIR))

from train_model import (  # noqa: E402
    EXPERIMENT_NAME,
    IMBALANCE_CLASS_WEIGHT,
    _split_columns,
    apply_imbalance,
    build_preprocessor,
    compute_metrics,
    load_feature_store,
    load_imbalance_strategy,
    load_selected_features,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
ARTIFACT_DIR = _PROJECT_ROOT / "docs" / "artifacts"

# Sprint-2 baseline from docs/artifacts/us10_best_model.json
BASELINE_AUC = 0.9826
N_TRIALS_DEFAULT = 30


# ── Objective factory ─────────────────────────────────────────────────────


def create_objective(
    seed: int,
    test_size: float,
    threshold: float,
    sample_rows: int | None,
    experiment_name: str,
) -> Callable[[optuna.Trial], float]:
    """Return an objective closure that shares pre-processed data across trials."""
    selected_features = load_selected_features()
    strategy = load_imbalance_strategy()
    df = load_feature_store(selected_features, sample_rows, seed)

    y = df["is_churn"].astype(int)
    X = df[selected_features].copy()
    cat_cols, num_cols = _split_columns(selected_features)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    X_res, y_res, _ = apply_imbalance(X_train_t, y_train.values, strategy, seed)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1) if strategy == IMBALANCE_CLASS_WEIGHT else 1.0

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = XGBClassifier(
            **params,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=seed,
        )
        model.fit(X_res, y_res)

        y_proba = model.predict_proba(X_val_t)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        metrics = compute_metrics(y_val.values, y_pred, y_proba)

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(
                {**params, "scale_pos_weight": spw, "trial_number": trial.number},
            )
            mlflow.log_metrics(metrics)

        return metrics["roc_auc"]

    return objective


# ── Artifact helpers ──────────────────────────────────────────────────────


def save_best_params(study: optuna.Study) -> Path:
    """Write the best hyperparameters to a JSON artifact."""
    payload = {
        "best_trial_number": study.best_trial.number,
        "best_roc_auc": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "baseline_roc_auc": BASELINE_AUC,
        "improvement": study.best_value - BASELINE_AUC,
    }
    path = ARTIFACT_DIR / "us17_best_hyperparams.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Best params saved to %s", path)
    return path


def save_optimization_curve(study: optuna.Study) -> Path:
    """Save Optuna optimisation-history plot as PNG."""
    ax = plot_optimization_history(study)
    fig = ax.get_figure()
    if fig is None:
        raise RuntimeError("plot_optimization_history returned an Axes with no Figure")
    path = ARTIFACT_DIR / "us17_optimization_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Optimisation curve saved to %s", path)
    return path


def save_param_importance(study: optuna.Study) -> Path:
    """Save Optuna parameter-importance plot as PNG."""
    ax = plot_param_importances(study)
    fig = ax.get_figure()
    if fig is None:
        raise RuntimeError("plot_param_importances returned an Axes with no Figure")
    path = ARTIFACT_DIR / "us17_param_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Param importance saved to %s", path)
    return path


def save_improvement_summary(study: optuna.Study) -> Path:
    """Write a Markdown summary comparing baseline vs tuned AUC-ROC."""
    best = study.best_value
    delta = best - BASELINE_AUC
    direction = "improvement" if delta >= 0 else "regression"

    lines = [
        "# US-17 Hyperparameter Tuning Results\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Sprint 2 Baseline AUC-ROC | {BASELINE_AUC:.4f} |",
        f"| Tuned Best AUC-ROC | {best:.4f} |",
        f"| Delta ({direction}) | {delta:+.4f} |",
        f"| Trials Run | {len(study.trials)} |",
        "",
        "## Best Hyperparameters\n",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for k, v in sorted(study.best_params.items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    path = ARTIFACT_DIR / "us17_improvement_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Improvement summary saved to %s", path)
    return path


# ── CLI entry point ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="US-17: Hyperparameter Tuning with Optuna",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI (default: %(default)s)",
    )
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--sample-rows", default=None, type=int)
    parser.add_argument("--n-trials", default=N_TRIALS_DEFAULT, type=int)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="us17-xgboost-hpo",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="US17-Optuna-HPO") as parent_run:
        objective = create_objective(
            seed=args.seed,
            test_size=args.test_size,
            threshold=args.threshold,
            sample_rows=args.sample_rows,
            experiment_name=args.experiment_name,
        )

        study.optimize(objective, n_trials=args.n_trials)

        # Log best results on the parent run.
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics(
            {
                "best_roc_auc": study.best_value,
                "baseline_roc_auc": BASELINE_AUC,
                "auc_improvement": study.best_value - BASELINE_AUC,
            }
        )

        # Generate and log artifacts.
        artifacts = [
            save_best_params(study),
            save_optimization_curve(study),
            save_param_importance(study),
            save_improvement_summary(study),
        ]
        for artifact_path in artifacts:
            mlflow.log_artifact(str(artifact_path))

        logger.info(
            "Study complete — best AUC-ROC: %.4f (baseline %.4f, delta %+.4f)",
            study.best_value,
            BASELINE_AUC,
            study.best_value - BASELINE_AUC,
        )
        logger.info("Parent MLflow run ID: %s", parent_run.info.run_id)


if __name__ == "__main__":
    main()
