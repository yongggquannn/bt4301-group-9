"""US-12 MLflow Model Registry.

Register the best model from US-10 as ``KKBox-Churn-Classifier`` in the
MLflow Model Registry and promote it through lifecycle stages:
None → Staging → Production.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
ARTIFACT_DIR = _PROJECT_ROOT / "docs" / "artifacts"
BEST_MODEL_PATH = ARTIFACT_DIR / "us10_best_model.json"

MODEL_NAME = "KKBox-Churn-Classifier"
DEFAULT_TRACKING_URI = "http://localhost:5001"


@dataclass(frozen=True)
class StageTransition:
    """Record of a single stage transition."""

    from_stage: str
    to_stage: str
    timestamp: str


@dataclass(frozen=True)
class RegistryResult:
    """Evidence output for US-12 acceptance criteria."""

    model_name: str
    version: int
    run_id: str
    source_model: str
    roc_auc: float
    transitions: tuple[StageTransition, ...]
    final_stage: str
    registry_uri: str


def load_best_model_info(path: Path) -> dict:
    """Read the best model JSON produced by US-10 training."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run train_model.py (US-10) first."
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def register_best_model(
    client: MlflowClient,
    run_id: str,
    model_name: str,
) -> int:
    """Register the model and return its version number."""
    model_uri = f"runs:/{run_id}/model"
    logger.info("Registering model from %s as '%s'", model_uri, model_name)

    result = mlflow.register_model(model_uri, model_name)
    version = int(result.version)

    # Wait for the version to become READY (async backends may delay).
    mv = client.get_model_version(model_name, str(version))
    for _ in range(30):
        if mv.status == "READY":
            break
        time.sleep(1)
        mv = client.get_model_version(model_name, str(version))
    else:
        logger.warning(
            "Model version %d did not reach READY status after 30s (last: %s)",
            version,
            mv.status,
        )

    logger.info("Registered %s version %d (status: %s)", model_name, version, mv.status)
    return version


def transition_stage(
    client: MlflowClient,
    model_name: str,
    version: int,
    stage: str,
) -> StageTransition:
    """Transition a model version to the given stage and return a record."""
    mv_before = client.get_model_version(model_name, str(version))
    from_stage = mv_before.current_stage or "None"

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
    )

    ts = datetime.now(timezone.utc).isoformat()
    transition = StageTransition(
        from_stage=from_stage,
        to_stage=stage,
        timestamp=ts,
    )
    logger.info("Transitioned version %d: %s → %s", version, from_stage, stage)
    return transition


def save_evidence(result: RegistryResult, output_path: Path) -> None:
    """Write the evidence JSON for US-12 acceptance criteria."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": result.model_name,
        "version": result.version,
        "run_id": result.run_id,
        "source_model": result.source_model,
        "roc_auc": result.roc_auc,
        "transitions": [
            {
                "from_stage": t.from_stage,
                "to_stage": t.to_stage,
                "timestamp": t.timestamp,
            }
            for t in result.transitions
        ],
        "final_stage": result.final_stage,
        "registry_uri": result.registry_uri,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Evidence saved to %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="US-12: Register best model in MLflow Model Registry",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking server URI (default: %(default)s)",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Registry model name (default: %(default)s)",
    )
    parser.add_argument(
        "--best-model-json",
        default=str(BEST_MODEL_PATH),
        help="Path to best model JSON from US-10 (default: %(default)s)",
    )
    args = parser.parse_args()

    # Point MLflow at the tracking server (required for Model Registry).
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    # Load best model info from US-10.
    best_info = load_best_model_info(Path(args.best_model_json))
    run_id = best_info["run_id"]
    source_model = best_info["best_model"]
    roc_auc = best_info["metrics"]["roc_auc"]
    logger.info(
        "Best model: %s (run_id=%s, ROC AUC=%.4f)", source_model, run_id, roc_auc,
    )

    # Step 1: Register → creates version in "None" stage.
    version = register_best_model(client, run_id, args.model_name)

    # Step 2: None → Staging.
    t1 = transition_stage(client, args.model_name, version, "Staging")

    # Step 3: Staging → Production.
    t2 = transition_stage(client, args.model_name, version, "Production")

    registry_uri = f"models:/{args.model_name}/Production"

    result = RegistryResult(
        model_name=args.model_name,
        version=version,
        run_id=run_id,
        source_model=source_model,
        roc_auc=roc_auc,
        transitions=(t1, t2),
        final_stage="Production",
        registry_uri=registry_uri,
    )

    evidence_path = ARTIFACT_DIR / "us12_model_registry.json"
    save_evidence(result, evidence_path)

    print(
        f"\nModel '{args.model_name}' v{version} is now in Production.\n"
        f"Evidence: {evidence_path}\n"
        f"MLflow UI: {args.tracking_uri}/#/models/{args.model_name}\n"
        f"Registry URI for scoring: {registry_uri}"
    )


if __name__ == "__main__":
    main()
