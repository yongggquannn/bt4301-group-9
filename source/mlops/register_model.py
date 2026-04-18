"""US-12 / US-20 Champion-Challenger MLflow Model Registry.

Register the best model from US-10 as ``KKBox-Churn-Classifier`` and apply a
champion-challenger promotion rule:

- if no current Production champion exists, promote the new version
- if ``new_model_auc > champion_auc + threshold``, promote the challenger
- otherwise keep the existing champion in Production and leave the new version
  visible in the registry as a challenger
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from source.mlops.github_dispatch import (
    DEFAULT_EVENT_TYPE,
    DEFAULT_GITHUB_API_URL,
    build_model_promoted_payload,
    emit_repository_dispatch,
)

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
ARTIFACT_DIR = _PROJECT_ROOT / "docs" / "artifacts"
BEST_MODEL_PATH = ARTIFACT_DIR / "best_model.json"

MODEL_NAME = "KKBox-Churn-Classifier"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
DEFAULT_PROMOTION_THRESHOLD = 0.0


@dataclass(frozen=True)
class StageTransition:
    """Record of a single stage transition."""

    from_stage: str
    to_stage: str
    timestamp: str


@dataclass(frozen=True)
class RegistryResult:
    """Evidence output for registry + champion-challenger acceptance criteria."""

    model_name: str
    version: int
    run_id: str
    source_model: str
    challenger_auc: float
    champion_version: int | None
    champion_run_id: str | None
    champion_auc: float | None
    promotion_threshold: float
    auc_margin: float | None
    decision: str
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


def validate_model_artifact(best_info: dict, model_uri: str) -> None:
    """Ensure the model artifact is usable for production scoring.

    Validates from the best_model.json handoff artifact written by train_model.py,
    avoiding an expensive artifact download just to read signature metadata.
    """
    if best_info.get("model_artifact_type") != "serving_pipeline":
        raise RuntimeError(
            f"Model artifact {model_uri} is not a serving_pipeline. "
            "Re-run train_model.py so the serving pipeline is logged correctly before registration."
        )

    input_features = best_info.get("input_features", [])
    if not input_features:
        raise RuntimeError(
            f"best_model.json has no input_features for {model_uri}. "
            "Re-run train_model.py so the serving pipeline is logged from a DataFrame input."
        )


def get_model_roc_auc(client: MlflowClient, run_id: str) -> float:
    """Read ROC AUC from the source MLflow run."""
    run = client.get_run(run_id)
    try:
        return float(run.data.metrics["roc_auc"])
    except KeyError as exc:
        raise KeyError(f"Run {run_id} is missing metric 'roc_auc'") from exc


def get_current_production_version(
    client: MlflowClient,
    model_name: str,
):
    """Return the most recent Production model version, if any."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except RestException:
        return None

    prod_versions = [mv for mv in versions if (mv.current_stage or "None") == "Production"]
    if not prod_versions:
        return None

    prod_versions.sort(key=lambda mv: int(mv.version), reverse=True)
    return prod_versions[0]


def transition_stage(
    client: MlflowClient,
    model_name: str,
    version: int,
    stage: str,
    *,
    archive_existing_versions: bool = False,
) -> StageTransition:
    """Transition a model version to the given stage and return a record."""
    mv_before = client.get_model_version(model_name, str(version))
    from_stage = mv_before.current_stage or "None"

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )

    ts = datetime.now(timezone.utc).isoformat()
    transition = StageTransition(
        from_stage=from_stage,
        to_stage=stage,
        timestamp=ts,
    )
    logger.info("Transitioned version %d: %s -> %s", version, from_stage, stage)
    return transition


def set_version_tags(
    client: MlflowClient,
    model_name: str,
    version: int,
    tags: dict[str, str],
) -> None:
    """Attach audit tags to a model version."""
    for key, value in tags.items():
        client.set_model_version_tag(model_name, str(version), key, value)


def write_version_description(
    client: MlflowClient,
    model_name: str,
    version: int,
    lines: list[str],
) -> None:
    """Write a human-readable audit note onto the model version."""
    client.update_model_version(
        name=model_name,
        version=str(version),
        description="\n".join(lines),
    )


def save_evidence(result: RegistryResult, output_path: Path) -> None:
    """Write the evidence JSON for registry + champion-challenger criteria."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": result.model_name,
        "version": result.version,
        "run_id": result.run_id,
        "source_model": result.source_model,
        "roc_auc": result.challenger_auc,
        "challenger_auc": result.challenger_auc,
        "champion_version": result.champion_version,
        "champion_run_id": result.champion_run_id,
        "champion_auc": result.champion_auc,
        "promotion_threshold": result.promotion_threshold,
        "auc_margin": result.auc_margin,
        "decision": result.decision,
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


def maybe_emit_model_promoted_event(result: RegistryResult) -> None:
    repository = os.getenv("GITHUB_DISPATCH_REPO")
    token = os.getenv("GITHUB_DISPATCH_TOKEN")
    if not repository or not token:
        logger.info(
            "Skipping GitHub repository_dispatch for model promotion because "
            "GITHUB_DISPATCH_REPO or GITHUB_DISPATCH_TOKEN is not set."
        )
        return

    payload = build_model_promoted_payload(
        model_name=result.model_name,
        version=result.version,
        run_id=result.run_id,
        decision=result.decision,
        final_stage=result.final_stage,
        registry_uri=result.registry_uri,
        challenger_auc=result.challenger_auc,
        champion_version=result.champion_version,
        champion_auc=result.champion_auc,
        promotion_threshold=result.promotion_threshold,
        auc_margin=result.auc_margin,
    )

    emit_repository_dispatch(
        repository=repository,
        token=token,
        client_payload=payload,
        event_type=os.getenv("GITHUB_DISPATCH_EVENT_TYPE", DEFAULT_EVENT_TYPE),
        github_api_url=os.getenv("GITHUB_API_URL", DEFAULT_GITHUB_API_URL),
    )
    logger.info(
        "Sent GitHub repository_dispatch '%s' for %s version %d.",
        os.getenv("GITHUB_DISPATCH_EVENT_TYPE", DEFAULT_EVENT_TYPE),
        result.model_name,
        result.version,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Register best model in MLflow Model Registry with champion-challenger gating",
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
    parser.add_argument(
        "--promotion-threshold",
        default=DEFAULT_PROMOTION_THRESHOLD,
        type=float,
        help="Required AUC improvement over the current champion before promotion",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    best_info = load_best_model_info(Path(args.best_model_json))
    run_id = best_info["run_id"]
    source_model = best_info["best_model"]
    challenger_auc = float(best_info["metrics"]["roc_auc"])
    model_uri = f"runs:/{run_id}/model"
    validate_model_artifact(best_info, model_uri)
    logger.info(
        "Best model: %s (run_id=%s, ROC AUC=%.4f)", source_model, run_id, challenger_auc,
    )

    champion = get_current_production_version(client, args.model_name)
    champion_version = int(champion.version) if champion is not None else None
    champion_run_id = champion.run_id if champion is not None else None
    champion_auc = (
        get_model_roc_auc(client, champion.run_id)
        if champion is not None and champion.run_id
        else None
    )

    if champion is None:
        logger.info("No current Production champion found. New version will become the initial champion.")
    else:
        logger.info(
            "Current champion: version=%s run_id=%s ROC AUC=%.4f",
            champion.version,
            champion.run_id,
            champion_auc,
        )

    version = register_best_model(client, run_id, args.model_name)

    transitions: list[StageTransition] = []
    transitions.append(transition_stage(client, args.model_name, version, "Staging"))

    if champion_auc is None:
        promote = True
        decision = "promoted_initial_champion"
        auc_margin = None
    else:
        auc_margin = challenger_auc - champion_auc
        promote = challenger_auc > champion_auc + args.promotion_threshold
        decision = "promoted_over_champion" if promote else "kept_existing_champion"

    timestamp = datetime.now(timezone.utc).isoformat()
    comparison_note = [
        f"Decision timestamp: {timestamp}",
        f"Decision: {decision}",
        f"Challenger run_id: {run_id}",
        f"Challenger ROC AUC: {challenger_auc:.6f}",
        f"Promotion threshold: {args.promotion_threshold:.6f}",
        (
            f"Champion version: {champion_version}, run_id: {champion_run_id}, ROC AUC: {champion_auc:.6f}"
            if champion_auc is not None
            else "Champion version: none"
        ),
        (
            f"AUC margin (challenger - champion): {auc_margin:.6f}"
            if auc_margin is not None
            else "AUC margin (challenger - champion): not applicable"
        ),
    ]

    if promote:
        transitions.append(
            transition_stage(
                client,
                args.model_name,
                version,
                "Production",
                archive_existing_versions=True,
            )
        )
        final_stage = "Production"
    else:
        final_stage = "Staging"

    set_version_tags(
        client,
        args.model_name,
        version,
        {
            "selection_role": "champion" if promote else "challenger",
            "promotion_decision": decision,
            "challenger_auc": f"{challenger_auc:.6f}",
            "promotion_threshold": f"{args.promotion_threshold:.6f}",
            "champion_version": str(champion_version) if champion_version is not None else "none",
            "champion_auc": f"{champion_auc:.6f}" if champion_auc is not None else "none",
            "auc_margin": f"{auc_margin:.6f}" if auc_margin is not None else "none",
            "decision_timestamp": timestamp,
        },
    )
    write_version_description(client, args.model_name, version, comparison_note)

    if champion_version is not None:
        set_version_tags(
            client,
            args.model_name,
            champion_version,
            {
                "selection_role": "former_champion" if promote else "champion",
                "last_compared_against_version": str(version),
                "last_challenger_auc": f"{challenger_auc:.6f}",
                "promotion_threshold": f"{args.promotion_threshold:.6f}",
                "promotion_decision": decision,
                "decision_timestamp": timestamp,
            },
        )

    registry_uri = f"models:/{args.model_name}/Production"

    result = RegistryResult(
        model_name=args.model_name,
        version=version,
        run_id=run_id,
        source_model=source_model,
        challenger_auc=challenger_auc,
        champion_version=champion_version,
        champion_run_id=champion_run_id,
        champion_auc=champion_auc,
        promotion_threshold=args.promotion_threshold,
        auc_margin=auc_margin,
        decision=decision,
        transitions=tuple(transitions),
        final_stage=final_stage,
        registry_uri=registry_uri,
    )

    registry_path = ARTIFACT_DIR / "model_registry.json"
    champion_challenger_path = ARTIFACT_DIR / "champion_challenger_registry.json"
    save_evidence(result, registry_path)
    save_evidence(result, champion_challenger_path)

    if promote:
        maybe_emit_model_promoted_event(result)

    print(
        f"\nModel '{args.model_name}' v{version} decision: {decision}.\n"
        f"Final stage: {final_stage}\n"
        f"Registry evidence: {registry_path}\n"
        f"Champion-challenger evidence: {champion_challenger_path}\n"
        f"MLflow UI: {args.tracking_uri}/#/models/{args.model_name}\n"
        f"Registry URI for scoring: {registry_uri}"
    )


if __name__ == "__main__":
    main()
