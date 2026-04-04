from __future__ import annotations

import json
from typing import Any
from urllib import request


DEFAULT_EVENT_TYPE = "mlflow-model-promoted"
DEFAULT_GITHUB_API_URL = "https://api.github.com"


def build_model_promoted_payload(
    *,
    model_name: str,
    version: int,
    run_id: str,
    decision: str,
    final_stage: str,
    registry_uri: str,
    challenger_auc: float,
    champion_version: int | None,
    champion_auc: float | None,
    promotion_threshold: float,
    auc_margin: float | None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "decision": decision,
        "final_stage": final_stage,
        "registry_uri": registry_uri,
        "challenger_auc": challenger_auc,
        "champion_version": champion_version,
        "champion_auc": champion_auc,
        "promotion_threshold": promotion_threshold,
        "auc_margin": auc_margin,
    }


def emit_repository_dispatch(
    *,
    repository: str,
    token: str,
    client_payload: dict[str, Any],
    event_type: str = DEFAULT_EVENT_TYPE,
    github_api_url: str = DEFAULT_GITHUB_API_URL,
    timeout: int = 10,
) -> None:
    body = json.dumps(
        {
            "event_type": event_type,
            "client_payload": client_payload,
        }
    ).encode("utf-8")

    req = request.Request(
        url=f"{github_api_url.rstrip('/')}/repos/{repository}/dispatches",
        data=body,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "bt4301-group-9-mlflow-dispatch",
        },
    )
    with request.urlopen(req, timeout=timeout):
        return
