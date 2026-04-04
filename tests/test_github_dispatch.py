from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.mlops.github_dispatch import build_model_promoted_payload


def test_build_model_promoted_payload_contains_expected_fields() -> None:
    payload = build_model_promoted_payload(
        model_name="KKBox-Churn-Classifier",
        version=2,
        run_id="abc123",
        decision="promoted_over_champion",
        final_stage="Production",
        registry_uri="models:/KKBox-Churn-Classifier/Production",
        challenger_auc=0.9816,
        champion_version=1,
        champion_auc=0.9815,
        promotion_threshold=0.0,
        auc_margin=0.0001,
    )

    assert payload["model_name"] == "KKBox-Churn-Classifier"
    assert payload["version"] == 2
    assert payload["decision"] == "promoted_over_champion"
    assert payload["final_stage"] == "Production"
    assert json.loads(json.dumps(payload)) == payload
