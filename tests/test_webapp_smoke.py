from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_home_page_renders_with_required_env(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "placeholder")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "kkbox")
    monkeypatch.setenv("POSTGRES_USER", "bt4301")
    monkeypatch.setenv("POSTGRES_PASSWORD", "bt4301pass")

    sys.modules.pop("source.webapp.app", None)
    app_module = importlib.import_module("source.webapp.app")

    client = TestClient(app_module.app)
    response = client.get("/")

    assert response.status_code == 200
    assert "retention" in response.text.lower()


def _load_app_module(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "placeholder")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "kkbox")
    monkeypatch.setenv("POSTGRES_USER", "bt4301")
    monkeypatch.setenv("POSTGRES_PASSWORD", "bt4301pass")

    sys.modules.pop("source.webapp.app", None)
    return importlib.import_module("source.webapp.app")


def test_dashboard_renders_summary_and_actions(monkeypatch) -> None:
    app_module = _load_app_module(monkeypatch)
    sample_customers = [
        {
            "customer_id": "cust-1",
            "customer_url": "cust-1",
            "churn_probability": 0.87,
            "risk_tier": "High",
            "scored_at": "2026-04-16 09:00 UTC",
            "customer_segment": "Low Engagement",
            "intervention_priority": "Urgent",
            "recommended_action": "Send re-engagement campaign",
            "business_explanations": [
                "Listening activity has dropped, which is often an early sign of churn."
            ],
            "top_3_features": ["num_active_days"],
            "top_3_shap": None,
        }
    ]
    monkeypatch.setattr(app_module, "_get_top_customers", lambda: sample_customers)
    monkeypatch.setattr(
        app_module,
        "_get_dashboard_context",
        lambda: {
            "portfolio": {
                "high_count": 12,
                "medium_count": 18,
                "low_count": 20,
                "total_scored": 50,
                "latest_scored_at": "2026-04-16 09:00 UTC",
            },
            "monitoring": {
                "monitored_at": "2026-04-15 08:00 UTC",
                "baseline_auc": 0.85,
                "current_auc": 0.83,
                "auc_delta": 0.02,
                "max_psi": 0.08,
                "breached": False,
                "breached_reasons": None,
            },
            "distribution": [
                {"bucket": "0-10%", "count": 2, "pct": 10.0},
                {"bucket": "10-20%", "count": 3, "pct": 15.0},
            ],
            "segments_by_type": {
                "tenure": [
                    {"name": "New (<=3 txns)", "total": 10, "high_count": 5, "avg_churn_prob": 0.72},
                    {"name": "Established (11+ txns)", "total": 20, "high_count": 3, "avg_churn_prob": 0.35},
                ],
                "payment_behavior": [
                    {"name": "Auto-Renew", "total": 30, "high_count": 8, "avg_churn_prob": 0.45},
                ],
            },
            "segment_type_labels": {
                "tenure": "Customer Tenure",
                "payment_behavior": "Payment Behavior",
            },
        },
    )

    client = TestClient(app_module.app)
    response = client.get("/dashboard")

    assert response.status_code == 200
    # Enriched customer data from _get_top_customers reaches the template
    assert "cust-1" in response.text
    assert "High" in response.text
    # B-specific template assertions (portfolio cards, segment breakdowns,
    # distribution chart) are verified in feat/workstream-b-manager-view.


def test_customer_page_and_api_include_recommendation_fields(monkeypatch) -> None:
    app_module = _load_app_module(monkeypatch)
    sample_payload = {
        "customer_id": "cust-2",
        "customer_url": "cust-2",
        "churn_probability": 0.64,
        "risk_tier": "Medium",
        "scored_at": "2026-04-16 09:00 UTC",
        "customer_segment": "Payment Friction",
        "intervention_priority": "Medium",
        "recommended_action": "Send targeted reminder or lightweight incentive",
        "business_explanations": [
            "Recent cancellation behaviour suggests the subscription relationship is unstable."
        ],
        "top_3_features": ["cancel_count"],
        "top_3_shap": [{"feature": "cancel_count", "shap_value": 0.42}],
    }
    monkeypatch.setattr(app_module, "_get_prediction", lambda customer_id: sample_payload)

    client = TestClient(app_module.app)

    customer_response = client.get("/customer/cust-2")
    assert customer_response.status_code == 200
    assert "Recommended Action" in customer_response.text
    assert "Payment Friction" in customer_response.text

    api_response = client.get("/customer/cust-2/churn-risk")
    assert api_response.status_code == 200
    payload = api_response.json()
    assert payload["recommended_action"] == sample_payload["recommended_action"]
    assert payload["intervention_priority"] == sample_payload["intervention_priority"]
    assert payload["customer_segment"] == sample_payload["customer_segment"]
    assert payload["business_explanations"] == sample_payload["business_explanations"]
