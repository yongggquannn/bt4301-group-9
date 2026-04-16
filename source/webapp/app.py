"""Action-oriented churn-risk web app (FastAPI)."""

import csv
import json
import os
from pathlib import Path
from urllib.parse import quote, unquote

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Required environment variable {name} is not set. "
            "See README.md Step 3 for setup instructions."
        )
    return val


DB_CONFIG = {
    "host": _require_env("POSTGRES_HOST"),
    "port": int(_require_env("POSTGRES_PORT")),
    "dbname": _require_env("POSTGRES_DB"),
    "user": _require_env("POSTGRES_USER"),
    "password": _require_env("POSTGRES_PASSWORD"),
}

# ---------------------------------------------------------------------------
# Global feature importance (loaded once at startup, optional fallback)
# ---------------------------------------------------------------------------

_IMPORTANCE_CSV = PROJECT_ROOT / "docs" / "artifacts" / "permutation_importance.csv"


def _load_top_features(n: int = 3) -> list[str]:
    if not _IMPORTANCE_CSV.exists():
        raise FileNotFoundError(
            f"Feature importance CSV not found: {_IMPORTANCE_CSV}. "
            "Run the training pipeline first (Step 5)."
        )
    with open(_IMPORTANCE_CSV) as f:
        rows = sorted(
            csv.DictReader(f),
            key=lambda r: float(r["perm_importance_mean"]),
            reverse=True,
        )
    return [r["feature"] for r in rows[:n]]


try:
    TOP_3_FEATURES: list[str] = _load_top_features()
except FileNotFoundError:
    TOP_3_FEATURES = []

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="KKBox Churn Risk")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LATEST_PREDICTION_SQL = """
SELECT
    p.customer_id,
    p.churn_probability,
    p.risk_tier,
    p.scored_at,
    p.shap_values,
    cf.transaction_count,
    cf.cancel_count,
    cf.num_active_days,
    cf.avg_total_secs,
    cf.avg_plan_days,
    cf.latest_is_auto_renew
FROM processed.churn_predictions p
LEFT JOIN processed.customer_features cf
  ON cf.msno = p.customer_id
WHERE p.customer_id = %s
ORDER BY p.scored_at DESC
LIMIT 1;
"""

_DASHBOARD_SQL = """
SELECT
    latest.customer_id,
    latest.churn_probability,
    latest.risk_tier,
    latest.scored_at,
    cf.transaction_count,
    cf.cancel_count,
    cf.num_active_days,
    cf.avg_total_secs,
    cf.avg_plan_days,
    cf.latest_is_auto_renew
FROM (
    SELECT DISTINCT ON (customer_id)
           customer_id, churn_probability, risk_tier, scored_at
      FROM processed.churn_predictions
     ORDER BY customer_id, scored_at DESC
) latest
LEFT JOIN processed.customer_features cf
  ON cf.msno = latest.customer_id
ORDER BY churn_probability DESC
LIMIT 50;
"""

_SUMMARY_COUNTS_SQL = """
SELECT
    COUNT(*) AS total_scored,
    COUNT(*) FILTER (WHERE risk_tier = 'High')   AS high_count,
    COUNT(*) FILTER (WHERE risk_tier = 'Medium') AS medium_count,
    COUNT(*) FILTER (WHERE risk_tier = 'Low')    AS low_count,
    MAX(scored_at) AS latest_scored_at
FROM (
    SELECT DISTINCT ON (customer_id)
           customer_id, risk_tier, scored_at
      FROM processed.churn_predictions
     ORDER BY customer_id, scored_at DESC
) latest;
"""

_DISTRIBUTION_SQL = """
SELECT
    bucket,
    COUNT(*) AS cnt
FROM (
    SELECT
        CASE
            WHEN churn_probability < 0.1 THEN '0-10%'
            WHEN churn_probability < 0.2 THEN '10-20%'
            WHEN churn_probability < 0.3 THEN '20-30%'
            WHEN churn_probability < 0.4 THEN '30-40%'
            WHEN churn_probability < 0.5 THEN '40-50%'
            WHEN churn_probability < 0.6 THEN '50-60%'
            WHEN churn_probability < 0.7 THEN '60-70%'
            WHEN churn_probability < 0.8 THEN '70-80%'
            WHEN churn_probability < 0.9 THEN '80-90%'
            ELSE '90-100%'
        END AS bucket
    FROM (
        SELECT DISTINCT ON (customer_id)
               customer_id, churn_probability
          FROM processed.churn_predictions
         ORDER BY customer_id, scored_at DESC
    ) latest
) bucketed
GROUP BY bucket
ORDER BY bucket;
"""

_SEGMENT_BREAKDOWN_SQL = """
SELECT
    CASE WHEN cf.latest_is_auto_renew = 1 THEN 'Auto-Renew' ELSE 'Manual' END AS segment_name,
    'payment_behavior' AS segment_type,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE cp.risk_tier = 'High') AS high_count,
    ROUND(AVG(cp.churn_probability)::NUMERIC, 4) AS avg_churn_prob
FROM (
    SELECT DISTINCT ON (customer_id)
           customer_id, churn_probability, risk_tier
      FROM processed.churn_predictions
     ORDER BY customer_id, scored_at DESC
) cp
JOIN processed.customer_features cf ON cf.msno = cp.customer_id
GROUP BY cf.latest_is_auto_renew

UNION ALL

SELECT
    CASE
        WHEN cf.cancel_count = 0 THEN 'Never Cancelled'
        WHEN cf.cancel_count = 1 THEN '1 Cancellation'
        ELSE '2+ Cancellations'
    END AS segment_name,
    'cancel_history' AS segment_type,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE cp.risk_tier = 'High') AS high_count,
    ROUND(AVG(cp.churn_probability)::NUMERIC, 4) AS avg_churn_prob
FROM (
    SELECT DISTINCT ON (customer_id)
           customer_id, churn_probability, risk_tier
      FROM processed.churn_predictions
     ORDER BY customer_id, scored_at DESC
) cp
JOIN processed.customer_features cf ON cf.msno = cp.customer_id
GROUP BY
    CASE
        WHEN cf.cancel_count = 0 THEN 'Never Cancelled'
        WHEN cf.cancel_count = 1 THEN '1 Cancellation'
        ELSE '2+ Cancellations'
    END

UNION ALL

SELECT
    CASE
        WHEN cf.num_active_days < 30 THEN 'Light (<30 days)'
        WHEN cf.num_active_days < 90 THEN 'Moderate (30-90 days)'
        ELSE 'Heavy (90+ days)'
    END AS segment_name,
    'usage_intensity' AS segment_type,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE cp.risk_tier = 'High') AS high_count,
    ROUND(AVG(cp.churn_probability)::NUMERIC, 4) AS avg_churn_prob
FROM (
    SELECT DISTINCT ON (customer_id)
           customer_id, churn_probability, risk_tier
      FROM processed.churn_predictions
     ORDER BY customer_id, scored_at DESC
) cp
JOIN processed.customer_features cf ON cf.msno = cp.customer_id
GROUP BY
    CASE
        WHEN cf.num_active_days < 30 THEN 'Light (<30 days)'
        WHEN cf.num_active_days < 90 THEN 'Moderate (30-90 days)'
        ELSE 'Heavy (90+ days)'
    END

UNION ALL

SELECT
    CASE
        WHEN cf.transaction_count <= 3 THEN 'New (<=3 txns)'
        WHEN cf.transaction_count <= 10 THEN 'Developing (4-10 txns)'
        ELSE 'Established (11+ txns)'
    END AS segment_name,
    'tenure' AS segment_type,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE cp.risk_tier = 'High') AS high_count,
    ROUND(AVG(cp.churn_probability)::NUMERIC, 4) AS avg_churn_prob
FROM (
    SELECT DISTINCT ON (customer_id)
           customer_id, churn_probability, risk_tier
      FROM processed.churn_predictions
     ORDER BY customer_id, scored_at DESC
) cp
JOIN processed.customer_features cf ON cf.msno = cp.customer_id
GROUP BY
    CASE
        WHEN cf.transaction_count <= 3 THEN 'New (<=3 txns)'
        WHEN cf.transaction_count <= 10 THEN 'Developing (4-10 txns)'
        ELSE 'Established (11+ txns)'
    END
ORDER BY segment_type, segment_name;
"""

_MONITORING_STATUS_SQL = """
SELECT
    monitored_at,
    baseline_auc,
    current_auc,
    auc_delta,
    max_psi,
    breached,
    breached_reasons,
    baseline_row_count,
    current_row_count
FROM processed.model_monitoring_results
ORDER BY monitored_at DESC
LIMIT 1;
"""

_SEGMENT_TYPE_LABELS: dict[str, str] = {
    "tenure": "Customer Tenure",
    "payment_behavior": "Payment Behavior",
    "cancel_history": "Cancellation History",
    "usage_intensity": "Usage Intensity",
}

_FEATURE_EXPLANATIONS = {
    "transaction_count": "Transaction activity is lower than what we usually see for retained customers.",
    "cancel_count": "Recent cancellation behaviour suggests the subscription relationship is unstable.",
    "num_active_days": "Listening activity has dropped, which is often an early sign of churn.",
    "avg_total_secs": "Average listening time is low, pointing to weaker engagement.",
    "avg_plan_days": "Plan duration patterns look less sticky than healthier subscribers.",
    "latest_is_auto_renew": "Auto-renewal behaviour suggests weaker renewal commitment.",
    "total_amount_paid": "Payment history indicates weaker subscription commitment than typical retained users.",
    "total_num_songs": "Overall listening volume is lower than the platform usually sees from retained customers.",
    "avg_num_songs": "Average listening volume has softened compared with healthier users.",
    "total_num_unq": "Content variety has narrowed, which can signal declining engagement.",
    "avg_num_unq": "The customer is exploring less content than typical retained listeners.",
    "latest_membership_expire_date": "Membership timing indicates upcoming renewal risk.",
    "renewal_count": "Renewal history is weaker than what we usually see among retained subscribers.",
    "registered_via": "Signup channel patterns resemble customers who are more likely to churn.",
    "bd": "Age-band patterns resemble users with elevated churn risk.",
    "gender": "This customer profile matches churn patterns seen in similar user groups.",
    "city": "Location-based patterns align with groups that show elevated churn risk.",
}


def _parse_shap_values(raw: object) -> list[dict] | None:
    """Parse shap_values from DB (JSONB auto-parsed or raw string)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(raw, list) and raw:
        return raw
    return None


def _safe_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _safe_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _humanize_feature_name(feature: str) -> str:
    return feature.replace("_", " ").strip().capitalize()


def _explanation_for_feature(feature: str) -> str:
    return _FEATURE_EXPLANATIONS.get(
        feature,
        f"{_humanize_feature_name(feature)} is contributing to this customer's churn risk.",
    )


def _derive_segment(metrics: dict) -> str:
    cancel_count = metrics["cancel_count"] or 0
    num_active_days = metrics["num_active_days"]
    avg_total_secs = metrics["avg_total_secs"]
    latest_is_auto_renew = metrics["latest_is_auto_renew"]

    if cancel_count > 0 or latest_is_auto_renew == 0:
        return "Payment Friction"
    if (num_active_days is not None and num_active_days < 10) or (
        avg_total_secs is not None and avg_total_secs < 900
    ):
        return "Low Engagement"
    if metrics["risk_tier"] == "Low":
        return "Stable"
    return "General"


def _derive_priority(metrics: dict, segment: str) -> str:
    if metrics["risk_tier"] == "High":
        if segment in {"Payment Friction", "Low Engagement"}:
            return "Urgent"
        return "High"
    if metrics["risk_tier"] == "Medium":
        return "Medium"
    return "Low"


def _derive_recommended_action(metrics: dict, segment: str) -> str:
    if metrics["risk_tier"] == "High" and segment == "Low Engagement":
        return "Send re-engagement campaign"
    if metrics["risk_tier"] == "High" and segment == "Payment Friction":
        return "Escalate to retention team"
    if metrics["risk_tier"] == "Medium":
        return "Send targeted reminder or lightweight incentive"
    return "No immediate action; monitor only"


def _build_business_explanations(
    shap_entries: list[dict] | None,
    top_features: list[str],
) -> list[str]:
    feature_names: list[str] = []
    if shap_entries:
        feature_names.extend(entry["feature"] for entry in shap_entries[:3] if entry.get("feature"))
    if not feature_names:
        feature_names.extend(top_features[:3])
    return [_explanation_for_feature(feature) for feature in feature_names[:3]]


def _enrich_prediction_row(row: dict) -> dict:
    shap_entries = _parse_shap_values(row.get("shap_values"))
    if shap_entries:
        top_3_shap = shap_entries[:3]
        top_3_features = [entry["feature"] for entry in top_3_shap]
    else:
        top_3_shap = None
        top_3_features = TOP_3_FEATURES

    metrics = {
        "risk_tier": row["risk_tier"],
        "transaction_count": _safe_int(row.get("transaction_count")),
        "cancel_count": _safe_int(row.get("cancel_count")),
        "num_active_days": _safe_int(row.get("num_active_days")),
        "avg_total_secs": _safe_float(row.get("avg_total_secs")),
        "avg_plan_days": _safe_float(row.get("avg_plan_days")),
        "latest_is_auto_renew": _safe_int(row.get("latest_is_auto_renew")),
    }
    customer_segment = _derive_segment(metrics)
    intervention_priority = _derive_priority(metrics, customer_segment)
    recommended_action = _derive_recommended_action(metrics, customer_segment)

    return {
        "customer_id": row["customer_id"],
        "customer_url": quote(row["customer_id"], safe=""),
        "churn_probability": float(row["churn_probability"]),
        "risk_tier": row["risk_tier"],
        "scored_at": row["scored_at"].strftime("%Y-%m-%d %H:%M UTC") if row["scored_at"] else "",
        "top_3_features": top_3_features,
        "top_3_shap": top_3_shap,
        "customer_segment": customer_segment,
        "intervention_priority": intervention_priority,
        "recommended_action": recommended_action,
        "business_explanations": _build_business_explanations(top_3_shap, top_3_features),
    }


def _build_dashboard_summary(customers: list[dict]) -> dict:
    total = len(customers)
    if total == 0:
        return {
            "total_customers": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "avg_churn_probability": 0.0,
        }
    return {
        "total_customers": total,
        "high_risk_count": sum(1 for c in customers if c["risk_tier"] == "High"),
        "medium_risk_count": sum(1 for c in customers if c["risk_tier"] == "Medium"),
        "avg_churn_probability": sum(c["churn_probability"] for c in customers) / total,
    }


def _get_prediction(customer_id: str) -> dict | None:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_LATEST_PREDICTION_SQL, (customer_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None

    return _enrich_prediction_row(row)


def _get_top_customers() -> list[dict]:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_DASHBOARD_SQL)
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_enrich_prediction_row(row) for row in rows]


def _get_dashboard_context() -> dict:
    """Fetch portfolio-level summary, distribution, segments, and monitoring."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except psycopg2.OperationalError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SUMMARY_COUNTS_SQL)
            summary_row = cur.fetchone() or {}

            cur.execute(_DISTRIBUTION_SQL)
            distribution_rows = cur.fetchall()

            cur.execute(_SEGMENT_BREAKDOWN_SQL)
            segment_rows = cur.fetchall()

            cur.execute(_MONITORING_STATUS_SQL)
            monitoring_row = cur.fetchone()
    finally:
        conn.close()

    # -- Portfolio summary --
    latest_scored_at = summary_row.get("latest_scored_at")
    portfolio = {
        "total_scored": int(summary_row.get("total_scored", 0)),
        "high_count": int(summary_row.get("high_count", 0)),
        "medium_count": int(summary_row.get("medium_count", 0)),
        "low_count": int(summary_row.get("low_count", 0)),
        "latest_scored_at": (
            latest_scored_at.strftime("%Y-%m-%d %H:%M UTC")
            if latest_scored_at
            else "N/A"
        ),
    }

    # -- Distribution: fill all 10 buckets, compute bar-width percentages --
    all_buckets = [
        "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
        "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
    ]
    dist_map = {r["bucket"]: int(r["cnt"]) for r in distribution_rows}
    max_count = max(dist_map.values()) if dist_map else 1
    distribution = [
        {
            "bucket": b,
            "count": dist_map.get(b, 0),
            "pct": round(100 * dist_map.get(b, 0) / max_count) if max_count else 0,
        }
        for b in all_buckets
    ]

    # -- Segments: group by type --
    segments_by_type: dict[str, list[dict]] = {}
    for row in segment_rows:
        seg_type = row["segment_type"]
        segments_by_type.setdefault(seg_type, []).append(
            {
                "name": row["segment_name"],
                "total": int(row["total"]),
                "high_count": int(row["high_count"]),
                "avg_churn_prob": float(row["avg_churn_prob"]),
            }
        )

    # -- Monitoring status --
    monitoring: dict | None = None
    if monitoring_row:
        monitoring = {
            "monitored_at": (
                monitoring_row["monitored_at"].strftime("%Y-%m-%d %H:%M UTC")
                if monitoring_row["monitored_at"]
                else "N/A"
            ),
            "baseline_auc": (
                float(monitoring_row["baseline_auc"])
                if monitoring_row["baseline_auc"] is not None
                else None
            ),
            "current_auc": (
                float(monitoring_row["current_auc"])
                if monitoring_row["current_auc"] is not None
                else None
            ),
            "auc_delta": (
                float(monitoring_row["auc_delta"])
                if monitoring_row["auc_delta"] is not None
                else None
            ),
            "max_psi": float(monitoring_row["max_psi"]),
            "breached": monitoring_row["breached"],
            "breached_reasons": monitoring_row["breached_reasons"],
        }

    return {
        "portfolio": portfolio,
        "distribution": distribution,
        "segments_by_type": segments_by_type,
        "monitoring": monitoring,
        "segment_type_labels": _SEGMENT_TYPE_LABELS,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    customers = _get_top_customers()
    ctx = _get_dashboard_context()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "customers": customers,
            "summary": _build_dashboard_summary(customers),
            **ctx,
        },
    )


@app.get("/customer/{customer_id:path}/churn-risk")
def churn_risk_api(customer_id: str) -> dict:
    customer_id = unquote(customer_id)
    result = _get_prediction(customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result


@app.get("/customer/{customer_id:path}", response_class=HTMLResponse)
def customer_detail(request: Request, customer_id: str) -> HTMLResponse:
    customer_id = unquote(customer_id)
    result = _get_prediction(customer_id)
    error = "Customer not found." if result is None else None
    return templates.TemplateResponse(
        request,
        "customer.html",
        {
            "result": result,
            "error": error,
            "customer_id": customer_id,
        },
    )


@app.get("/lookup", response_class=HTMLResponse)
def lookup_redirect(customer_id: str = Query(...)) -> RedirectResponse:
    return RedirectResponse(url=f"/customer/{quote(customer_id, safe='')}", status_code=302)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("source.webapp.app:app", host="0.0.0.0", port=8000, reload=True)
