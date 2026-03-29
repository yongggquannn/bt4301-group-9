"""Minimal churn-risk web app (FastAPI)."""

import csv
import os
from pathlib import Path

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
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
# Global feature importance (loaded once at startup)
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


TOP_3_FEATURES: list[str] = _load_top_features()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="KKBox Churn Risk")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LATEST_PREDICTION_SQL = """
SELECT customer_id, churn_probability, risk_tier, scored_at
  FROM processed.churn_predictions
 WHERE customer_id = %s
 ORDER BY scored_at DESC
 LIMIT 1;
"""


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
    return {
        "customer_id": row["customer_id"],
        "churn_probability": float(row["churn_probability"]),
        "risk_tier": row["risk_tier"],
        "top_3_features": TOP_3_FEATURES,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "error": None, "customer_id": None},
    )


@app.get("/customer/{customer_id}/churn-risk")
def churn_risk_api(customer_id: str) -> dict:
    result = _get_prediction(customer_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result


@app.get("/lookup", response_class=HTMLResponse)
def lookup(request: Request, customer_id: str = Query(...)) -> HTMLResponse:
    result = _get_prediction(customer_id)
    error = "Customer not found." if result is None else None
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "error": error,
            "customer_id": customer_id,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("source.webapp.app:app", host="0.0.0.0", port=8000, reload=True)
