"""End-to-end integration test for the churn-prediction pipeline.

Prerequisites
-------------
1. Full Docker Compose stack running: ``docker compose up -d``
2. A production model registered in MLflow (``KKBox-Churn-Classifier``,
   alias ``Production``).
3. Raw CSV data in ``data/`` for the DataOps DAG.

Run
---
::

    pytest source/tests/test_e2e_integration.py -v -s
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import psycopg2
import pytest
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080/api/v1")
WEBAPP_BASE_URL = os.getenv("WEBAPP_BASE_URL", "http://localhost:8000")

AIRFLOW_USER = os.getenv("AIRFLOW_ADMIN_USER", "admin")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_ADMIN_PASSWORD", "admin")

DATAOPS_DAG_ID = "us8_dataops_e2e_pipeline"
SCORING_DAG_ID = "daily_churn_scoring"

DATAOPS_TIMEOUT = int(os.getenv("E2E_DATAOPS_TIMEOUT", "600"))
SCORING_TIMEOUT = int(os.getenv("E2E_SCORING_TIMEOUT", "300"))
POLL_INTERVAL = int(os.getenv("E2E_POLL_INTERVAL", "15"))

from source.common.db import get_db_config

DB_CONFIG = get_db_config()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def airflow_session() -> requests.Session:
    """Return a ``requests.Session`` pre-configured with Airflow basic auth."""
    session = requests.Session()
    session.auth = (AIRFLOW_USER, AIRFLOW_PASSWORD)
    session.headers.update({"Content-Type": "application/json"})
    return session


def trigger_dag(session: requests.Session, dag_id: str) -> str:
    """Trigger an Airflow DAG and return the ``dag_run_id``."""
    url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"
    resp = session.post(url, json={"conf": {}})
    assert resp.status_code == 200, (
        f"Failed to trigger DAG {dag_id}: {resp.status_code} {resp.text}"
    )
    return resp.json()["dag_run_id"]


def wait_for_dag(
    session: requests.Session,
    dag_id: str,
    dag_run_id: str,
    timeout: int = 600,
    poll_interval: int = 15,
) -> str:
    """Poll until the DAG run reaches a terminal state.

    Returns the final state (``success`` or ``failed``).
    Raises ``TimeoutError`` if *timeout* seconds elapse.
    """
    url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{dag_run_id}"
    deadline = time.monotonic() + timeout

    while True:
        resp = session.get(url)
        assert resp.status_code == 200, (
            f"Failed to poll DAG {dag_id}: {resp.status_code} {resp.text}"
        )
        state = resp.json()["state"]

        if state == "success":
            return state
        if state == "failed":
            pytest.fail(f"DAG {dag_id} run {dag_run_id} failed")
        if state not in ("running", "queued"):
            pytest.fail(
                f"DAG {dag_id} run {dag_run_id} reached unexpected "
                f"state: {state!r}"
            )

        if time.monotonic() > deadline:
            raise TimeoutError(
                f"DAG {dag_id} run {dag_run_id} still '{state}' "
                f"after {timeout}s"
            )

        print(f"  [E2E] {dag_id} state={state}, waiting {poll_interval}s ...")
        time.sleep(poll_interval)


def get_scored_customer_id() -> str:
    """Query PostgreSQL for any customer_id that has a churn prediction."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT customer_id FROM processed.churn_predictions LIMIT 1"
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        pytest.fail(
            "No predictions found in processed.churn_predictions "
            "after scoring DAG succeeded"
        )
    return row[0]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_e2e_dataops_scoring_webapp() -> None:
    """Trigger DataOps DAG -> Scoring DAG -> query webapp API -> assert."""

    # -- 1. Health checks -----------------------------------------------------
    print("\n[E2E] Step 1: Health checks")
    try:
        airflow_resp = requests.get(
            f"{AIRFLOW_BASE_URL}/health", timeout=10
        )
    except requests.ConnectionError:
        pytest.skip("Airflow API not reachable — is the stack running?")

    if airflow_resp.status_code != 200:
        pytest.skip(f"Airflow health check failed: {airflow_resp.status_code}")

    try:
        webapp_resp = requests.get(f"{WEBAPP_BASE_URL}/", timeout=10)
    except requests.ConnectionError:
        pytest.skip("Webapp not reachable — is the stack running?")

    if webapp_resp.status_code != 200:
        pytest.skip(f"Webapp health check failed: {webapp_resp.status_code}")

    print("  [E2E] Airflow and webapp are healthy")

    session = airflow_session()

    # -- 2. Trigger DataOps DAG -----------------------------------------------
    print("[E2E] Step 2: Triggering DataOps DAG")
    dataops_run_id = trigger_dag(session, DATAOPS_DAG_ID)
    print(f"  [E2E] DataOps DAG triggered: {dataops_run_id}")

    # -- 3. Wait for DataOps DAG ----------------------------------------------
    print("[E2E] Step 3: Waiting for DataOps DAG to complete")
    dataops_state = wait_for_dag(
        session, DATAOPS_DAG_ID, dataops_run_id,
        timeout=DATAOPS_TIMEOUT, poll_interval=POLL_INTERVAL,
    )
    print(f"  [E2E] DataOps DAG finished: {dataops_state}")

    # -- 4. Trigger Scoring DAG -----------------------------------------------
    print("[E2E] Step 4: Triggering Scoring DAG")
    scoring_run_id = trigger_dag(session, SCORING_DAG_ID)
    print(f"  [E2E] Scoring DAG triggered: {scoring_run_id}")

    # -- 5. Wait for Scoring DAG ----------------------------------------------
    print("[E2E] Step 5: Waiting for Scoring DAG to complete")
    scoring_state = wait_for_dag(
        session, SCORING_DAG_ID, scoring_run_id,
        timeout=SCORING_TIMEOUT, poll_interval=POLL_INTERVAL,
    )
    print(f"  [E2E] Scoring DAG finished: {scoring_state}")

    # -- 6. Get a scored customer_id ------------------------------------------
    print("[E2E] Step 6: Fetching a scored customer ID from DB")
    customer_id = get_scored_customer_id()
    print(f"  [E2E] Using customer_id: {customer_id}")

    # -- 7. Query webapp API --------------------------------------------------
    print("[E2E] Step 7: Querying webapp churn-risk API")
    api_url = f"{WEBAPP_BASE_URL}/customer/{customer_id}/churn-risk"
    api_resp = requests.get(api_url, timeout=10)

    assert api_resp.status_code == 200, (
        f"Webapp API returned {api_resp.status_code}: {api_resp.text}"
    )

    data = api_resp.json()
    print(f"  [E2E] API response: {data}")

    # -- 8. Assertions --------------------------------------------------------
    print("[E2E] Step 8: Validating response")

    assert data["customer_id"] == customer_id
    assert isinstance(data["churn_probability"], float)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ("High", "Medium", "Low")
    assert isinstance(data["top_3_features"], list)
    assert len(data["top_3_features"]) > 0

    print(
        f"  [E2E] PASSED - customer={customer_id}, "
        f"churn_prob={data['churn_probability']:.4f}, "
        f"tier={data['risk_tier']}"
    )


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
