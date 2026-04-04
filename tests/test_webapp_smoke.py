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
    assert "customer" in response.text.lower()
