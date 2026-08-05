"""Smoke test: the app boots and ``/health`` responds.

Green from Day 1. It proves Settings load and FastAPI wires up correctly,
which means later phases build on a known-good foundation.
"""

from fastapi.testclient import TestClient

from copilot.api.main import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert "app" in body
    assert "environment" in body
