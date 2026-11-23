"""Offline tests for ``/``, ``/health``, versioning, error handling and the OpenAPI description.

The app runs with fakes for the store and the models; see ``conftest.py``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import api.service
from api.main import app


def test_root_lists_endpoints(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert "POST /v1/query" in body["endpoints"]
    assert "POST /v1/query/stream" in body["endpoints"]

def test_api_is_versioned_but_root_and_health_are_not(client):
    """Every endpoint but ``/`` and ``/health`` moved under ``/v1``; old paths are gone."""
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/v1/health").status_code == 404
    assert client.post("/retrieve", json={"query": "x"}).status_code == 404
    assert client.post("/query", json={"query": "x"}).status_code == 404
    assert client.get("/query/stream", params={"query": "x"}).status_code == 404
    assert client.get("/images/any-id/0").status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert {p for p in paths if not p.startswith("/v1/")} == {"/", "/health"}

def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vector_store": "loaded"}

def test_health_reports_store_failure_as_503(client, backend):
    """A broken store is a 503, so an uptime probe can read the code alone."""
    backend.store_error = "store missing"
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert response.json()["detail"] == "Database unavailable: store missing"

def test_unexpected_error_is_500_without_internals(backend, monkeypatch):
    """The caller sees a generic message; the real one stays in the server log."""

    def broken(retriever, query):
        raise RuntimeError("secret internal path /srv/app")

    monkeypatch.setattr(api.service, "retrieve_chunks", broken)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/v1/retrieve", json={"query": "LM317 pins"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}

def test_every_route_and_error_is_described(client):
    """Each operation has a summary and description, and each error a schema."""
    spec = client.get("/openapi.json").json()
    assert "base64" not in spec["info"]["description"], "responses carry figure URLs, not bytes"
    assert {tag["name"] for tag in spec["tags"]} == {"query", "retrieval", "providers", "documents", "images", "system"}
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            assert operation.get("summary") and operation.get("description"), f"{method} {path}"
            for code in ("404", "500", "502"):
                if code in operation["responses"]:
                    schema = operation["responses"][code]["content"]["application/json"]["schema"]
                    assert schema == {"$ref": "#/components/schemas/ErrorResponse"}, f"{method} {path} {code}"
            if "422" in operation["responses"]:
                schema = operation["responses"]["422"]["content"]["application/json"]["schema"]
                assert schema == {"$ref": "#/components/schemas/ValidationErrorResponse"}, f"{method} {path}"
    issue = spec["components"]["schemas"]["ValidationIssue"]["properties"]
    assert "input" not in issue, "the 422 handler strips the rejected value"
