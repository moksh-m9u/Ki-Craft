"""Offline tests for ``POST /v1/retrieve`` and the figure URLs ``/v1/images/{record_id}/{position}``.

The app runs with fakes for the store and the models; see ``conftest.py``.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

import src.figures
from api.main import app
from tests.api.fakes import JPEG_BYTES, LM317_ID, TLV1117_ID, ESCAPING_ID


def test_retrieve_returns_chunks_with_tables_and_images(client):
    response = client.post("/v1/retrieve", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "LM317 pins"
    assert body["num_chunks"] == 2
    assert body["total_images"] == 1
    assert body["total_tables"] == 1
    assert "response_id" not in body

    first, second = body["chunks"]
    assert first["chunk_id"] == 1
    assert first["source"] == "lm317"
    assert first["raw_text"] == "Pin 1 ADJ, pin 2 VOUT, pin 3 VIN."
    assert first["has_table"] and not first["has_image"]
    assert first["record_id"] == LM317_ID
    assert "images_base64" not in first
    assert first["image_urls"] == []
    assert "original_content" not in first["metadata"]
    assert second["record_id"] == TLV1117_ID
    assert second["has_image"]
    assert second["image_urls"] == [f"/v1/images/{TLV1117_ID}/0"]

def test_retrieve_rejects_bad_input(client):
    assert client.post("/v1/retrieve", json={"query": ""}).status_code == 422
    assert client.post("/v1/retrieve", json={"query": "x", "top_k": 31}).status_code == 422

def test_building_a_response_reads_no_image_file(client, monkeypatch):
    """Responses carry URLs, so retrieval alone never opens a figure."""

    def opened(path):
        raise AssertionError(f"a response build read the file {path}")

    monkeypatch.setattr(src.figures.Path, "read_bytes", opened)
    assert client.post("/v1/retrieve", json={"query": "x", "top_k": 2}).status_code == 200

def test_image_url_serves_the_file(client):
    body = client.post("/v1/retrieve", json={"query": "LM317 pins", "top_k": 2}).json()
    response = client.get(body["chunks"][1]["image_urls"][0])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=86400"
    assert response.content == JPEG_BYTES

def test_image_url_is_stable_across_restarts(backend):
    """The same URL works from a fresh app: nothing is kept between requests."""
    with TestClient(app) as first:
        url = first.post("/v1/retrieve", json={"query": "x", "top_k": 2}).json()["chunks"][1]["image_urls"][0]
    with TestClient(app) as second:
        assert second.get(url).content == JPEG_BYTES

def test_unknown_image_is_404(client):
    assert client.get(f"/v1/images/{TLV1117_ID}/1").status_code == 404, "no second figure"
    assert client.get("/v1/images/99999999-9999-4999-8999-999999999999/0").status_code == 404
    assert client.get(f"/v1/images/{LM317_ID}/0").status_code == 404, "a chunk without figures"

def test_image_url_must_name_a_record_and_position(client):
    assert client.get("/v1/images/not-a-uuid/0").status_code == 422
    assert client.get(f"/v1/images/{TLV1117_ID}/-1").status_code == 422
    assert client.get("/v1/images/..%2F..%2Fsecret.txt/0").status_code in (404, 422)

def test_stored_path_outside_the_images_folder_is_not_served(client):
    response = client.get(f"/v1/images/{ESCAPING_ID}/0")
    assert response.status_code == 404
    assert b"outside the images folder" not in response.content

def test_image_with_broken_store_is_503(client, backend):
    backend.store_error = "store missing"
    assert client.get(f"/v1/images/{TLV1117_ID}/0").status_code == 503

def test_database_lost_mid_request_is_503_without_the_sql(client, monkeypatch):
    """A connection that drops after startup is a 503 the client can retry, not a 500."""
    import api.service

    def lost(*args):
        raise OperationalError("SELECT images.path FROM images", {"chunk_id_1": "x"}, Exception("No route to host"))

    monkeypatch.setattr(api.service, "figure_path", lost)
    response = client.get(f"/v1/images/{TLV1117_ID}/0")
    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable; try again shortly."}
    assert "SELECT" not in response.text


def test_retrieve_does_not_take_anllm_body(client):
    """Retrieval never calls a model, so its body has no ``llm`` field."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert "llm" not in schemas["RetrieveRequest"]["properties"]
    assert "llm" in schemas["QueryRequest"]["properties"]

def test_retrieve_documents_only_the_fields_it_uses(client):
    """max_images only matters when a model is called, so only query bodies list it."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert set(schemas["RetrieveRequest"]["properties"]) == {"query", "top_k"}
    assert {"max_images", "llm"} <= set(schemas["QueryRequest"]["properties"])
