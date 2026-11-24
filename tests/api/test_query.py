"""Offline tests for ``POST /v1/query`` and ``POST /v1/query/stream``, with the default model and a chosen one.

The app runs with fakes for the store and the models; see ``conftest.py``.
"""

from __future__ import annotations

import logging

import pytest

from tests.api.fakes import TLV1117_ID, parse_sse, llm_body


def test_query_with_broken_store_is_503(client, backend):
    backend.store_error = "store missing"
    response = client.post("/v1/query", json={"query": "LM317 pins"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable: store missing"}

def test_query_returns_answer_and_usage(client, backend):
    response = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "Hello world"
    assert body["answer_characters"] == len("Hello world")
    assert body["num_chunks"] == 2
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["prompt_images"] == 1

    system, user = backend.prompts[0]
    assert "using only the context provided" in system.content
    content = user.content
    assert content[-1] == {"type": "text", "text": "QUESTION: LM317 pins"}
    assert content[2]["text"].startswith("[C2] Document: tlv1117")
    assert content[3] == {"type": "text", "text": "[IMG1] Image from tlv1117, chunk C2"}
    assert content[4]["type"] == "image_url"

def test_query_lists_the_figures_the_model_saw(client):
    """Each prompt figure comes back with its label, URL, chunk and document."""
    body = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2}).json()
    assert body["figures"] == [
        {"label": "IMG1", "url": f"/v1/images/{TLV1117_ID}/0", "chunk_id": 2, "source": "tlv1117"}
    ]
    assert body["figures"][0]["url"] == body["chunks"][1]["image_urls"][0]

def test_query_checks_the_answer_citations(client, backend):
    """Cited chunks and figures that exist are listed; invented labels are unknown."""
    backend.second_token = " world [C2, IMG1] and [IMG3] [C9] [C2] [1]"
    body = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2}).json()
    assert body["citations"] == {"chunks": [2], "figures": ["IMG1"], "unknown": ["IMG3", "C9"]}

def test_answer_without_citations_has_empty_citations(client):
    body = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2}).json()
    assert body["citations"] == {"chunks": [], "figures": [], "unknown": []}

def test_max_images_zero_lists_every_sent_figure(client):
    body = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2, "max_images": 0}).json()
    assert [figure["label"] for figure in body["figures"]] == ["IMG1"]

def test_query_defaults_come_from_settings(client, backend):
    """Without top_k or max_images the settings decide, and every chunk is used."""
    from config.settings import GENERATION_MAX_IMAGES, RETRIEVAL_K

    body = client.post("/v1/query", json={"query": "LM317 pins"}).json()
    assert body["num_chunks"] == min(RETRIEVAL_K, 2)
    assert body["usage"]["chunks_used"] == body["num_chunks"]
    assert body["usage"]["prompt_images"] <= GENERATION_MAX_IMAGES

    prompt = "\n".join(part["text"] for part in backend.prompts[0][1].content if part["type"] == "text")
    assert "[C2] Document:" in prompt

def test_query_llm_failure_is_502(client, backend):
    """A failing provider is a 502 whose detail carries the provider's message."""
    backend.llm_error = "quota exceeded"
    response = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert response.json() == {"detail": "The model provider failed: quota exceeded"}

@pytest.mark.parametrize("finish_reason", ["length", "MAX_TOKENS"])
def test_answer_cut_off_before_any_text_is_502(client, backend, finish_reason):
    """OpenAI-style and Gemini names for "stopped at max_tokens" both count."""
    backend.finish_reason = finish_reason
    response = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert "whole output budget (512 tokens" in response.json()["detail"]
    assert "Raise max_tokens" in response.json()["detail"]

def test_stream_answer_cut_off_before_any_text_is_an_error_event(client, backend):
    """The retrieval frame was already sent, so the failure arrives in the stream."""
    backend.finish_reason = "MAX_TOKENS"
    response = client.post("/v1/query/stream", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["retrieval", "error"]
    assert events[-1][1]["status"] == 502 and "Raise max_tokens" in events[-1][1]["message"]

def test_empty_answer_for_another_reason_is_502(client, backend):
    backend.finish_reason = "content_filter"
    response = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert response.json()["detail"] == "The model returned no answer text (finish_reason: content_filter)."

def test_stream_emits_retrieval_tokens_done(client):
    response = client.post("/v1/query/stream", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["retrieval", "token", "token", "done"]
    assert events[0][1]["num_chunks"] == 2
    assert [data["delta"] for name, data in events if name == "token"] == ["Hello", " world"]
    done = events[-1][1]
    assert done["answer"] == "Hello world"
    assert done["num_chunks"] == 2
    assert done["figures"] == [
        {"label": "IMG1", "url": f"/v1/images/{TLV1117_ID}/0", "chunk_id": 2, "source": "tlv1117"}
    ]
    assert done["citations"] == {"chunks": [], "figures": [], "unknown": []}
    assert "response_id" not in done and "response_id" not in events[0][1]

def test_stream_done_event_checks_citations(client, backend):
    backend.second_token = " [C1] [IMG1] [IMG2]"
    events = parse_sse(client.post("/v1/query/stream", json={"query": "LM317 pins", "top_k": 2}).text)
    assert events[-1][1]["citations"] == {"chunks": [1], "figures": ["IMG1"], "unknown": ["IMG2"]}

def test_stream_reports_llm_failure_as_error_event(client, backend):
    """A provider failure after the first token ends the stream with an error."""
    backend.llm_error_after_first_token = "quota exceeded"
    response = client.post("/v1/query/stream", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["retrieval", "token", "error"]
    assert events[-1][1] == {"status": 502, "message": "The model provider failed: quota exceeded"}

def test_stream_sends_chunks_before_the_model_answers(client, backend):
    """Retrieval goes out first; a model failure before any token is an error event."""
    backend.llm_error = "quota exceeded"
    response = client.post("/v1/query/stream", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["retrieval", "error"]
    assert events[0][1]["num_chunks"] == 2
    assert events[-1][1] == {"status": 502, "message": "The model provider failed: quota exceeded"}

def test_stream_uses_requested_llm_and_body_options(client, backend):
    response = client.post(
        "/v1/query/stream",
        json={"query": "LM317 pins", "top_k": 1, "max_images": 0, "llm": llm_body(max_tokens=720)},
    )
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[0][1]["num_chunks"] == 1
    assert events[-1][1]["usage"]["model"] == "openai/gpt-oss-20b"
    assert backend.client_kwargs[-1]["api_key"] == "test-key"
    assert backend.client_kwargs[-1]["max_tokens"] == 720

def test_stream_with_broken_store_is_503(client, backend):
    backend.store_error = "store missing"
    response = client.post("/v1/query/stream", json={"query": "LM317 pins"})
    assert response.status_code == 503

def test_query_with_llm_uses_chosen_model_and_key(client, backend):
    response = client.post(
        "/v1/query",
        json={"query": "LM317 pins", "top_k": 2,
              "llm": llm_body(provider="gemini", model="gemini-3.5-flash-lite")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hello world"
    assert body["usage"]["model"] == "gemini-3.5-flash-lite"
    # The SecretStr is unwrapped once, so the client gets the plain key.
    assert backend.client_kwargs[-1]["google_api_key"] == "test-key"

def test_query_with_text_only_model_gets_no_figures(client, backend):
    """A model the catalogue marks text-only is sent the text and tables only."""
    response = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2, "llm": llm_body()})
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_images"] == 0
    assert {part["type"] for part in backend.prompts[0][1].content} == {"text"}
    assert response.json()["total_images"] == 1, "the response still lists the chunk's figure URL"
    assert response.json()["figures"] == [], "but no figure was sent to the model"

def test_query_with_llm_without_max_tokens_uses_catalogue_default(client, backend):
    response = client.post(
        "/v1/query", json={"query": "LM317 pins", "top_k": 2, "llm": llm_body(max_tokens=None)}
    )
    assert response.status_code == 200
    assert backend.client_kwargs[-1]["max_tokens"] == 2048

def test_llm_without_api_key_is_422(client, backend):
    """The server's own key is never used for a model the caller picks."""
    llm = llm_body()
    del llm["api_key"]
    response = client.post("/v1/query", json={"query": "x", "llm": llm})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "llm", "api_key"]
    assert backend.prompts == []

@pytest.mark.parametrize("path", ["/v1/query", "/v1/query/stream"])
def test_validation_error_does_not_echo_the_key(client, path):
    """A missing ``llm.model`` would echo the whole ``llm`` object as ``input``."""
    llm = llm_body(api_key="sk-must-not-leak")
    del llm["model"]
    response = client.post(path, json={"query": "x", "llm": llm})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "llm", "model"]
    assert set(error) == {"type", "loc", "msg"}
    assert "sk-must-not-leak" not in response.text

@pytest.mark.parametrize("path", ["/v1/query", "/v1/query/stream"])
def test_query_unknown_model_is_422(client, backend, path):
    """An unknown model is the caller's mistake, rejected before retrieval."""
    response = client.post(
        path, json={"query": "x", "llm": llm_body(provider="gemini", model="no-such-model")}
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Model no-such-model not found for provider gemini"}
    assert backend.prompts == []

def test_query_with_llm_provider_failure_is_502(client, backend):
    backend.llm_error = "invalid API key"
    response = client.post("/v1/query", json={"query": "x", "llm": llm_body()})
    assert response.status_code == 502
    assert response.json() == {"detail": "The model provider failed: invalid API key"}

@pytest.mark.parametrize("llm_error", [None, "invalid API key"])
def test_api_key_never_reaches_the_logs(client, backend, caplog, llm_error):
    backend.llm_error = llm_error
    caplog.set_level(logging.DEBUG)
    client.post("/v1/query", json={"query": "x", "llm": llm_body(api_key="sk-must-not-leak")})
    assert caplog.records, "the request should have logged something"
    assert "sk-must-not-leak" not in caplog.text

def test_custom_route_is_gone(client):
    response = client.post("/v1/query/custom", json={"query": "x", **llm_body()})
    assert response.status_code == 404
