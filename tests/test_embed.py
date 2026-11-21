"""Tests for the embedding factory: watsonx, local GGUF and OpenRouter (no network calls)."""

import functools
import json

import httpx
import openai
import pytest
from langchain_openai import OpenAIEmbeddings

import src.embed as embed


@pytest.fixture(autouse=True)
def watsonx_backend(monkeypatch):
    """Default to watsonx so the tests do not depend on EMBEDDING_BACKEND in .env."""
    monkeypatch.setattr(embed, "EMBEDDING_BACKEND", "watsonx")


def test_builds_watsonx_embeddings_from_settings(monkeypatch):
    captured = {}
    monkeypatch.setattr(embed, "WATSONX_API_KEY", "key")
    monkeypatch.setattr(embed, "WATSONX_PROJECT_ID", "project")
    monkeypatch.setattr(embed, "WatsonxEmbeddings", lambda **kw: captured.update(kw) or "model")

    assert embed.load_embedding_model() == "model"
    assert captured == {
        "model_id": embed.EMBEDDING_MODEL,
        "url": embed.WATSONX_URL,
        "project_id": "project",
        "api_key": "key",
        "params": {"truncate_input_tokens": embed.EMBEDDING_MAX_INPUT_TOKENS},
    }


def test_missing_credentials_are_named(monkeypatch):
    monkeypatch.setattr(embed, "WATSONX_API_KEY", "")
    monkeypatch.setattr(embed, "WATSONX_PROJECT_ID", "")

    with pytest.raises(RuntimeError, match="WATSONX_API_KEY, WATSONX_PROJECT_ID"):
        embed.load_embedding_model()


class _FakeLlama:
    """Stands in for llama_cpp.Llama: records its arguments and the embed calls."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def embed(self, text, normalize, truncate):
        self.calls.append((text, normalize, truncate))
        return [0.6, 0.8]


def _fake_llama_cpp(monkeypatch):
    """Install a fake llama_cpp module and return it."""
    import sys
    import types

    fake = types.SimpleNamespace(LLAMA_POOLING_TYPE_CLS=2, Llama=_FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    return fake


def test_local_backend_loads_gguf_and_skips_watsonx(monkeypatch, tmp_path):
    _fake_llama_cpp(monkeypatch)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"")
    monkeypatch.setattr(embed, "EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(embed, "LOCAL_EMBEDDING_GGUF_PATH", str(model_file))
    monkeypatch.setattr(embed, "WATSONX_API_KEY", "")

    model = embed.load_embedding_model()

    assert isinstance(model, embed.GGUFEmbeddings)
    assert model._llm.kwargs == {
        "model_path": str(model_file),
        "embedding": True,
        "pooling_type": 2,
        "n_ctx": embed.EMBEDDING_MAX_INPUT_TOKENS,
        "n_batch": embed.EMBEDDING_MAX_INPUT_TOKENS,
        "n_threads": embed.LOCAL_EMBEDDING_THREADS,
        "verbose": False,
    }


def test_gguf_embeddings_normalise_and_truncate(monkeypatch, tmp_path):
    _fake_llama_cpp(monkeypatch)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"")
    model = embed.GGUFEmbeddings(str(model_file), threads=1)

    assert model.embed_documents(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]
    assert model._llm.calls == [("a", True, True), ("b", True, True)]


def test_missing_gguf_file_is_named(tmp_path):
    missing = tmp_path / "absent.gguf"

    with pytest.raises(FileNotFoundError, match="absent.gguf"):
        embed.GGUFEmbeddings(str(missing), threads=1)


def test_gguf_embeddings_never_run_two_calls_at_once(monkeypatch, tmp_path):
    import threading
    import time

    class _OverlapLlama(_FakeLlama):
        """Records how many embed calls ran at the same time."""

        active = 0
        most = 0

        def embed(self, text, normalize, truncate):
            type(self).active += 1
            type(self).most = max(type(self).most, type(self).active)
            time.sleep(0.01)
            type(self).active -= 1
            return [1.0]

    fake = _fake_llama_cpp(monkeypatch)
    fake.Llama = _OverlapLlama
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"")
    model = embed.GGUFEmbeddings(str(model_file), threads=1)

    threads = [threading.Thread(target=model.embed_query, args=("q",)) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert _OverlapLlama.most == 1


def test_local_backend_without_llama_cpp_says_what_to_do(monkeypatch):
    # None in sys.modules makes `import llama_cpp` raise ImportError, as in the API image.
    import sys

    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setattr(embed, "EMBEDDING_BACKEND", "local")

    with pytest.raises(RuntimeError, match="needs llama-cpp-python.*API Docker image"):
        embed.load_embedding_model()


@pytest.fixture
def openrouter_backend(monkeypatch):
    """Select the OpenRouter backend with a fake key and a known model."""
    monkeypatch.setattr(embed, "EMBEDDING_BACKEND", "openrouter")
    monkeypatch.setattr(embed, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(embed, "OPENROUTER_EMBEDDING_MODEL", "vendor/embed-model:free")


def test_openrouter_backend_builds_an_openai_client(openrouter_backend, monkeypatch):
    captured = {}
    monkeypatch.setattr(embed, "OpenAIEmbeddings", lambda **kw: captured.update(kw) or "model")
    monkeypatch.setattr(embed, "WatsonxEmbeddings", lambda **kw: pytest.fail("watsonx must not be built"))

    assert embed.load_embedding_model() == "model"
    assert captured == {
        "model": "vendor/embed-model:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "check_embedding_ctx_length": False,
        "model_kwargs": {"encoding_format": "float"},
        "chunk_size": embed.OPENROUTER_EMBEDDING_BATCH_SIZE,
        "request_timeout": embed.OPENROUTER_TIMEOUT_SECONDS,
        "max_retries": embed.OPENROUTER_MAX_RETRIES,
    }


def test_missing_openrouter_key_is_named(openrouter_backend, monkeypatch):
    monkeypatch.setattr(embed, "OPENROUTER_API_KEY", "")

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        embed.load_embedding_model()


def test_embedding_model_id_follows_the_backend(monkeypatch):
    monkeypatch.setattr(embed, "OPENROUTER_EMBEDDING_MODEL", "vendor/embed-model")
    for backend, expected in (("watsonx", embed.EMBEDDING_MODEL), ("local", embed.EMBEDDING_MODEL),
                              ("openrouter", "vendor/embed-model")):
        monkeypatch.setattr(embed, "EMBEDDING_BACKEND", backend)
        assert embed.embedding_model_id() == expected


def _openrouter_with(monkeypatch, handler) -> list[httpx.Request]:
    """Route the real OpenAI client through ``handler``; return the requests it saw."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record))
    monkeypatch.setattr(embed, "OpenAIEmbeddings", functools.partial(OpenAIEmbeddings, http_client=client))
    return seen


def _vectors_reply(request: httpx.Request) -> httpx.Response:
    """An OpenRouter-shaped reply: one 2-number vector per input text."""
    texts = json.loads(request.content)["input"]
    data = [{"object": "embedding", "index": i, "embedding": [0.6, 0.8]} for i in range(len(texts))]
    return httpx.Response(200, json={"object": "list", "data": data, "model": "vendor/embed-model"})


def test_openrouter_sends_text_not_token_ids_and_parses_vectors(openrouter_backend, monkeypatch):
    seen = _openrouter_with(monkeypatch, _vectors_reply)
    model = embed.load_embedding_model()

    assert model.embed_documents(["LM317 reference voltage", "TLV1117 dropout"]) == [[0.6, 0.8], [0.6, 0.8]]
    assert model.embed_query("NE555 astable") == [0.6, 0.8]

    first = seen[0]
    assert str(first.url) == "https://openrouter.ai/api/v1/embeddings"
    assert first.headers["authorization"] == "Bearer test-key"
    assert json.loads(first.content) == {
        "input": ["LM317 reference voltage", "TLV1117 dropout"],
        "model": "vendor/embed-model:free",
        "encoding_format": "float",
    }


def test_openrouter_batches_documents(openrouter_backend, monkeypatch):
    monkeypatch.setattr(embed, "OPENROUTER_EMBEDDING_BATCH_SIZE", 2)
    seen = _openrouter_with(monkeypatch, _vectors_reply)

    vectors = embed.load_embedding_model().embed_documents(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert [json.loads(r.content)["input"] for r in seen] == [["a", "b"], ["c", "d"], ["e"]]


def test_openrouter_bad_key_is_not_retried(openrouter_backend, monkeypatch):
    seen = _openrouter_with(monkeypatch, lambda r: httpx.Response(401, json={"error": {"message": "No auth"}}))

    with pytest.raises(openai.AuthenticationError):
        embed.load_embedding_model().embed_query("q")
    assert len(seen) == 1


def test_openrouter_rate_limit_is_retried_then_succeeds(openrouter_backend, monkeypatch):
    replies = iter([httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "slow down"}})])
    seen = _openrouter_with(monkeypatch, lambda r: next(replies, None) or _vectors_reply(r))

    assert embed.load_embedding_model().embed_query("q") == [0.6, 0.8]
    assert len(seen) == 2


def test_granite_keeps_the_original_collection(monkeypatch):
    for backend in ("watsonx", "local"):
        monkeypatch.setattr(embed, "EMBEDDING_BACKEND", backend)
        assert embed.collection_name() == "langchain"


def test_other_models_get_a_collection_named_after_them(openrouter_backend):
    assert embed.collection_name() == "vendor-embed-model"
    assert embed.collection_name("nvidia/nemotron-3.5-embed-8b") == "nvidia-nemotron-3.5-embed-8b"


def test_free_and_paid_variants_share_a_collection():
    assert embed.collection_name("vendor/model:free") == embed.collection_name("vendor/model")
    assert embed.vector_space("vendor/model:free") == "vendor/model"


def test_collection_metadata_records_the_model(openrouter_backend):
    assert embed.collection_metadata() == {
        "hnsw:space": "cosine",
        "embedding_model": "vendor/embed-model",
        "embedding_backend": "openrouter",
    }
