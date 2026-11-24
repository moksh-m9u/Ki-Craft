"""Fakes and data shared by the API tests in this folder.

The real FastAPI app and its lifespan run, but the heavy things are replaced
before the app starts (see ``conftest.py:backend``):

* the watsonx embedding client, the Postgres engine and the reads that use
  it (patched in ``api.service``, where ``RAGService`` looks them up): the
  retriever, ``list_documents``, ``document_chunks`` and ``figure_path``
  answer from the in-memory records below;
* the two chat client classes inside ``src/llm.py``. The real ``create_llm``
  still runs, so its catalogue checks are part of every test.

Guards make any attempt to create a real database engine or build a real watsonx client fail the test, and LangSmith tracing is switched
off, so no test touches Neon, ``dbv2/`` or the network.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessageChunk

from src.browse import DocumentSummary
from src.chunk import Chunk

# Starts with the JPEG magic bytes so it looks like a real figure.
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg-for-tests"

LM317_ID = "11111111-1111-4111-8111-111111111111"
TLV1117_ID = "22222222-2222-4222-8222-222222222222"
# A record whose stored path climbs out of the images folder.
ESCAPING_ID = "33333333-3333-4333-8333-333333333333"
# The first LM317 chunk, stored after the second one, to check browsing sorts.
LM317_FIRST_ID = "44444444-4444-4444-8444-444444444444"


def _fake_chunks() -> list[Chunk]:
    """Two stored chunks: one with a table and no figure, one with a figure file."""
    lm317 = Chunk(
        record_id=LM317_ID,
        source="lm317",
        chunk_id=2,
        summary="LM317 summary: adjustable regulator with ADJ, VOUT, VIN pins.",
        raw_text="Pin 1 ADJ, pin 2 VOUT, pin 3 VIN.",
        tables_html=["<table><tr><td>ADJ</td></tr></table>"],
    )
    tlv1117 = Chunk(
        record_id=TLV1117_ID,
        source="tlv1117",
        chunk_id=1,
        summary="TLV1117 summary: fixed 3.3 V LDO.",
        raw_text="Dropout 1.1 V at 800 mA.",
        image_paths=["dbv2/images/tlv1117_c1_img0.jpg"],
    )
    return [lm317, tlv1117]


def _stored_chunks() -> list[Chunk]:
    """Every record in the fake store: the retrievable chunks plus LM317's first chunk."""
    lm317_first = Chunk(
        record_id=LM317_FIRST_ID,
        source="lm317",
        chunk_id=1,
        summary="LM317 overview: 1.2 V to 37 V adjustable regulator.",
    )
    return _fake_chunks() + [lm317_first]


class _FakeRetriever:
    """Returns the first ``k`` fake chunks for any question."""

    def __init__(self, k: int):
        self.k = k

    def invoke(self, query: str) -> list[Chunk]:
        """Mimic a LangChain retriever's ``invoke``."""
        return _fake_chunks()[: self.k]


class FakeStore:
    """Stands in for the Postgres reads in ``src/browse.py`` over the records above."""

    @staticmethod
    def _records() -> dict:
        """Every stored record by id, plus one whose figure path climbs out of the images folder."""
        records = {chunk.record_id: chunk for chunk in _stored_chunks()}
        records[ESCAPING_ID] = Chunk(
            record_id=ESCAPING_ID,
            source="escape",
            chunk_id=1,
            summary="",
            image_paths=["dbv2/images/../../secret.txt"],
        )
        return records

    def list_documents(self, engine) -> list[DocumentSummary]:
        """Mimic ``list_documents``: counts per source, sorted by name."""
        counts: dict[str, list[int]] = {}
        for chunk in _stored_chunks():
            n, images, tables = counts.get(chunk.source, [0, 0, 0])
            counts[chunk.source] = [n + 1, images + len(chunk.image_paths), tables + len(chunk.tables_html)]
        return [DocumentSummary(source, *counts[source]) for source in sorted(counts)]

    def document_chunks(self, engine, document_id: str, offset: int, limit: int) -> tuple[list[Chunk], int]:
        """Mimic ``document_chunks``: one document's chunks by ``chunk_id``, paged."""
        found = sorted(
            (chunk for chunk in _stored_chunks() if chunk.source == document_id),
            key=lambda chunk: chunk.chunk_id,
        )
        return found[offset : offset + limit], len(found)

    def figure_path(self, engine, record_id: str, position: int) -> str | None:
        """Mimic ``figure_path``: the stored path, or None."""
        record = self._records().get(record_id)
        paths = record.image_paths if record else []
        return paths[position] if 0 <= position < len(paths) else None


class FakeBackend:
    """Switches the tests flip to make the fake store or LLM fail.

    ``prompts`` records every message list sent to the fake LLM so a test can
    check what the model would have seen.
    """

    def __init__(self):
        self.store_error: str | None = None
        self.llm_error: str | None = None
        self.llm_error_after_first_token: str | None = None
        self.finish_reason: str | None = None
        # The second streamed token; a test sets it to put citations in the answer.
        self.second_token = " world"
        self.prompts: list = []
        self.client_kwargs: list[dict] = []

    def ping(self, engine) -> None:
        """Replacement for ``src/db/connection.py:ping``; fails when ``store_error`` is set."""
        if self.store_error:
            raise RuntimeError(self.store_error)

    @staticmethod
    def retriever(engine, embedding_model, k: int) -> _FakeRetriever:
        """Replacement for ``PostgresRetriever``."""
        return _FakeRetriever(k)

    def make_llm(self, *args, **kwargs) -> "_FakeLLM":
        """Replacement for both chat client classes; records what it was given."""
        self.client_kwargs.append(kwargs)
        return _FakeLLM(self)


class _FakeLLM:
    """Streams two tokens, then reports usage on the last chunk like Gemini.

    With ``backend.finish_reason`` set it writes no text at all and stops with
    that reason, like a reasoning model that spent its whole budget.
    """

    def __init__(self, backend: FakeBackend):
        self._backend = backend

    def stream(self, messages):
        """Mimic ``BaseChatModel.stream``."""
        self._backend.prompts.append(messages)
        if self._backend.llm_error:
            raise RuntimeError(self._backend.llm_error)
        if self._backend.finish_reason:
            yield AIMessageChunk(
                content="",
                usage_metadata={"input_tokens": 10, "output_tokens": 512, "total_tokens": 522},
                response_metadata={"finish_reason": self._backend.finish_reason},
            )
            return
        yield AIMessageChunk(content="Hello")
        if self._backend.llm_error_after_first_token:
            raise RuntimeError(self._backend.llm_error_after_first_token)
        yield AIMessageChunk(
            content=self._backend.second_token,
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


def forbidden(name: str):
    """Build a stand-in that fails the test if the real ``name`` is constructed."""

    def _raise(*args, **kwargs):
        raise AssertionError(f"offline tests must not construct {name}")

    return _raise


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a text/event-stream body into ``(event, data)`` pairs."""
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def llm_body(**overrides) -> dict:
    """An ``llm`` object for a request body, Groq GPT-OSS 20B unless overridden."""
    return {"provider": "groq", "model": "openai/gpt-oss-20b", "api_key": "test-key", **overrides}
