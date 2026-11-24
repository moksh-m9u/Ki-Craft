"""Tests for the LangSmith trace a query produces (no network).

A stub client records the runs ``@traceable`` would upload, so these tests
check the shape of the trace: which runs exist, how they nest, and what each
records as input and output.
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging

import pytest
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langsmith.run_helpers import tracing_context

import src.llm
from src.llm import LLMProviderError, create_llm

from src.retrieval import generate

ANSWER = "Pin 1 is GND and pin 8 is VCC."


class _RecordingClient:
    """Collects run payloads by id instead of sending them to LangSmith."""

    def __init__(self):
        self.runs: dict[str, dict] = {}

    def create_run(self, **run):
        """Record a new run."""
        self.runs[str(run["id"])] = dict(run)

    def update_run(self, run_id, **fields):
        """Merge the fields sent when a run ends."""
        self.runs.setdefault(str(run_id), {}).update(
            {key: value for key, value in fields.items() if value is not None}
        )

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class _FakeRetriever:
    """Returns three plain chunks for any question."""

    def invoke(self, query: str, **kwargs) -> list[Document]:
        """Mimic a LangChain retriever's ``invoke``."""
        return [
            Document(
                page_content=f"summary {i}",
                metadata={"raw_text": "", "tables_html": [], "image_paths": []},
            )
            for i in range(3)
        ]


def _run_query() -> tuple[list, dict[str, dict]]:
    """Consume one traced query, each ``next()`` on a worker thread as Starlette does."""
    client = _RecordingClient()
    llm = GenericFakeChatModel(messages=iter([AIMessage(content=ANSWER)]))
    with tracing_context(enabled=True, client=client, project_name="test"):
        stream = generate.answer_query_stream(_FakeRetriever(), "LM555 pins?", llm, "gemini", "test-model")
        events = []
        with ThreadPoolExecutor(2) as pool:
            while (item := pool.submit(contextvars.copy_context().run, next, stream, None).result()) is not None:
                events.append(item)
    return events, client.runs


def _by_name(runs: dict[str, dict]) -> dict[str, dict]:
    return {run["name"]: run for run in runs.values()}


def test_events_come_in_order():
    events, _ = _run_query()
    kinds = [kind for kind, _ in events]
    assert kinds[:2] == ["chunks", "figures"] and kinds[-1] == "usage"
    assert set(kinds[2:-1]) == {"token"}
    assert "".join(value for kind, value in events if kind == "token") == ANSWER


def test_runs_nest_under_one_answer_query():
    _, runs = _run_query()
    named = _by_name(runs)
    root = named["AnswerQuery"]
    assert root.get("parent_run_id") is None
    assert named["RetrieveChunks"]["parent_run_id"] == root["id"]
    assert named["GenerateAnswerStream"]["parent_run_id"] == root["id"]
    llm_runs = [run for run in runs.values() if run["run_type"] == "llm"]
    assert len(llm_runs) == 1, "only LangChain's own model run is an llm run"
    assert llm_runs[0]["parent_run_id"] == named["GenerateAnswerStream"]["id"]


def test_runs_record_the_answer_not_a_stream_object():
    _, runs = _run_query()
    named = _by_name(runs)
    assert named["AnswerQuery"]["outputs"]["answer"] == ANSWER
    assert named["AnswerQuery"]["outputs"]["usage"]["chunks_used"] == 3
    assert named["GenerateAnswerStream"]["outputs"] == {"answer": ANSWER}
    assert named["GenerateAnswerStream"]["run_type"] == "chain"


def test_root_run_lasts_until_the_last_token():
    _, runs = _run_query()
    named = _by_name(runs)
    assert named["AnswerQuery"]["end_time"] >= named["GenerateAnswerStream"]["end_time"]


def test_inputs_hold_the_question_and_metadata_holds_the_model():
    _, runs = _run_query()
    root = _by_name(runs)["AnswerQuery"]
    assert root["inputs"]["question"] == "LM555 pins?"
    assert "llm" not in root["inputs"] and "model_id" not in root["inputs"]
    assert root["extra"]["metadata"]["model"] == "test-model"
    assert root["extra"]["metadata"]["provider"] == "gemini"
    assert "GenericFakeChatModel" not in json.dumps(root["inputs"], default=str)


def test_a_failed_generation_is_recorded_as_an_error():
    """A provider failure ends both runs with an error, not a successful apology.

    This fake model raises when it has no text to stream, which stands in for
    any failing provider.
    """
    client = _RecordingClient()
    llm = GenericFakeChatModel(messages=iter([AIMessage(content="")]))
    with tracing_context(enabled=True, client=client, project_name="test"):
        stream = generate.answer_query_stream(_FakeRetriever(), "LM555 pins?", llm, "gemini", "test-model")
        with pytest.raises(LLMProviderError, match="The model provider failed"):
            for _ in stream:
                pass
    named = _by_name(client.runs)
    assert "The model provider failed" in named["GenerateAnswerStream"]["error"]
    assert "The model provider failed" in named["AnswerQuery"]["error"]


SECRET_KEY = "sk-test-0123456789abcdef-must-not-leak"

# One streamed OpenAI chat completion, as the provider sends it over SSE.
_OPENAI_STREAM = (
    'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"Pin 1 is GND."},"finish_reason":null}]}\n\n'
    'data: {"id":"c1","object":"chat.completion.chunk","created":0,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


@pytest.mark.parametrize("status", [200, 401])
def test_api_key_never_reaches_the_trace_or_logs(monkeypatch, caplog, status):
    """A real ChatOpenAI client holding the caller's key, over a fake network.

    ``create_llm`` builds the same client a request with an ``llm`` object
    gets; only its HTTP transport is replaced, so LangChain's own model run
    and the OpenAI SDK's debug logs are the real ones. Both a successful
    answer and a rejected key (401) are checked.
    """
    requests: list[httpx.Request] = []

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if status == 401:
            return httpx.Response(401, json={"error": {"message": "Invalid API Key", "code": "invalid_api_key"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_OPENAI_STREAM)

    transport = httpx.Client(transport=httpx.MockTransport(provider))
    monkeypatch.setattr(src.llm, "ChatOpenAI", functools.partial(ChatOpenAI, http_client=transport))
    caplog.set_level(logging.DEBUG)
    client = _RecordingClient()
    llm = create_llm("groq", "openai/gpt-oss-20b", SECRET_KEY)

    with tracing_context(enabled=True, client=client, project_name="test"):
        stream = generate.answer_query_stream(_FakeRetriever(), "LM555 pins?", llm, "groq", "openai/gpt-oss-20b")
        if status == 200:
            assert ("token", "Pin 1 is GND.") in list(stream)
        else:
            with pytest.raises(LLMProviderError):
                list(stream)

    assert requests[0].headers["authorization"] == f"Bearer {SECRET_KEY}", "the key was really used"
    recorded = json.dumps(client.runs, default=str)
    assert "ChatOpenAI" in recorded, "LangChain's own model run was recorded"
    for text in (recorded, caplog.text):
        assert SECRET_KEY not in text
        assert SECRET_KEY[-12:] not in text, "not even a masked tail of the key"
