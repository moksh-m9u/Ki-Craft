"""Answer endpoints: retrieve chunks, then generate an answer from them.

``/v1/query`` returns the whole answer at once and ``/v1/query/stream`` sends
it token by token over Server-Sent Events. Both use the server's default
model, or the provider, model and API key in the body's
optional ``llm`` object. The paths below are short; ``api/main.py`` adds the
``/v1`` prefix.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Generator
from contextlib import closing
from typing import Annotated

from anyio import CancelScope
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from api.openapi_examples import QUERY_BODY_EXAMPLES, error_responses
from api.responses import citations_payload, figure_payloads, retrieve_response
from api.schemas import LLMSettings, QueryRequest, QueryResponse
from api.service import RAGService
from src.llm import LLMProviderError, UnknownModelError, create_llm
from src.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["query"])


def _requested_llm(llm: LLMSettings | None) -> BaseChatModel | None:
    """Build the requested model, or leave model selection to the service."""
    if llm is None:
        return None
    try:
        return create_llm(
            llm.provider,
            llm.model,
            llm.api_key.get_secret_value(),
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
        )
    except UnknownModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Answer a question",
    description=(
        "Retrieves the `top_k` most relevant chunks and returns them with an "
        "answer written from their text, tables and up to `max_images` figures.\n\n"
        "The prompt labels chunks `C1`, `C2`, ... (the `chunk_id` in `chunks`) and "
        "images `IMG1`, `IMG2`, ...; the answer cites them in square brackets, as "
        "`[C1]` or `[IMG1]`. `figures` lists the images sent to the model, and "
        "`citations` the chunks and figures the answer cites, checked against "
        "the prompt: a cited label that matches neither is listed in `unknown`.\n\n"
        "Without `llm` the server's default model answers. With `llm` the given "
        "provider and model answer using your `api_key`. A provider or model "
        "not in the server's catalogue returns 422; a provider failure (bad key, "
        "quota, timeout, empty answer) returns 502."
    ),
    response_description="The retrieved chunks, the answer and the figures it drew on.",
    responses=error_responses(502, 503),
)
async def query(
    request: Request,
    req: Annotated[QueryRequest, Body(openapi_examples=QUERY_BODY_EXAMPLES)],
) -> QueryResponse:
    """Retrieve chunks and generate a full answer.

    Raises:
        HTTPException: 422 when ``llm`` names a provider or model that is
            not in the catalogue.
    """
    service = request.app.state.service
    answer, chunks, figures, usage = await service.answer(
        req.query,
        req.top_k,
        req.max_images,
        llm=_requested_llm(req.llm),
        provider_id=req.llm.provider if req.llm else None,
        model_id=req.llm.model if req.llm else None,
    )
    base = retrieve_response(req.query, chunks)
    sent = figure_payloads(figures)
    return QueryResponse(
        **base.model_dump(),
        answer=answer,
        answer_characters=len(answer),
        figures=sent,
        citations=citations_payload(answer, base.num_chunks, sent),
        usage=usage,
    )


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_frames(
    service: RAGService,
    req: QueryRequest,
    llm: BaseChatModel | None,
) -> Generator[str, None, None]:
    """Format the service events as SSE frames.

    The ``retrieval`` frame goes out as soon as the chunks are found, before
    the model is called, so a client can show the evidence while the model
    is still thinking. Retrieval and store failures happen before that first
    frame and so are still HTTP errors; a model failure at any point after
    it, before or after the first token, ends the stream with an ``error``
    event (the 200 status has already been sent).
    """
    events = service.stream_answer(
        req.query,
        req.top_k,
        req.max_images,
        llm=llm,
        provider_id=req.llm.provider if req.llm else None,
        model_id=req.llm.model if req.llm else None,
    )
    with closing(events):
        _, chunks = next(events)
        _, figures = next(events)
        retrieval = retrieve_response(req.query, chunks)
        sent = figure_payloads(figures)
        yield _sse("retrieval", retrieval.model_dump())
        tokens: list[str] = []
        try:
            for kind, value in events:
                if kind == "token":
                    tokens.append(value)
                    yield _sse("token", {"delta": value})
                elif kind == "usage":
                    answer = "".join(tokens)
                    yield _sse(
                        "done",
                        {
                            "answer": answer,
                            "answer_characters": len(answer),
                            "num_chunks": retrieval.num_chunks,
                            "figures": [figure.model_dump() for figure in sent],
                            "citations": citations_payload(answer, retrieval.num_chunks, sent).model_dump(),
                            "usage": value,
                        },
                    )
        except LLMProviderError as exc:
            yield _sse("error", {"status": 502, "message": str(exc)})
        except Exception:
            logger.exception("Unexpected failure while streaming an answer")
            yield _sse("error", {"status": 500, "message": "Internal server error"})


@router.post(
    "/query/stream",
    response_class=StreamingResponse,
    summary="Answer a question, streamed",
    description=(
        "Same request body as `POST /v1/query`. The answer arrives as "
        "Server-Sent Events:\n\n"
        "| Event | Data |\n"
        "|---|---|\n"
        "| `retrieval` | The retrieved chunks, as `POST /v1/retrieve` returns them |\n"
        "| `token` | `{\"delta\": \"...\"}`, one per answer fragment |\n"
        "| `done` | `answer`, `answer_characters`, `num_chunks`, `figures`, `citations`, `usage` |\n"
        "| `error` | `{\"status\": 502, \"message\": \"...\"}`, in place of `done` |\n\n"
        "`retrieval` is sent as soon as the chunks are found, before the model is "
        "called. Errors before it (an invalid `llm`, a store that cannot load) are "
        "ordinary HTTP errors (422, 503). A model failure after it, even before the "
        "first `token`, ends the stream with an `error` event; use `POST /v1/query` "
        "if you need a provider failure as an HTTP 502. Swagger UI shows "
        "the stream only once it ends; use `curl -N` or `fetch` to read it live."
    ),
    response_description="A `text/event-stream` of the events above.",
    responses={200: {"content": {"text/event-stream": {}}}, **error_responses(500, 503)},
)
async def query_stream(
    request: Request,
    req: Annotated[QueryRequest, Body(openapi_examples=QUERY_BODY_EXAMPLES)],
) -> StreamingResponse:
    """Take a question and model options and stream the answer over SSE.

    An invalid model raises HTTPException (422). Store and retrieval errors
    before the first frame reach the application's HTTP error handlers; model
    errors arrive later, as ``error`` events.
    """
    service = request.app.state.service
    llm = _requested_llm(req.llm)
    await service.ensure_ready()
    frames = _stream_frames(service, req, llm)
    # StreamingResponse sends HTTP headers before it reads the body iterator,
    # so the retrieval frame is built first: a store failure is still an HTTP error.
    first_frame = await run_in_threadpool(next, frames)

    async def generate() -> AsyncIterator[str]:
        """Send the prepared frame and continue the generator off the event loop."""
        try:
            yield first_frame
            async for frame in iterate_in_threadpool(frames):
                yield frame
        finally:
            with CancelScope(shield=True):
                await run_in_threadpool(frames.close)

    return StreamingResponse(generate(), media_type="text/event-stream")
