"""Answer generation: retrieve, build the prompt, stream the model's answer.

The query path's last step. ``answer_query_stream`` retrieves chunks
(``src/retrieval/search.py``), chooses figures and builds the messages
(``src/retrieval/prompt.py``), then streams the answer from the chat model
it is given; ``src/retrieval/usage.py`` counts its tokens and prices it.
Retrieval and generation run under one LangSmith trace, so a query shows up
as one run.
The default model and a user-chosen model share this one path; only the
``llm`` passed in differs.
"""

from __future__ import annotations

import time
from collections.abc import Generator, Iterator, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langsmith import traceable

from config.settings import GENERATION_MAX_IMAGES
from src.chunk import Chunk
from src.llm import LLMProviderError
from src.logger import get_logger
from src.retrieval.prompt import PromptFigure, build_messages, prompt_stats, select_figures
from src.retrieval.search import retrieve_chunks
from src.retrieval.usage import capture_usage, estimate_cost

logger = get_logger(__name__)

# One item from answer_query_stream: ("chunks", list[Chunk]),
# ("figures", list[PromptFigure]), ("token", str) or ("usage", dict).
StreamEvent = tuple[str, Any]


def _extract_text(content: Any) -> str:
    """Extract plain text from LLM response content (string or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


# What providers report when a reply stopped at max_tokens: OpenAI-style
# APIs (OpenAI, Groq, Hugging Face) say "length", Gemini says "MAX_TOKENS".
_OUT_OF_BUDGET = {"length", "MAX_TOKENS"}


def _empty_answer_message(finish_reason: str | None, usage: dict[str, Any]) -> str:
    """Explain why the model returned no answer text."""
    if finish_reason in _OUT_OF_BUDGET:
        return (
            f"The model used its whole output budget ({usage.get('output_tokens', '?')} "
            "tokens, reasoning included) before writing any answer text. "
            "Raise max_tokens and try again."
        )
    return f"The model returned no answer text (finish_reason: {finish_reason or 'not reported'})."


def _attach_figures_to_trace(run_tree, figures: list[PromptFigure]) -> None:
    """Attach the figures sent to the model to ``run_tree``, named by label."""
    if run_tree is None or not figures:
        return
    run_tree.attachments = {
        figure.label: ("image/jpeg", figure.path.read_bytes()) for figure in figures
    }


def _trace_inputs(inputs: dict) -> dict:
    """Choose what LangSmith records as a generation run's inputs.

    The retriever, the chunks and the chat model object are left out: they
    serialise to long reprs, and the model object carries the caller's key.
    The question is the only text input, so the LangSmith runs table previews
    it; the model id goes into run metadata instead (see
    :func:`answer_query_stream`).
    """
    recorded = {"question": inputs.get("query")}
    for key in ("max_images", "send_images"):
        if key in inputs:
            recorded[key] = inputs[key]
    if "chunks" in inputs:
        recorded["num_chunks"] = len(inputs["chunks"])
    if "figures" in inputs:
        recorded["figures"] = [figure.label for figure in inputs["figures"]]
    return recorded


def _answer_output(tokens: list[str]) -> dict:
    """Turn the tokens a generation run yielded into its recorded output."""
    return {"answer": "".join(tokens)}


def _query_output(events: list[StreamEvent]) -> dict:
    """Turn the events a query run yielded into its recorded output.

    The chunks are left out: the nested ``RetrieveChunks`` run records them.
    """
    return {
        "answer": "".join(value for kind, value in events if kind == "token"),
        "usage": next((value for kind, value in events if kind == "usage"), {}),
    }


@traceable(
    run_type="chain",
    name="AnswerQuery",
    process_inputs=_trace_inputs,
    reduce_fn=_query_output,
    dangerously_allow_filesystem=True,
)
def answer_query_stream(
    retriever,
    query: str,
    llm: BaseChatModel,
    provider_id: str,
    model_id: str,
    *,
    max_images: int = GENERATION_MAX_IMAGES,
    send_images: bool = True,
    run_tree=None,
) -> Generator[StreamEvent, None, None]:
    """Retrieve chunks then stream an answer under a single LangSmith trace.

    This is the one generation path: the default model and a user-chosen
    model both come through here, differing only in the ``llm`` passed in.
    It is a generator, so LangSmith keeps the ``AnswerQuery`` run open until
    the last event and records the full answer as its output. ``run_tree`` is
    injected by LangSmith.

    Args:
        retriever: Vector store retriever.
        query: The user's question.
        llm: The chat model to answer with (see ``src/llm.py``).
        provider_id: The model's catalogue provider, used to price the answer.
        model_id: The model's catalogue id, recorded in ``usage``.
        max_images: Most images to send to the model (0 = all).
        send_images: False for a text-only model; no figures are sent.

    Yields:
        ``("chunks", list[Chunk])`` once, as soon as retrieval finishes;
        ``("figures", list[PromptFigure])`` once, the figures the prompt
        carries (empty for a text-only model); then ``("token", str)`` for
        each piece of the answer; then
        ``("usage", dict)`` once, with token counts, latency and prompt size.

    Raises:
        LLMProviderError: From ``generate_answer_stream``, if the provider
            fails or writes no answer text.
    """
    if run_tree is not None:
        run_tree.add_metadata({"provider": provider_id, "model": model_id})
    chunks = retrieve_chunks(retriever, query)
    yield "chunks", chunks
    figures = select_figures(chunks, max_images) if send_images else []
    _attach_figures_to_trace(run_tree, figures)
    yield "figures", figures

    usage: dict[str, Any] = {}
    for token in generate_answer_stream(
        chunks, query, llm, provider_id, model_id, figures=figures, usage=usage
    ):
        yield "token", token
    yield "usage", usage


@traceable(
    run_type="chain",
    name="GenerateAnswerStream",
    process_inputs=_trace_inputs,
    reduce_fn=_answer_output,
)
def generate_answer_stream(
    chunks: list[Chunk],
    query: str,
    llm: BaseChatModel,
    provider_id: str,
    model_id: str,
    *,
    figures: Sequence[PromptFigure] = (),
    usage: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Stream an answer from ``llm`` for already retrieved chunks.

    Traced as a ``chain`` run that stays open until the last token. The model
    call itself appears under it as LangChain's own ``llm`` run, which carries
    the provider, model name and token counts.

    Args:
        chunks: Retrieved document chunks.
        query: The original user query.
        llm: The chat model to answer with.
        provider_id: The model's catalogue provider, used to price the answer.
        model_id: The model's catalogue id, recorded in ``usage``.
        figures: The figures to send, from :func:`select_figures`; none
            for a text-only model.
        usage: Optional shared dict filled in place with token counts,
            latency and prompt size as the stream runs; a fresh one is used
            when omitted.

    Yields:
        Answer token strings.

    Raises:
        LLMProviderError: If the provider call fails, or the model finishes
            without writing any answer text (for example, it spent its whole
            ``max_tokens`` budget on reasoning).
    """
    if usage is None:
        usage = {}

    messages = build_messages(chunks, query, list(figures))

    prompt_chars, prompt_images = prompt_stats(messages[1].content)
    prompt_chars += len(messages[0].content)
    usage.update(
        {
            "model": model_id,
            "prompt_chars": prompt_chars,
            "prompt_images": prompt_images,
            "chunks_used": len(chunks),
        }
    )

    logger.info(
        "Prompt: %d chunks, %d figures, %d characters of text",
        len(chunks),
        prompt_images,
        prompt_chars,
    )

    start = time.perf_counter()
    wrote_text = False
    finish_reason = None
    try:
        for chunk in llm.stream(messages):
            token_text = _extract_text(chunk.content)
            if token_text:
                wrote_text = True
                yield token_text
            capture_usage(chunk, usage)
            finish_reason = (chunk.response_metadata or {}).get("finish_reason") or finish_reason
    except Exception as exc:
        logger.warning("Model provider failed: %s", type(exc).__name__)
        raise LLMProviderError(f"The model provider failed: {exc}") from exc
    usage["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    usage["estimated_cost_usd"] = estimate_cost(provider_id, model_id, usage)

    if not wrote_text:
        raise LLMProviderError(_empty_answer_message(finish_reason, usage))

