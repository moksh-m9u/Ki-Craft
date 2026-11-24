"""Token counts and cost of one answer.

Generation (``src/retrieval/generate.py``) streams the answer and hands each
streamed piece to :func:`capture_usage`, which adds up the tokens the provider
reports; when the stream ends, :func:`estimate_cost` prices those tokens with
the model's catalogue prices (``config/providers.py``). The totals end up in
the ``usage`` object of ``/v1/query`` and of the stream's ``done`` event.
Nothing here calls a model.
"""

from __future__ import annotations

from typing import Any

from config.providers import get_model


def capture_usage(chunk: Any, usage: dict[str, Any]) -> None:
    """Add the token counts on one streamed chunk to the totals in ``usage``.

    LangChain gives every chunk's ``usage_metadata`` as the tokens added by
    that chunk, so the totals are the sum over the stream. Gemini reports
    running totals, which ``ChatGoogleGenerativeAI`` turns into these
    differences; OpenAI-style APIs send the whole count once, on a final
    chunk with no text, and nothing on the others.

    Args:
        chunk: One streamed message chunk from the chat model.
        usage: The running totals, updated in place.
    """
    counts = getattr(chunk, "usage_metadata", None) or {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if isinstance(counts.get(key), int):
            usage[key] = usage.get(key, 0) + counts[key]


def estimate_cost(provider_id: str, model_id: str, usage: dict[str, Any]) -> float | None:
    """Estimate the answer's cost in USD from its token counts and the model's prices.

    Args:
        provider_id: The model's catalogue provider.
        model_id: The model's catalogue id.
        usage: Token totals with ``input_tokens`` and ``output_tokens``.

    Returns:
        The cost rounded to 6 decimals, or None when the model is not in the
        catalogue or both its prices are 0, which the catalogue uses for
        "unknown", so an unknown price is never shown as $0.
    """
    model = get_model(provider_id, model_id)
    if model is None or not (model.input_price_per_1m or model.output_price_per_1m):
        return None
    cost = (
        (usage.get("input_tokens") or 0) / 1_000_000 * model.input_price_per_1m
        + (usage.get("output_tokens") or 0) / 1_000_000 * model.output_price_per_1m
    )
    return round(cost, 6)
