"""Text and example payloads shown in the interactive API docs (``/docs``).

Shared by the retrieval and query routers. Nothing here affects how a request
is handled; it only shapes the OpenAPI schema FastAPI generates.
"""

from __future__ import annotations

from api.schemas import ErrorResponse, ValidationErrorResponse

# Named examples for request bodies, in the shape FastAPI's
# ``Body(openapi_examples=...)`` expects: each key is one entry in the
# Examples dropdown on /docs, ``summary`` is its label, ``description`` the
# note under it, and ``value`` the payload. Every example spells out every
# setting, so a reader sees the whole body without opening the schema.
_LM2596 = "Tell me about the pin configuration of the LM2596."

RETRIEVE_BODY_EXAMPLES = {
    "lm2596_pins": {
        "summary": "LM2596 pins, 5 chunks",
        "value": {"query": _LM2596, "top_k": 5},
    },
    "lm317_pins": {
        "summary": "LM317, 10 chunks",
        "value": {"query": "Tell me about the pin configuration of the LM317.", "top_k": 10},
    },
}

QUERY_BODY_EXAMPLES = {
    "default_model": {
        "summary": "Default model (Gemini 3.5 Flash-Lite)",
        "description": "Without `llm`, the server's default model answers using the server's key.",
        "value": {"query": _LM2596, "top_k": 5, "max_images": 3},
    },
    "default_model_every_image": {
        "summary": "Default model, 10 chunks, every figure",
        "description": "`max_images` 0 sends every retrieved figure, which costs far more tokens.",
        "value": {"query": "Tell me about the pin configuration of the LM317.", "top_k": 10, "max_images": 0},
    },
    "gemini": {
        "summary": "Gemini 3.5 Flash-Lite, your key",
        "description": "This model uses fixed sampling, so `temperature` is ignored.",
        "value": {
            "query": _LM2596,
            "top_k": 5,
            "max_images": 3,
            "llm": {
                "provider": "gemini",
                "model": "gemini-3.5-flash-lite",
                "api_key": "YOUR_GEMINI_API_KEY",
                "temperature": 0.0,
                "max_tokens": 512,
            },
        },
    },
    "groq_gpt_oss": {
        "summary": "Groq GPT-OSS 120B, your key",
        "description": (
            "Text-only: no figures are sent. It reasons before answering, and the "
            "reasoning counts against `max_tokens`; if the budget runs out before "
            "any answer text, the response is 502."
        ),
        "value": {
            "query": _LM2596,
            "top_k": 5,
            "max_images": 3,
            "llm": {
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "api_key": "YOUR_GROQ_API_KEY",
                "temperature": 0.0,
                "max_tokens": 2048,
            },
        },
    },
    "openai_gpt_4o_mini": {
        "summary": "OpenAI GPT-4o mini, your key",
        "description": "Reads images, so up to `max_images` figures are sent.",
        "value": {
            "query": _LM2596,
            "top_k": 5,
            "max_images": 3,
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "YOUR_OPENAI_API_KEY",
                "temperature": 0.0,
                "max_tokens": 512,
            },
        },
    },
    "huggingface_llama": {
        "summary": "Hugging Face Llama 3.3 70B, your token",
        "description": "Text-only: no figures are sent. The key is a Hugging Face token.",
        "value": {
            "query": _LM2596,
            "top_k": 5,
            "max_images": 3,
            "llm": {
                "provider": "huggingface",
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "api_key": "YOUR_HF_TOKEN",
                "temperature": 0.0,
                "max_tokens": 512,
            },
        },
    },
}

def error_responses(*codes: int) -> dict[int, dict]:
    """Describe a route's error status codes for ``/docs``.

    Every error body is ``ErrorResponse``, ``{"detail": "..."}``, except 422,
    whose body is ``ValidationErrorResponse``; the handlers that produce them
    are in ``api/main.py``. Every route that takes input gets 422, replacing
    FastAPI's default schema, which lists an ``input`` field this API strips.
    """
    descriptions = {
        404: "Not found: no figure at that record and position, or no document with that id.",
        500: "Unexpected server error. Details are in the server log.",
        502: "The model provider failed, or returned no answer text.",
        503: "The database could not be reached.",
    }
    responses = {code: {"model": ErrorResponse, "description": descriptions[code]} for code in codes}
    responses[422] = {"model": ValidationErrorResponse, "description": "The request is invalid."}
    return responses
