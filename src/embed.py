"""Embedding model factory.

Provides a single function to build the embedding model used by both the
ingestion pipeline (when creating the vector store) and the retrieval path
(when loading it back), guaranteeing both sides use the same configuration.

``EMBEDDING_BACKEND`` picks where the vectors come from:

- ``watsonx`` calls ``ibm/granite-embedding-278m-multilingual`` on IBM
  watsonx.ai. The original store was built with it.
- ``local`` runs a GGUF build of that same Granite model with llama.cpp on this
  machine, so the Granite store can be queried and extended without watsonx
  credentials and without torch. It needs ``uv sync --extra local-embeddings``
  and the model file, so the API Docker image does not include it.
- ``openrouter`` calls ``OPENROUTER_EMBEDDING_MODEL`` through OpenRouter's
  OpenAI-compatible embeddings API. Granite is not offered there, so this is a
  different model.

A query must be embedded into the same vector space as the stored vectors, or
its distances to them mean nothing. Two separately trained models almost never
share a space, even with the same number of dimensions, so each model's
vectors live in their own Chroma collection (see :func:`collection_name`). The
watsonx and local backends share Granite's collection: the GGUF Q8_0 file
scores cosine 0.9999 against the stored watsonx vectors. To check another
candidate for the same space, embed some stored summaries with it and compare
each with its stored vector: a cosine near 1.0 means the same space.

The Granite backends give unit-length vectors and read at most
``EMBEDDING_MAX_INPUT_TOKENS`` tokens per text. The store uses cosine distance,
so vector length does not affect ranking for any backend.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_ibm import WatsonxEmbeddings
from langchain_openai import OpenAIEmbeddings

from config.settings import (
    CHROMA_COLLECTION_METADATA,
    EMBEDDING_BACKEND,
    EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_GGUF_PATH,
    LOCAL_EMBEDDING_THREADS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_BATCH_SIZE,
    OPENROUTER_EMBEDDING_MODEL,
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_TIMEOUT_SECONDS,
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
)
from src.logger import get_logger

logger = get_logger(__name__)


class GGUFEmbeddings(Embeddings):
    """Granite embeddings from a GGUF file, computed in-process by llama.cpp.

    Matches the watsonx model's output: CLS pooling (as Granite's own
    sentence-transformers config uses), at most ``EMBEDDING_MAX_INPUT_TOKENS``
    tokens per text, and unit-length vectors. Calls are serialised with a lock:
    the API embeds each request in its own worker thread, and llama.cpp
    crashes (segmentation fault) when two threads use one model at once.

    Args:
        model_path: Path to the ``.gguf`` file.
        threads: CPU threads llama.cpp may use.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
    """

    def __init__(self, model_path: str, threads: int) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"GGUF embedding model not found at {model_path}; download it as "
                "described in CONTRIBUTING.md or set LOCAL_EMBEDDING_GGUF_PATH"
            )
        # Imported here so the watsonx deployment never needs llama.cpp installed.
        import llama_cpp

        self._llm = llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_CLS,
            # embed() truncates each text to n_batch tokens, so both are the
            # model's input limit, like watsonx's truncate_input_tokens.
            n_ctx=EMBEDDING_MAX_INPUT_TOKENS,
            n_batch=EMBEDDING_MAX_INPUT_TOKENS,
            n_threads=threads,
            verbose=False,
        )
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each text; returns one unit-length 768-number vector per text."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one text; returns a unit-length 768-number vector."""
        with self._lock:
            return self._llm.embed(text, normalize=True, truncate=True)


def _load_local_embedding_model() -> Embeddings:
    """Load the GGUF Granite model configured in settings."""
    try:
        import llama_cpp  # noqa: F401  (checked here so the error says what to do)
    except ImportError as exc:
        raise RuntimeError(
            "EMBEDDING_BACKEND=local needs llama-cpp-python (uv sync --extra local-embeddings). "
            "The API Docker image does not include it: use EMBEDDING_BACKEND=watsonx or "
            "openrouter there, or build the image with --build-arg EMBEDDING_BACKEND=local"
        ) from exc
    logger.info(
        "Loading local embedding model: %s (%d thread(s))",
        LOCAL_EMBEDDING_GGUF_PATH,
        LOCAL_EMBEDDING_THREADS,
    )
    return GGUFEmbeddings(LOCAL_EMBEDDING_GGUF_PATH, LOCAL_EMBEDDING_THREADS)


def _load_openrouter_embedding_model() -> Embeddings:
    """Build an OpenAI-protocol client for the OpenRouter model configured in settings."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OpenRouter credentials: OPENROUTER_API_KEY")
    logger.info("Loading embedding model: %s (OpenRouter)", OPENROUTER_EMBEDDING_MODEL)
    return OpenAIEmbeddings(
        model=OPENROUTER_EMBEDDING_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        # By default the client splits texts into tiktoken token ids, which only
        # OpenAI's own models accept; OpenRouter needs the text itself.
        check_embedding_ctx_length=False,
        # The openai SDK otherwise asks for base64 and decodes it itself.
        model_kwargs={"encoding_format": "float"},
        chunk_size=OPENROUTER_EMBEDDING_BATCH_SIZE,
        request_timeout=OPENROUTER_TIMEOUT_SECONDS,
        max_retries=OPENROUTER_MAX_RETRIES,
    )


def embedding_model_id() -> str:
    """Return the model whose vector space the configured backend produces.

    The watsonx and local backends both give Granite vectors, so both return
    ``EMBEDDING_MODEL``; OpenRouter returns ``OPENROUTER_EMBEDDING_MODEL``.
    """
    return OPENROUTER_EMBEDDING_MODEL if EMBEDDING_BACKEND == "openrouter" else EMBEDDING_MODEL


# The original store's collection, made by Granite, keeps LangChain's default
# name, so adding collections for other models renamed nothing.
GRANITE_COLLECTION = "langchain"


def vector_space(model_id: str) -> str:
    """Return the model behind ``model_id``, without an OpenRouter variant suffix.

    ``nvidia/nemotron-3-embed-1b:free`` and ``nvidia/nemotron-3-embed-1b`` are
    the same model at two price tiers, so they give the same vectors.
    """
    return model_id.split(":", 1)[0]


def collection_name(model_id: str | None = None) -> str:
    """Return the Chroma collection that holds ``model_id``'s vectors.

    Ingestion and retrieval both call this, so they always agree on which
    collection belongs to the configured model.

    Args:
        model_id: An embedding model id; the configured one when omitted.

    Returns:
        ``langchain`` for Granite (the original collection); otherwise the
        model id with every character Chroma does not allow in a name turned
        into ``-``, for example ``nvidia-nemotron-3-embed-1b``.
    """
    space = vector_space(model_id or embedding_model_id())
    if space == EMBEDDING_MODEL:
        return GRANITE_COLLECTION
    return re.sub(r"[^A-Za-z0-9._-]", "-", space)


def collection_metadata() -> dict:
    """Return the metadata a new collection is created with for the configured model.

    ``embedding_model`` lets :func:`src.retrieval.search.load_vector_store`
    refuse to search the collection with another model. ``embedding_backend``
    records which service wrote it, for information only.
    """
    return {
        **CHROMA_COLLECTION_METADATA,
        "embedding_model": vector_space(embedding_model_id()),
        "embedding_backend": EMBEDDING_BACKEND,
    }


def load_embedding_model() -> Embeddings:
    """Create the embedding model configured in settings.

    ``EMBEDDING_BACKEND`` picks the backend: watsonx (the default), local (the
    GGUF Granite model on this machine, see :class:`GGUFEmbeddings`) or
    openrouter.

    Inputs longer than ``EMBEDDING_MAX_INPUT_TOKENS`` are truncated by
    watsonx; without this, watsonx rejects the whole request with a 400.

    Returns:
        A configured embeddings object (:class:`WatsonxEmbeddings`,
        :class:`GGUFEmbeddings` or ``OpenAIEmbeddings`` pointed at OpenRouter)
        ready to embed text or load a vector store.

    Raises:
        RuntimeError: If the chosen backend's credentials are not set, so the
            failure names the missing variable instead of surfacing as an
            opaque authentication error; or if the local backend is chosen
            and llama-cpp-python is not installed.
        FileNotFoundError: If the local backend is chosen and the GGUF file
            is missing.
    """
    if EMBEDDING_BACKEND == "local":
        return _load_local_embedding_model()
    if EMBEDDING_BACKEND == "openrouter":
        return _load_openrouter_embedding_model()

    missing = [
        name
        for name, value in (
            ("WATSONX_API_KEY", WATSONX_API_KEY),
            ("WATSONX_PROJECT_ID", WATSONX_PROJECT_ID),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing watsonx.ai credentials: {', '.join(missing)}")

    logger.info("Loading embedding model: %s (%s)", EMBEDDING_MODEL, WATSONX_URL)
    return WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL,
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
        api_key=WATSONX_API_KEY,
        params={"truncate_input_tokens": EMBEDDING_MAX_INPUT_TOKENS},
    )
