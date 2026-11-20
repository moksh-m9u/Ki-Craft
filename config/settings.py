"""Central configuration and environment variable loading.

All tunable knobs for the pipeline live here so they can be changed from a
single place.  Secrets are read from the ``.env`` file (loaded once at import)
and may be overridden by real environment variables.

Module docstrings keep the file self-explanatory; see each section below for
the pipeline stage it controls.
"""

from __future__ import annotations

import os
from typing import Literal, get_args

from dotenv import load_dotenv

# Load secrets from a .env file if present (safe to call multiple times).
load_dotenv()

# LangSmith tracing has one switch here: LANGSMITH_TRACING, falling back to the
# legacy LANGCHAIN_TRACING_V2. The langsmith library reads *_TRACING_V2 before
# *_TRACING, so without this a LANGCHAIN_TRACING_V2=true in .env would win over
# LANGSMITH_TRACING=false. Writing the decision to LANGSMITH_TRACING_V2, the
# name langsmith checks first, makes it final.
LANGSMITH_TRACING: bool = (
    (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "false").strip().lower() == "true"
)
os.environ["LANGSMITH_TRACING_V2"] = "true" if LANGSMITH_TRACING else "false"


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    """Read ``name`` from the environment and check it is one of ``allowed``.

    Raises:
        ValueError: If the variable is set to a value not in ``allowed``, so a
            typo stops the process at startup instead of failing per query.
    """
    value = os.getenv(name, default)
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not one of: {', '.join(allowed)}")
    return value

# API credentials (loaded from environment / .env)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
# Regional watsonx.ai endpoint; must be the region the project lives in.
WATSONX_URL: str = os.getenv("WATSONX_URL", "https://eu-de.ml.cloud.ibm.com")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# Model selection
# Embedding model (served by watsonx.ai) used to vectorise chunks during
# ingestion and retrieval.  Changing it requires re-indexing the store.
EMBEDDING_MODEL: str = "ibm/granite-embedding-278m-multilingual"
# The model's context limit; longer inputs are truncated by watsonx.
EMBEDDING_MAX_INPUT_TOKENS: int = 512
# "watsonx" calls the hosted service; "local" runs the same model's open weights
# on this machine (for retrieval without watsonx credentials). Both give Granite
# vectors. "openrouter" calls OPENROUTER_EMBEDDING_MODEL, a different model, so
# its vectors live in their own collection of the store (see src/embed.py).
EMBEDDING_BACKEND: str = _env_choice("EMBEDDING_BACKEND", "watsonx", ("watsonx", "local", "openrouter"))
# OpenRouter speaks the OpenAI embeddings protocol at this address.
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
# Any model listed by GET https://openrouter.ai/api/v1/embeddings/models.
# Changing it means building that model's collection (python -m scripts.reembed).
OPENROUTER_EMBEDDING_MODEL: str = os.getenv("OPENROUTER_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")
# Texts per request. Free models allow 50 requests a day, so the 310 stored
# summaries are sent in 5 requests rather than 310.
OPENROUTER_EMBEDDING_BATCH_SIZE: int = int(os.getenv("OPENROUTER_EMBEDDING_BATCH_SIZE", "64"))
# One request's time limit, and how often the openai client retries a timeout,
# connection error, 408, 429 or 5xx (with exponential backoff) before failing.
OPENROUTER_TIMEOUT_SECONDS: float = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))
OPENROUTER_MAX_RETRIES: int = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
# GGUF build of the model above, run by llama.cpp (no torch). Q8_0 matches the
# stored vectors at cosine 0.9999; see CONTRIBUTING.md for the download.
LOCAL_EMBEDDING_GGUF_PATH: str = os.getenv(
    "LOCAL_EMBEDDING_GGUF_PATH", "models/granite-embedding-278m-multilingual-Q8_0.gguf"
)
# CPU threads for the local embedder. One keeps a small server (for example
# 0.1 vCPU) from oversubscribing its CPU share.
LOCAL_EMBEDDING_THREADS: int = int(os.getenv("LOCAL_EMBEDDING_THREADS", "1"))
# Default provider and model used to answer retrieval queries.  Both must be
# listed in config/providers.py; the API key is GEMINI_API_KEY.
GENERATION_PROVIDER: str = "gemini"
GENERATION_MODEL: str = "gemini-3.5-flash-lite"

# Storage locations
CHROMA_PERSIST_DIR: str = "dbv2/chroma_db"
IMAGES_DIR: str = "dbv2/images"
# One chunk export per ingested document, <source>.json.
JSON_EXPORT_DIR: str = "json"

# Retrieval settings
# How the vector store picks chunks for a question. These are the three
# search types LangChain's ``as_retriever`` accepts:
#   similarity                  the k nearest chunks by cosine similarity
#   mmr                         fetch RETRIEVAL_FETCH_K nearest, keep k that are
#                               relevant and different from each other
#   similarity_score_threshold  the nearest chunks scoring at least
#                               RETRIEVAL_SCORE_THRESHOLD, at most k; can
#                               return none
# Every value below can be overridden by an environment variable of the same
# name, so an evaluation run can compare them without editing code.
RetrievalSearchType = Literal["similarity", "mmr", "similarity_score_threshold"]
RETRIEVAL_SEARCH_TYPES: tuple[str, ...] = get_args(RetrievalSearchType)
RETRIEVAL_SEARCH_TYPE: RetrievalSearchType = _env_choice(  # type: ignore[assignment]
    "RETRIEVAL_SEARCH_TYPE", "mmr", RETRIEVAL_SEARCH_TYPES
)
# Number of chunks to return. Every returned chunk's text and tables go into
# the prompt, so this is also the model's context size. A request's top_k
# overrides it.
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
# mmr only: candidate pool the k chunks are chosen from.
RETRIEVAL_FETCH_K: int = int(os.getenv("RETRIEVAL_FETCH_K", "20"))
# mmr only: 1 ranks purely by relevance, 0 purely by diversity. 0.5 is
# LangChain's default, so setting it here changes nothing until overridden.
RETRIEVAL_LAMBDA_MULT: float = float(os.getenv("RETRIEVAL_LAMBDA_MULT", "0.5"))
# similarity_score_threshold only: minimum relevance score (0 to 1; with
# cosine distance the score is 1 - distance).
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))

# Generation settings
# Sent only to models whose catalogue entry accepts a temperature.  The
# default model (gemini-3.5-flash-lite) uses fixed sampling, so this value
# does not reach it.
GENERATION_TEMPERATURE: float = 0.0

# Most figures sent to the model per question, taken in chunk order (all of
# chunk 1's, then chunk 2's, ...). Each image costs far more input tokens
# than a chunk's text, so this is the main cost lever. 0 sends every image.
# A request's max_images overrides it.
GENERATION_MAX_IMAGES: int = int(os.getenv("GENERATION_MAX_IMAGES", "3"))

# Maximum output tokens for the generation model.  Limits cost and prevents
# runaway responses.  512 is a good balance for concise technical answers;
# increase if answers get truncated.  Override via env.
GENERATION_MAX_TOKENS: int = int(os.getenv("GENERATION_MAX_TOKENS", "512"))

# PDF extraction settings (unstructured partition_pdf)
PDF_STRATEGY: str = "hi_res"
PDF_EXTRACT_IMAGE_BLOCK_TYPES: list[str] = ["Image"]
PDF_INFER_TABLE_STRUCTURE: bool = True
# An image whose OCR words repeat on this many pages is a logo, not a figure.
PDF_DECORATION_MIN_PAGES: int = 3
# A short image whose OCR text contains one of these words is a vendor logo,
# even on fewer pages (OCR sometimes reads only part of it, as "instruments").
# Whole words, lowercase; comma-separated in the environment.
PDF_LOGO_WORDS: tuple[str, ...] = tuple(
    word.strip().lower()
    for word in os.getenv(
        "PDF_LOGO_WORDS",
        "texas,instruments,microchip,onsemi,nxp,infineon,renesas,stmicroelectronics",
    ).split(",")
    if word.strip()
)

# Chunking settings (unstructured chunk_by_title)
CHUNK_MAX_CHARACTERS: int = 3000
CHUNK_NEW_AFTER_N_CHARS: int = 2400
CHUNK_COMBINE_UNDER_N_CHARS: int = 500
CHUNK_ISOLATE_TABLES: bool = False

# Enhancement (AI summary) settings
# Which vision model writes the chunk summaries during ingestion. Both speak
# the OpenAI chat protocol:
#   huggingface: GLM-4.5V through the Hugging Face router (needs HF_TOKEN and credit);
#   local: a model served on this machine by `python -m mlx_vlm.server`.
ENHANCEMENT_BACKEND: str = _env_choice("ENHANCEMENT_BACKEND", "huggingface", ("huggingface", "local"))
ENHANCEMENT_BACKENDS: dict[str, dict[str, str]] = {
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "model": "zai-org/GLM-4.5V",
    },
    "local": {
        "base_url": os.getenv("LOCAL_ENHANCEMENT_BASE_URL", "http://127.0.0.1:8080/v1"),
        "model": os.getenv("LOCAL_ENHANCEMENT_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit"),
    },
}
ENHANCEMENT_TEMPERATURE: float = float(os.getenv("ENHANCEMENT_TEMPERATURE", "0.0"))
# The embedder reads only the first EMBEDDING_MAX_INPUT_TOKENS (512) of a
# summary, so a longer one costs generation time and is never searched.
ENHANCEMENT_MAX_TOKENS: int = int(os.getenv("ENHANCEMENT_MAX_TOKENS", "512"))
# Seconds one summary call may take before it is retried. A 4B model on a
# laptop can need a minute or more for a chunk with several figures.
ENHANCEMENT_TIMEOUT_SECONDS: float = float(os.getenv("ENHANCEMENT_TIMEOUT_SECONDS", "300"))
# Retries after a timeout, connection error, 429 or 5xx, with exponential
# backoff (done by the OpenAI client). After the last one the document fails.
ENHANCEMENT_MAX_RETRIES: int = int(os.getenv("ENHANCEMENT_MAX_RETRIES", "2"))
# Most figures sent with one chunk's summary call (0 = all). Each figure costs
# prompt tokens and, on a local model, memory.
ENHANCEMENT_MAX_IMAGES: int = int(os.getenv("ENHANCEMENT_MAX_IMAGES", "0"))
# Local backend only; sent as extra request fields that mlx_vlm.server reads.
# Largest image area in pixels; bigger figures are downscaled before encoding.
# 1280 * 28 * 28 is the cap Qwen-VL models are commonly run with.
LOCAL_ENHANCEMENT_MAX_PIXELS: int = int(os.getenv("LOCAL_ENHANCEMENT_MAX_PIXELS", str(1280 * 28 * 28)))
# Above 1.0 discourages the repeated lines small models fall into.
LOCAL_ENHANCEMENT_REPETITION_PENALTY: float = float(os.getenv("LOCAL_ENHANCEMENT_REPETITION_PENALTY", "1.1"))
# Off: the summary is the whole answer, and reasoning would spend max_tokens.
LOCAL_ENHANCEMENT_THINKING: bool = os.getenv("LOCAL_ENHANCEMENT_THINKING", "false").lower() == "true"

# API server
# Browser origins allowed to call the API (CORS).  Comma-separated so you can
# add the production frontend URL later, e.g.
#   CORS_ALLOWED_ORIGINS=https://app.example.com,http://localhost:5173
# Non-browser clients (curl, servers, mobile) are never blocked by CORS.
_CORS_DEFAULT = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
CORS_ALLOWED_ORIGINS: list[str] = [
    o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", _CORS_DEFAULT).split(",") if o.strip()
]

# Vector store settings
# Every collection is created with cosine distance. The store holds one
# collection per embedding model; src/embed.py names it and adds the model to
# this metadata (collection_name, collection_metadata).
CHROMA_COLLECTION_METADATA: dict = {"hnsw:space": "cosine"}
