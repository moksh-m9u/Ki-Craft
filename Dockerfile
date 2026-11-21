# Multimodal RAG - FastAPI backend image.
#
# Railway auto-detects a Dockerfile named `Dockerfile` at the repo root and
# uses it instead of the built-in builders.  The `PORT` env var set by Railway
# (default 8000) is bound at startup.
#
# Build:   docker build -t multimodal-rag-api .
# Run:     docker run --rm -p 8000:8000 --env-file .env multimodal-rag-api
#
# The default image embeds through an API only: watsonx.ai (EMBEDDING_BACKEND=
# watsonx, the default) or OpenRouter (EMBEDDING_BACKEND=openrouter, with
# OPENROUTER_API_KEY). Both clients are in api/requirements.txt, so no model
# weights and no ML framework are installed. EMBEDDING_BACKEND=local fails at
# startup with a message saying how to get it.
#
# Local embeddings (no watsonx credentials, no torch):
#          docker build --build-arg EMBEDDING_BACKEND=local -t multimodal-rag-api:local .
# This compiles llama-cpp-python and bakes the pinned GGUF Granite model into
# the image. BuildKit skips the stage a build does not use, so the default
# watsonx build compiles and downloads nothing extra.

ARG EMBEDDING_BACKEND=watsonx
ARG GGUF_FILE=granite-embedding-278m-multilingual-Q8_0.gguf

# Extras for EMBEDDING_BACKEND=local: a llama-cpp-python wheel (PyPI ships only
# source) and the GGUF file, checked against its pinned sha256.
FROM python:3.13-slim AS extras-local
ARG LLAMA_CPP_PYTHON_VERSION=0.3.37
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
# GGML_NATIVE=OFF: do not tune for the build machine's CPU, or the image can
# crash on the server's. GGML_OPENMP=OFF: no libgomp needed at runtime.
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=OFF" \
    pip wheel --no-cache-dir "llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION}" -w /extras/wheels
# Declared after the compile step, so choosing another model file reuses the wheel.
ARG GGUF_REVISION=61fba13303a369d52a088ccb745be93ecc2034f3
ARG GGUF_FILE
ARG GGUF_SHA256=b8b3ebc5f98d179493adf28e6e9e16508766ed85b72cedfbd9bc1540c4905a23
RUN mkdir -p /extras/models \
    && python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
       "https://huggingface.co/bartowski/granite-embedding-278m-multilingual-GGUF/resolve/${GGUF_REVISION}/${GGUF_FILE}" \
       "/extras/models/${GGUF_FILE}" \
    && echo "${GGUF_SHA256}  /extras/models/${GGUF_FILE}" | sha256sum -c -

FROM python:3.13-slim AS extras-watsonx
RUN mkdir -p /extras/models

FROM extras-${EMBEDDING_BACKEND} AS extras

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Backend-only dependencies (ingestion/dashboard libs stay out of the image).
# In the default build embeddings are a watsonx.ai or OpenRouter API call, so no ML framework
# runs in this image; the local build adds only llama.cpp, never torch.
# Copy requirements first so layer caching kicks in on code-only changes.
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Nothing for the watsonx build; for local, llama-cpp-python and the GGUF file.
# A Q4_K_M build passes GGUF_FILE and GGUF_SHA256 as build arguments too.
ARG EMBEDDING_BACKEND
ARG GGUF_FILE
# The wheel is mounted only for this step, so it never becomes an image layer.
RUN --mount=type=bind,from=extras,source=/extras,target=/tmp/extras \
    if [ -d /tmp/extras/wheels ]; then pip install --no-cache-dir /tmp/extras/wheels/*.whl; fi
COPY --from=extras /extras/models /extras/models
ENV EMBEDDING_BACKEND=${EMBEDDING_BACKEND} \
    LOCAL_EMBEDDING_GGUF_PATH=/extras/models/${GGUF_FILE}

# .dockerignore is an allowlist, so this copies only the backend code (api/,
# src/, config/) and the pre-built vector store with its figures (dbv2/).
COPY . .

# Writable runtime artifacts (logs are also mirrored to stdout).
RUN mkdir -p /app/logs && chmod -R u+w /app

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT', '8000'), timeout=5)"

# Single worker: the vector store + embedding client are heavy in-memory objects.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
