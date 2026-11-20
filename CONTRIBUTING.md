# Contributing

This guide sets up the project for development: installing it, running each part, ingesting new documents, and the conventions for changes. What the project is and why it exists is in the [README](README.md). The working rules, architecture rules, recorded decisions and the current plan are in [AGENTS.md](AGENTS.md); read it before a non-trivial change.

## Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Choosing an embedding backend](#choosing-an-embedding-backend)
- [Environment variables](#environment-variables)
- [Dependencies](#dependencies)
- [Running the API](#running-the-api)
- [Running the dashboard](#running-the-dashboard)
- [Tests](#tests)
- [Ingesting a new PDF](#ingesting-a-new-pdf)
- [Running with Docker](#running-with-docker)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Making a change](#making-a-change)
- [Troubleshooting](#troubleshooting)

## Prerequisites

| Tool | Needed for | Install |
|---|---|---|
| git | cloning | usually present; `xcode-select --install` on a Mac |
| [uv](https://docs.astral.sh/uv/) | the virtual environment and every Python package | `brew install uv` |
| Python 3.13 | running the code | `uv sync` downloads it when the machine lacks it |
| poppler, tesseract, libmagic, pandoc | ingestion only | see [Ingesting a new PDF](#ingesting-a-new-pdf) |
| Docker | building the API image only | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |

You also need a Gemini API key for answers, and one way to embed questions: a free OpenRouter key (no credit card), a watsonx.ai key with its project id, or the local Granite model; see [Choosing an embedding backend](#choosing-an-embedding-backend).

## Setup

The vector store is already built, so the API and the dashboard run straight after these steps, with no PDF parsing tools.

Clone the repository:

```bash
git clone https://github.com/moksh-m9u/Ki-Craft.git
```

```bash
cd Ki-Craft
```

Create the virtual environment and install every dependency (API, dashboard, ingestion, tests). `pyproject.toml` lists them and `uv.lock` pins the exact versions; uv also downloads Python 3.13 if the machine lacks it:

```bash
uv sync
```

```bash
source .venv/bin/activate
```

Create `.env` from the template and fill in `GEMINI_API_KEY` and `OPENROUTER_API_KEY`. The template selects the OpenRouter embedding backend; see [Choosing an embedding backend](#choosing-an-embedding-backend) for the others and [Environment variables](#environment-variables) for the rest:

```bash
cp .env.example .env
```

Check that the code imports and the tests pass:

```bash
python -m pytest -q -p no:cacheprovider
```

Start the API in one terminal:

```bash
uvicorn api.main:app --reload --port 8001
```

Start the dashboard in a second terminal (activate the virtual environment there too):

```bash
API_BASE_URL=http://127.0.0.1:8001 streamlit run app/dashboard.py
```

Open http://localhost:8501 and ask a question on the Query & Retrieve page.

## Choosing an embedding backend

A question is turned into a vector (an [embedding](docs/GLOSSARY.md)) and compared with the vectors of the stored summaries, so both must come from the same model, or the search finds nonsense. `EMBEDDING_BACKEND` chooses the model, and the store keeps one [collection](docs/GLOSSARY.md) per model, so changing the backend never mixes vectors:

| `EMBEDDING_BACKEND` | Model | Collection in `dbv2/chroma_db` | Needs | In the API Docker image |
|---|---|---|---|---|
| `openrouter` | `OPENROUTER_EMBEDDING_MODEL`, default `nvidia/nemotron-3-embed-1b:free` (2,048 dimensions) | `nvidia-nemotron-3-embed-1b` (shipped) | a free OpenRouter key; no credit card | yes |
| `watsonx` (used when unset) | `ibm/granite-embedding-278m-multilingual` (768 dimensions) | `langchain` (shipped; the original) | an IBM Cloud API key and a watsonx.ai project | yes |
| `local` | the same Granite model, as a GGUF file run by llama.cpp | `langchain` | `uv sync --extra local-embeddings` and a 303 MB model file; about 630 MB of RAM | no |

Both API backends send text to a third party: every question, and at ingestion or re-embedding every summary. Use `local` when the documents must stay on your machine. The models are compared, with measurements, in [docs/TRADEOFFS.md](docs/TRADEOFFS.md), section 4.

List the store's collections and the one your settings use (no network call):

```bash
python -m scripts.reembed --status
```

The API never searches a collection with the wrong model. When the configured model has no collection, or its collection was made by another model, `/health` and every query return 503 with a message that names the fix.

### OpenRouter

Create a key at https://openrouter.ai/settings/keys and put it in `.env`, unquoted:

```
EMBEDDING_BACKEND=openrouter
OPENROUTER_API_KEY=sk-or-...
```

The default model is a free one, so no credit is needed. Free models allow 20 requests a minute and 50 a day per account (1,000 a day once the account has bought $10 of credit). Every question is one request, and re-embedding the whole corpus is 5, because 64 summaries go in each request (`OPENROUTER_EMBEDDING_BATCH_SIZE`). See how many are left today (this call is not counted):

```bash
python -c "import json, urllib.request; from dotenv import dotenv_values; key = dotenv_values('.env')['OPENROUTER_API_KEY']; print(json.load(urllib.request.urlopen(urllib.request.Request('https://openrouter.ai/api/v1/key', headers={'Authorization': 'Bearer ' + key})))['data']['free_model_daily_requests'])"
```

Free endpoints can also be withdrawn, can answer 503 when overloaded (retried `OPENROUTER_MAX_RETRIES` times), and their provider may log what it receives. OpenRouter refuses them with 404 "No endpoints found matching your data policy" until free endpoints are allowed at https://openrouter.ai/settings/privacy. The same model without `:free` is the paid tier: it gives the same vectors, so it uses the same collection.

### Using another OpenRouter model

Any model listed at https://openrouter.ai/api/v1/embeddings/models works. Its collection is built from the summaries already in the store: no PDF is parsed and nothing is summarised again. See what it would send first:

```bash
EMBEDDING_BACKEND=openrouter OPENROUTER_EMBEDDING_MODEL=baai/bge-m3 python -m scripts.reembed --dry-run
```

Then build it:

```bash
EMBEDDING_BACKEND=openrouter OPENROUTER_EMBEDDING_MODEL=baai/bge-m3 python -m scripts.reembed
```

This copies every record of the `langchain` collection (summary, metadata and record id) into the model's collection with new vectors. Record ids are kept, so a figure URL (`/v1/images/{record_id}/{position}`) works whichever collection the API serves, and the figure files are not touched. Batches are written to `<collection>-partial`, so an interrupted run (a rate limit, a network error) resumes with the same command without paying for the finished batches again; the copy is checked record by record (ids, summaries, metadata, one vector size) and only then renamed to `<collection>`, where the API finds it. Then set `OPENROUTER_EMBEDDING_MODEL` in `.env`.

To go back, change `EMBEDDING_BACKEND` or `OPENROUTER_EMBEDDING_MODEL` back; no other collection was changed. To delete a collection you built:

```bash
python -c "import chromadb; chromadb.PersistentClient('dbv2/chroma_db').delete_collection('baai-bge-m3')"
```

### watsonx

Set `EMBEDDING_BACKEND=watsonx` (or leave it unset) with `WATSONX_API_KEY`, `WATSONX_PROJECT_ID` and `WATSONX_URL`. IBM asks for a credit card when the account is created, even for the free plan.

### Local Granite (GGUF)

To use the store with no watsonx account, run the embedder on your machine. The stored vectors were made by `ibm/granite-embedding-278m-multilingual`; a [GGUF](docs/GLOSSARY.md) build of the same model, run by llama.cpp with no torch, lands in the same vector space. The Q8_0 file scores cosine 0.9999 against 50 stored vectors and returns the same top-5 chunks for 12 test questions, and re-storing `json/ne555.json` with it gave vectors at cosine 0.9999 to the watsonx ones, so it works for retrieval and ingestion.

Install llama-cpp-python (it compiles llama.cpp, so a C++ compiler and cmake are needed):

```bash
uv sync --extra local-embeddings
```

Download the model (303 MB) into `models/`, which git ignores. The revision is pinned so the file cannot change under you:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/granite-embedding-278m-multilingual-GGUF', 'granite-embedding-278m-multilingual-Q8_0.gguf', revision='61fba13303a369d52a088ccb745be93ecc2034f3', local_dir='models')"
```

In `.env`, set `EMBEDDING_BACKEND=local`. Then retrieve without any answer model, so no Gemini key is needed:

```bash
curl -X POST http://127.0.0.1:8001/v1/retrieve -H 'Content-Type: application/json' -d '{"query": "LM317 reference voltage", "top_k": 5}'
```

The smaller Q4_K_M file (219 MB) also works but drifts a little (minimum cosine 0.991, 58 of 60 top-5 chunks the same); point `LOCAL_EMBEDDING_GGUF_PATH` at it to use it. Another embedder works only if it produces vectors in the same space. Two separately trained models almost never do, even with 768 dimensions. To check a candidate, embed a few stored summaries with it and compare each with its stored vector; a cosine near 1.0 means the same space, anything much lower means the store must be re-indexed.

## Environment variables

Copy `.env.example` to `.env` in the repository root and fill in the values. Write the values without quotes: `docker run --env-file` reads each line literally, so a quoted value reaches the container with the quotes included. `.env` is ignored by git; never commit it.

```bash
cp .env.example .env
```

| Variable | Needed for | Notes |
|---|---|---|
| `GEMINI_API_KEY` | Answer generation | Create one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `OPENROUTER_API_KEY` | `EMBEDDING_BACKEND=openrouter`: embedding every question, and every summary at ingestion | Create one at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys); free models need no credit |
| `OPENROUTER_EMBEDDING_MODEL` | Same | Defaults to `nvidia/nemotron-3-embed-1b:free`; another model needs its own collection (`python -m scripts.reembed`) |
| `WATSONX_API_KEY` | `EMBEDDING_BACKEND=watsonx`: embedding every question, and every summary at ingestion | An IBM Cloud API key |
| `WATSONX_PROJECT_ID` | Same | The id of your watsonx.ai project |
| `WATSONX_URL` | Same | Region endpoint; defaults to `https://eu-de.ml.cloud.ibm.com` |
| `EMBEDDING_BACKEND` | Choosing the embedding model | `openrouter`, `watsonx` (used when unset) or `local`; see [Choosing an embedding backend](#choosing-an-embedding-backend) |
| `LOCAL_EMBEDDING_GGUF_PATH` | Local embeddings | The GGUF file; defaults to `models/granite-embedding-278m-multilingual-Q8_0.gguf` |
| `LOCAL_EMBEDDING_THREADS` | Local embeddings | CPU threads for llama.cpp; defaults to 1, which suits a small server |
| `HF_TOKEN` | Ingestion (chunk summaries); optionally the local embedding model download | Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). The local embedding model is public, so the token only raises download limits |
| `API_BASE_URL` | Dashboard | Where the dashboard finds the API; defaults to `http://localhost:8000` |
| `CORS_ALLOWED_ORIGINS` | API | Comma-separated origins allowed to call the API from a browser |
| `GENERATION_MAX_TOKENS` | API | Output token limit per answer; defaults to 512 |
| `LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT` | Optional tracing | Leave unset to run without tracing. `LANGSMITH_TRACING=false` turns tracing off even when `.env` has the older `LANGCHAIN_TRACING_V2=true` |
| `UNSTRUCTURED_API_KEY`, `UNSTRUCTURED_URL` | Not used by the code yet | For Unstructured's hosted Parse API, the planned production parser |

## Dependencies

Each place the code runs installs its own list, so the API image and the hosted dashboard stay small:

| Where it runs | What runs there | Installed from |
|---|---|---|
| Your machine | everything: API, dashboard, ingestion, tests | `pyproject.toml`, pinned by `uv.lock` (`uv sync`) |
| Docker image (Railway) | the API only | `api/requirements.txt` |
| Streamlit Community Cloud | the dashboard only | `app/requirements.txt` |

Every package goes in `pyproject.toml`. One the API imports also goes in `api/requirements.txt`, and one the dashboard imports also goes in `app/requirements.txt`; ingestion and test packages stay in `pyproject.toml` only (tests in its `dev` group). Then update the lock file and the environment, and commit `pyproject.toml` with `uv.lock`:

```bash
uv lock
```

```bash
uv sync --extra local
```

## Running the API

```bash
uvicorn api.main:app --reload --port 8001
```

The first request to `/health` should return `{"status":"ok","vector_store":"loaded"}`:

```bash
curl http://127.0.0.1:8001/health
```

Interactive documentation is at http://127.0.0.1:8001/docs.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Lists the endpoints |
| GET | `/health` | Reports whether the vector store loaded |
| POST | `/v1/retrieve` | Returns the retrieved chunks, with no LLM call |
| POST | `/v1/query` | Retrieves, then answers with the default model, or with the provider, model and key in an optional `llm` object |
| POST | `/v1/query/stream` | The same, streamed token by token over Server-Sent Events |
| GET | `/v1/images/{record_id}/{position}` | Serves one figure of a stored chunk, at a stable URL |
| GET | `/v1/providers` | Lists the providers and models the `llm` object accepts |
| GET | `/v1/documents` | Lists the indexed documents with their chunk, figure and table counts |
| GET | `/v1/documents/{document_id}/chunks` | One page of a document's chunks, in reading order (`offset`, `limit`) |

Retrieve chunks only:

```bash
curl -X POST http://127.0.0.1:8001/v1/retrieve -H 'Content-Type: application/json' -d '{"query": "LM555 astable operation frequency formula", "top_k": 10}'
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8001/v1/query -H 'Content-Type: application/json' -d '{"query": "What is the dropout voltage of the TLV1117?"}'
```

Ask with another model and your own key (the API has no server-side key for other providers, so `api_key` is required whenever `llm` is sent):

```bash
curl -X POST http://127.0.0.1:8001/v1/query -H 'Content-Type: application/json' -d '{"query": "What is the dropout voltage of the TLV1117?", "top_k": 5, "max_images": 3, "llm": {"provider": "groq", "model": "openai/gpt-oss-120b", "api_key": "YOUR_GROQ_API_KEY", "max_tokens": 2048}}'
```

Stream an answer:

```bash
curl -N http://127.0.0.1:8001/v1/query/stream -H 'Content-Type: application/json' -d '{"query": "What is the output voltage range of the LM317?"}'
```

The full request and response shapes are in [docs/API.md](docs/API.md).

Opening the Chroma store rewrites tracked files under `dbv2/chroma_db`. After running the API from a clone you intend to commit from, restore them:

```bash
git checkout -- dbv2/chroma_db
```

## Running the dashboard

```bash
API_BASE_URL=http://127.0.0.1:8001 streamlit run app/dashboard.py
```

| Page | Purpose | Data source |
|---|---|---|
| Home | Overview of the pages | none |
| Query & Retrieve | Ask a question, watch the answer stream, see the chunks and figures behind it | the API |
| Chunk Explorer | Browse every chunk: summary, raw text, tables, figures, health score | the API (`/v1/documents`) |
| Compare Chunks | Two chunks side by side | the API |
| Dataset Analytics | Histograms and top-20 lists per document | the API |

Every page reads through the API, so the dashboard needs no local data files.

### On Streamlit Community Cloud

A second, smaller app shows evaluation results:

```bash
streamlit run app/rag_eval_dashboard.py
```

Set the main file path to `app/dashboard.py`, the Python version (Advanced settings) to 3.13, and the secret `API_BASE_URL = "https://<your API host>"`. Streamlit Cloud reads the dependency file next to the main file first, so it installs `app/requirements.txt`; with another main file path it would fall back to `uv.lock` and try to install the whole ingestion stack.

## Tests

The suite runs without network access; external clients are mocked.

```bash
python -m pytest -q -p no:cacheprovider
```

Run one file or one test:

```bash
python -m pytest -q -p no:cacheprovider tests/test_embed.py
```

## Ingesting a new PDF

Ingestion is not needed to use the five documents that ship with the repository. Read this section before running it:

- It calls a vision model once for every chunk with tables or figures. `--backend huggingface` (the default) uses GLM-4.5V on the Hugging Face router, which needs `HF_TOKEN` and credit. `--backend local` uses a model on your Mac through `mlx_vlm.server`; see below.
- It embeds every summary with the configured embedding backend and writes the records into that model's collection only. To add the document to another model's collection too, run `python -m scripts.ingest json/<name>.json` again with that backend. Figure files are shared by every collection, so replacing a document (`--replace`) rewrites its figures for all of them.
- It writes to the tracked store in `dbv2/chroma_db`, so it changes files that git tracks.
- One script, `scripts/ingest.py`, takes either input. Both are stored by the same function, `src/ingestion/pipeline.py:store_export`: a PDF is first turned into an export, then stored like any other. Each figure becomes a file in `dbv2/images`, and the record holds its path and a `source` name.

| Input | What it does |
|---|---|
| a PDF, or a folder of PDFs | Parses, chunks, summarises, writes the export to `json/<pdf name>.json`, then stores it as below. It skips a PDF whose export already exists, and refuses one the store already holds. If storing fails, the export stays, so `python -m scripts.ingest json/<pdf name>.json` finishes the job without summarising again |
| a `json/<name>.json` export | Embeds the existing summaries and stores the records, with no summarising. It refuses a document the store already holds unless `--replace` is given; replacing embeds first and only then removes the old records and figures |

PDF parsing needs programs that no Python dependency file installs: poppler reads PDFs, tesseract does OCR, libmagic detects file types and pandoc converts formats.

```bash
brew install poppler tesseract libmagic pandoc
```

On Ubuntu or Debian:

```bash
sudo apt-get install -y poppler-utils tesseract-ocr libmagic1 pandoc
```

Run the pipeline on one PDF:

```bash
python -m scripts.ingest path/to/document.pdf
```

### Summarising with a local model

`mlx-vlm` is an optional extra that installs only on Apple silicon:

```bash
uv sync --extra local
```

Start the model server in its own terminal and leave it running. The first start downloads the model (about 3 GB) into the Hugging Face cache:

```bash
python -m mlx_vlm.server --model mlx-community/Qwen3.5-4B-MLX-4bit --host 127.0.0.1 --port 8080
```

Then ingest one PDF, or every PDF in a folder:

```bash
python -m scripts.ingest data/ --backend local
```

To check the summaries before they reach the store, write only the export to `json/` and send the records and figures to a scratch folder. The dashboard lists every `json/*.json`, so the new document can be read in the chunk inspector:

```bash
python -m scripts.ingest data/ne555.pdf --backend local --store /tmp/trial/chroma_db --images /tmp/trial/images
```

Once the summaries look right, store that export as it is, with no summary model call (it is refused if the store already holds that document):

```bash
python -m scripts.ingest json/ne555.json
```

Each PDF is done in full or not at all. A PDF whose export (`json/<name>.json`) already exists is skipped, and a PDF that fails is logged while the batch carries on, so an interrupted overnight run resumes with the same command. `--store`, `--images` and `--exports` write somewhere other than the tracked store, for a trial run.

The summary settings are environment variables (defaults in `config/settings.py`):

| Variable | Default | What it does |
|---|---|---|
| `ENHANCEMENT_BACKEND` | `huggingface` | `huggingface` or `local`; `--backend` overrides it |
| `LOCAL_ENHANCEMENT_BASE_URL` | `http://127.0.0.1:8080/v1` | Where `mlx_vlm.server` listens |
| `LOCAL_ENHANCEMENT_MODEL` | `mlx-community/Qwen3.5-4B-MLX-4bit` | Model id sent to the server |
| `ENHANCEMENT_MAX_TOKENS` | `512` | Longest summary; the embedder reads only the first 512 tokens |
| `ENHANCEMENT_TEMPERATURE` | `0.0` | Sampling temperature |
| `ENHANCEMENT_TIMEOUT_SECONDS` | `300` | One call's time limit before a retry |
| `ENHANCEMENT_MAX_RETRIES` | `2` | Retries, with exponential backoff, before the PDF fails |
| `ENHANCEMENT_MAX_IMAGES` | `0` | Most figures per call (0 = all) |
| `LOCAL_ENHANCEMENT_MAX_PIXELS` | `1003520` | Figures larger than this area are downscaled by the server |
| `LOCAL_ENHANCEMENT_REPETITION_PENALTY` | `1.1` | Discourages repeated lines |
| `LOCAL_ENHANCEMENT_THINKING` | `false` | Lets the model reason first (spends `ENHANCEMENT_MAX_TOKENS`) |

Uploading a PDF through the API, with a hosted parser in production, is the goal of Checkpoint 7 in [AGENTS.md](AGENTS.md).

## Running with Docker

The image contains only the backend code and the pre-built store (`api/`, `src/`, `config/`, `dbv2/`), and installs only `api/requirements.txt`. It is about 1.17 GB and holds no secrets; credentials are passed at run time. Railway builds this same `Dockerfile` and sets `PORT` itself.

Build:

```bash
docker build -t multimodal-rag-api .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env multimodal-rag-api
```

The image embeds through an API only: it holds both shipped collections, the OpenRouter and watsonx clients, and no model weights or ML framework. `-e` overrides a value from `.env`, so one image serves either backend:

```bash
docker run --rm -p 8000:8000 --env-file .env -e EMBEDDING_BACKEND=openrouter multimodal-rag-api
```

`EMBEDDING_BACKEND=local` is not in this image: `/health` returns 503 saying so. The optional local build (`--build-arg EMBEDDING_BACKEND=local`, described at the top of the `Dockerfile`) adds llama.cpp and the GGUF file and grows the image to about 1.85 GB.

A collection you built on your machine, for example with another OpenRouter model, is used by mounting the store over the one in the image:

```bash
docker run --rm -p 8000:8000 --env-file .env -e EMBEDDING_BACKEND=openrouter -e OPENROUTER_EMBEDDING_MODEL=baai/bge-m3 -v "$PWD/dbv2/chroma_db:/app/dbv2/chroma_db" multimodal-rag-api
```

Measured on 2026-11-20 with OpenRouter: 191 MiB idle and 228 MB at peak. Under 512 MB and 0.1 CPU (Render's free tier) it starts in about 2.5 minutes and answers without being killed; the local GGUF image runs out of memory there.

Check:

```bash
curl http://127.0.0.1:8000/health
```

On an Apple Silicon Mac the build above produces a `linux/arm64` image. To build for an amd64 server and push it to Docker Hub, log in and replace `<user>` with your Docker Hub username:

```bash
docker login
```

```bash
docker buildx build --platform linux/amd64 -t <user>/multimodal-rag-api:latest --push .
```

## Configuration

Every tunable lives in `config/settings.py`.

| Setting | Value | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `ibm/granite-embedding-278m-multilingual` | The Granite model of the watsonx and local backends; its collection is `langchain` |
| `EMBEDDING_BACKEND` | `watsonx` | `openrouter` calls `OPENROUTER_EMBEDDING_MODEL`; `local` runs a GGUF build of the Granite model with llama.cpp (needs `uv sync --extra local-embeddings` and the model file). Each model has its own collection (environment variable) |
| `OPENROUTER_EMBEDDING_MODEL` | `nvidia/nemotron-3-embed-1b:free` | OpenRouter embedding model (environment variable) |
| `OPENROUTER_EMBEDDING_BATCH_SIZE` | `64` | Texts per OpenRouter request (environment variable) |
| `OPENROUTER_TIMEOUT_SECONDS` | `60` | One request's time limit (environment variable) |
| `OPENROUTER_MAX_RETRIES` | `3` | Retries after a timeout, connection error, 408, 429 or 5xx, with exponential backoff; 400, 401, 402 and 404 are not retried (environment variable) |
| `EMBEDDING_MAX_INPUT_TOKENS` | `512` | The model's input limit; longer text is truncated |
| `GENERATION_MODEL` | `gemini-3.5-flash-lite` | Default answer model |
| `GENERATION_MAX_TOKENS` | `512` | Output limit per answer (environment variable) |
| `GENERATION_MAX_IMAGES` | `3` | Most figures sent to the model, in chunk order; 0 sends all (environment variable) |
| `RETRIEVAL_SEARCH_TYPE` | `mmr` | `similarity`, `mmr`, or `similarity_score_threshold` (environment variable; anything else stops startup) |
| `RETRIEVAL_K` | `5` | Chunks returned; all of them go to the model (environment variable) |
| `RETRIEVAL_FETCH_K` | `20` | MMR only: candidates fetched before MMR picks `RETRIEVAL_K` (environment variable) |
| `RETRIEVAL_LAMBDA_MULT` | `0.5` | MMR only: 1 is pure relevance, 0 is pure diversity (environment variable) |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.5` | `similarity_score_threshold` only: minimum relevance score (environment variable) |
| `CHROMA_PERSIST_DIR` | `dbv2/chroma_db` | Vector store location; one collection per embedding model (`src/embed.py:collection_name`) |
| `IMAGES_DIR` | `dbv2/images` | Figure files |
| `PDF_STRATEGY` | `hi_res` | `unstructured` parsing mode |
| `CHUNK_MAX_CHARACTERS` | `3000` | Largest chunk |
| `ENHANCEMENT_MODEL` | `zai-org/GLM-4.5V` | Vision model that writes the chunk summaries |

The providers and models the `llm` object accepts are listed in `config/providers.py`, and served by `GET /v1/providers`.

## Project structure

```
config/
  settings.py            every tunable and environment variable
  providers.py           LLM providers and models the llm object accepts
src/
  embed.py               the one embedding factory (watsonx, OpenRouter, local GGUF) and each model's collection name
  llm.py                 the one chat model factory: catalogue entry -> LangChain client
  documents.py           turns stored chunk metadata back into text, tables, figures
  browse.py              lists the stored documents and a document's chunks in order
  tables.py              stored HTML tables -> Markdown for prompts
  logger.py              logging setup
  ingestion/             offline: PDF -> elements -> chunks -> summaries -> Chroma
    extract.py             unstructured partition_pdf
    chunk.py               chunk_by_title
    enrich.py              vision LLM summary per chunk
    export.py              chunks -> JSON export
    pipeline.py            writes the Chroma store
    reembed.py             copies a collection into another embedding model's collection
  retrieval/
    search.py              loads Chroma, builds the MMR retriever, retrieves
    prompt.py              builds the multimodal prompt, chooses and labels the figures
    generate.py            retrieves, streams the answer, counts tokens, prices, traces
api/
  main.py                FastAPI app: CORS, lifespan, routers
  routers/               endpoints by purpose: system, retrieval, query, providers, documents, images
  service.py             owns the store and retriever, picks the default LLM
  schemas.py             request and response models
  openapi_examples.py    docs text and example payloads shown at /docs
  requirements.txt       the backend's packages (used by the Dockerfile)
app/
  dashboard.py           Streamlit entry point: page config, styles, st.navigation
  views/                 one module per page and URL: home, explorer, compare, analytics, chat
  components/            pieces several pages draw: document picker, filters, model settings, chunk and answer views
  chunk_data.py          cached API reads and helpers over chunk dicts
  state.py               session state defaults and the loaded document
  styles.css             the dashboard's CSS
  api_client.py          the only place the UI talks to the backend
  rag_eval_dashboard.py  Streamlit viewer for evaluation results
  requirements.txt       the dashboard's packages (used by Streamlit Community Cloud)
scripts/                 command-line ingestion (scripts/ingest.py) and re-embedding (scripts/reembed.py)
dbv2/                    the built Chroma store and the figure images
json/                    per-document chunk exports used by the dashboard
data/                    source PDFs
evals/                   evaluation questions and notes
tests/                   pytest suite (no network)
docs/                    API reference, trade-offs, product research, glossary
Dockerfile               backend image
pyproject.toml, uv.lock  every dependency, with exact versions (uv sync)
.env.example             every environment variable, without values
```

## Making a change

- Keep one concern per change, and prefer several small commits to one large one.
- Every module starts with a docstring saying what it is for and where it sits in the data flow; every public function has a docstring and type hints. Comments explain why, not what.
- Business logic lives in `src/`. `api/` routers stay thin, and `app/` talks to the backend only over HTTP through `app/api_client.py`.
- Run the tests before every commit. Tests never call the network: mock external clients, as `tests/test_embed.py` does.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary` in the imperative, with a body that explains why. Types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `build`, `ci`, `chore`.
- Branch names describe the change: `feat/<feature>`, `fix/<bug>`, `refactor/<area>`.
- Restore `dbv2/chroma_db` before committing (see [Troubleshooting](#troubleshooting)) unless the change is a deliberate re-index.
- A new technical term goes in [docs/GLOSSARY.md](docs/GLOSSARY.md) in the same change.

## Troubleshooting

- **`uvicorn` starts the wrong Python.** Some machines have a stale Homebrew `uvicorn`. Activate `.venv` first, or call `.venv/bin/uvicorn`.
- **A plain `uv sync` removed `mlx-vlm`.** `uv sync` makes the environment match the lock exactly, so it drops extras you did not ask for and packages installed by hand. Use `uv sync --extra local`.
- **A plain `uv sync` removed the local embedding packages.** Use `uv sync --extra local-embeddings`. Without them, `EMBEDDING_BACKEND=local` fails with `No module named 'llama_cpp'`.
- **The Docker container cannot authenticate.** A value in `.env` has quotes; `docker run --env-file` keeps them. Remove them.
- **`/health` says "has no collection for embedding model ...".** The configured model has no collection yet. Build it with `python -m scripts.reembed`, or switch back to a model listed by `python -m scripts.reembed --status`.
- **`/health` says "Embedding configuration mismatch".** The collection was made by another model than the one configured. Query it with its own model, or build a collection for yours with `python -m scripts.reembed`.
- **OpenRouter returns 401.** `OPENROUTER_API_KEY` is wrong or expired; keys can carry an expiry date.
- **OpenRouter returns 402.** The model is a paid one and the account has no credit. Use a `:free` model or buy credit.
- **OpenRouter returns 404 "No endpoints found matching your data policy".** Allow free endpoints at https://openrouter.ai/settings/privacy, or use a paid model.
- **OpenRouter returns 429.** The free daily limit (50 requests) is spent; it resets at 00:00 UTC. `--batch-size` lowers the requests a re-embedding needs.
- **OpenRouter returns 503 "Service is overloaded".** The free endpoint is busy; the client already retried. Try again later or use another model.
- **`git status` shows changes under `dbv2/chroma_db` you did not make.** Opening the store rewrites them. Run `git checkout -- dbv2/chroma_db`.
