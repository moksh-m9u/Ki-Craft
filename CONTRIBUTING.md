# Contributing

This guide sets up the project for development: installing it, running each part, ingesting new documents, and the conventions for changes. What the project is and why it exists is in the [README](README.md). The working rules, architecture rules, recorded decisions and the current plan are in [AGENTS.md](AGENTS.md); read it before a non-trivial change.

## Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment variables](#environment-variables)
- [Dependencies](#dependencies)
- [Running the API](#running-the-api)
- [Running the dashboard](#running-the-dashboard)
- [Tests](#tests)
- [Database](#database)
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

You also need a Gemini API key, a watsonx.ai API key with its project id, and a Postgres database with pgvector; see [Environment variables](#environment-variables) and [Database](#database).

## Setup

The six documents' summaries ship in `json/` and their figures in `dbv2/images/`, so no PDF parsing tools are needed: you create the tables and store the summaries in your own database (step [Database](#database)), and the API and the dashboard run.

Clone the repository:

```bash
git clone https://github.com/moksh-m9u/Multimodal-Rag.git
```

```bash
cd Multimodal-Rag
```

Create the virtual environment and install every dependency (API, dashboard, ingestion, tests). `pyproject.toml` lists them and `uv.lock` pins the exact versions; uv also downloads Python 3.13 if the machine lacks it:

```bash
uv sync
```

```bash
source .venv/bin/activate
```

Create `.env` from the template and fill in the keys and the two database URLs (see [Environment variables](#environment-variables)):

```bash
cp .env.example .env
```

Create the tables and load the six documents into your database, as described in [Database](#database):

```bash
alembic upgrade head
```

```bash
python -m scripts.ingest json/*.json
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

## Environment variables

Copy `.env.example` to `.env` in the repository root and fill in the values. Write the values without quotes: `docker run --env-file` reads each line literally, so a quoted value reaches the container with the quotes included. `.env` is ignored by git; never commit it.

```bash
cp .env.example .env
```

| Variable | Needed for | Notes |
|---|---|---|
| `GEMINI_API_KEY` | Answer generation | Create one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `WATSONX_API_KEY` | Embedding every question, and every chunk at ingestion | An IBM Cloud API key |
| `WATSONX_PROJECT_ID` | Same | The id of your watsonx.ai project |
| `WATSONX_URL` | Same | Region endpoint; defaults to `https://eu-de.ml.cloud.ibm.com` |
| `DATABASE_URL` | Postgres (Neon), pooled | From the branch's Connect dialog with connection pooling on; the host contains `-pooler` |
| `DATABASE_URL_DIRECT` | Migrations and bulk copies | The same string with `-pooler` removed from the host |
| `PRODUCTION_DATABASE_URL_DIRECT` | Maintainers only: copying documents to production | The `production` branch's direct string; only `scripts/copy_document.py` reads it |
| `HF_TOKEN` | Ingestion only (chunk summaries) | Create one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
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
| Docker image (Render) | the API only | `api/requirements.txt` |
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

The first request to `/health` should return `{"status":"ok","database":"connected"}`:

```bash
curl http://127.0.0.1:8001/health
```

Interactive documentation is at http://127.0.0.1:8001/docs.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Lists the endpoints |
| GET | `/health` | Reports whether the database answers |
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

## Database

Everything the API reads lives in Postgres with the pgvector extension: documents, chunks with their summaries, original text, tables and embeddings, and figure paths. The figure image files themselves stay in `dbv2/images/`. Any Postgres 15 or later with pgvector works:

- **Neon (what the project uses).** Create a free project at neon.com in the region nearest your API, and copy two connection strings from the Connect dialog: pooled (connection pooling on, the host contains `-pooler`) into `DATABASE_URL`, and direct (pooling off) into `DATABASE_URL_DIRECT`. The maintainers develop on a `dev` branch; the deployed API reads `production`.
- **Docker.** `docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17`, then set both variables to `postgresql://postgres:postgres@localhost:5432/postgres`.
- **Any other host** with pgvector available (Supabase through its session pooler, a managed Postgres, a local install).

Then create the tables and store the six shipped documents. Storing re-embeds each summary with watsonx (310 calls' worth of text, about 100,000 tokens) and writes no summaries, since those are in `json/`:

```bash
alembic upgrade head
```

```bash
python -m scripts.ingest json/*.json
```

Chunks stored this way get new record ids, so figure URLs differ from another database's; they are stable within one database.

### From dev to production

The maintainers keep `.env` pointed at the `dev` branch, so ingestion, tests and experiments never touch the live data. A new document is ingested into `dev`, checked there, and then copied to `production` unchanged: the same record ids, summaries and vectors, with no embedding or summary call.

```bash
python -m scripts.copy_document new_datasheet
```

The script reads `DATABASE_URL_DIRECT` (dev) and writes `PRODUCTION_DATABASE_URL_DIRECT`. Each document is copied in one transaction. A document production already holds is skipped unless `--replace` is given. The figure files are not in the database: commit the new files in `dbv2/images/` and the export in `json/`, then redeploy.

### Migrations

The Postgres tables are described in `src/db/schema.py` and created by the Alembic migrations in `migrations/versions/`. Alembic connects with `DATABASE_URL_DIRECT`, the direct host, because a migration is one long session and the pooler is per transaction.

```bash
alembic upgrade head
```

```bash
alembic current
```

```bash
alembic check
```

`alembic check` reports any difference between `schema.py` and the database. A schema change is a new migration: edit `schema.py`, generate a draft with `alembic revision --autogenerate -m "what changed"`, read and correct it, and run `upgrade`. `alembic downgrade -1` undoes the latest one.

## Ingesting a new PDF

Ingestion of PDFs is not needed to use the six documents that ship with the repository (see [Database](#database)). Read this section before running it:

- It calls a vision model once for every chunk with tables or figures. `--backend huggingface` (the default) uses GLM-4.5V on the Hugging Face router, which needs `HF_TOKEN` and credit. `--backend local` uses a model on your Mac through `mlx_vlm.server`; see below.
- It calls watsonx once per chunk for the embedding.
- It writes to the database at `DATABASE_URL_DIRECT`, and figure files to `dbv2/images`.
- One script, `scripts/ingest.py`, takes either input. Both are stored by the same function, `src/ingestion/pipeline.py:store_export`: a PDF is first turned into an export, then stored like any other. Each figure becomes a file in `dbv2/images`, and the `images` table holds its path; the `documents` table holds the `source` name.

| Input | What it does |
|---|---|
| a PDF, or a folder of PDFs | Parses, chunks, summarises, writes the export to `json/<pdf name>.json`, then stores it as below. It skips a PDF whose export already exists, and refuses one the database already holds. If storing fails, the export stays, so `python -m scripts.ingest json/<pdf name>.json` finishes the job without summarising again |
| a `json/<name>.json` export | Embeds the existing summaries and stores the chunks, with no summarising. It refuses a document the database already holds unless `--replace` is given; replacing embeds first, then swaps the old chunks for the new ones in one transaction, and removes old figure files last. Replaced chunks get new record ids |

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

To check the summaries before they reach the shared database, send the chunks to a scratch database (for example a throwaway Neon branch, by pointing `DATABASE_URL_DIRECT` at it for this one command) and the figures to a scratch folder. The export still goes to `json/`:

```bash
DATABASE_URL_DIRECT=<scratch database URL> python -m scripts.ingest data/ne555.pdf --backend local --images /tmp/trial/images
```

Once the summaries look right, store that export as it is, with no summary model call (it is refused if the database already holds that document):

```bash
python -m scripts.ingest json/ne555.json
```

Each PDF is done in full or not at all. A PDF whose export (`json/<name>.json`) already exists is skipped, and a PDF that fails is logged while the batch carries on, so an interrupted overnight run resumes with the same command. `--images` and `--exports` write figures and exports somewhere other than the tracked folders, for a trial run.

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

The image contains only the backend code and the figure files (`api/`, `src/`, `config/`, `dbv2/images/`), and installs only `api/requirements.txt`. The chunks live in Postgres, so the container needs `DATABASE_URL` at run time along with the other credentials; the image holds no secrets. Render runs this image and sets `PORT` itself.

Build:

```bash
docker build -t multimodal-rag-api .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env multimodal-rag-api
```

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
| `EMBEDDING_MODEL` | `ibm/granite-embedding-278m-multilingual` | Embedding model on watsonx.ai; changing it means re-indexing |
| `EMBEDDING_MAX_INPUT_TOKENS` | `512` | The model's input limit; longer text is truncated |
| `GENERATION_MODEL` | `gemini-3.5-flash-lite` | Default answer model |
| `GENERATION_MAX_TOKENS` | `512` | Output limit per answer (environment variable) |
| `GENERATION_MAX_IMAGES` | `3` | Most figures sent to the model, in chunk order; 0 sends all (environment variable) |
| `RETRIEVAL_SEARCH_TYPE` | `mmr` | `similarity`, `mmr`, or `similarity_score_threshold` (environment variable; anything else stops startup) |
| `RETRIEVAL_K` | `5` | Chunks returned; all of them go to the model (environment variable) |
| `RETRIEVAL_FETCH_K` | `20` | MMR only: candidates fetched before MMR picks `RETRIEVAL_K` (environment variable) |
| `RETRIEVAL_LAMBDA_MULT` | `0.5` | MMR only: 1 is pure relevance, 0 is pure diversity (environment variable) |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.5` | `similarity_score_threshold` only: minimum relevance score (environment variable) |
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
  embed.py               the one embedding factory (watsonx Granite)
  llm.py                 the one chat model factory: catalogue entry -> LangChain client
  db/schema.py           Postgres tables: documents, chunks (vector(768)), images
  db/connection.py       Neon URL -> SQLAlchemy URL for psycopg 3
  documents.py           turns stored chunk metadata back into text, tables, figures
  browse.py              lists the stored documents and a document's chunks in order
  tables.py              stored HTML tables -> Markdown for prompts
  logger.py              logging setup
  ingestion/             offline: PDF -> elements -> chunks -> summaries -> Postgres
    extract.py             unstructured partition_pdf
    chunk.py               chunk_by_title
    enrich.py              vision LLM summary per chunk
    export.py              chunks -> JSON export
    pipeline.py            stores an export: embeds summaries, writes figure files and Postgres rows
  retrieval/
    search.py              pgvector search (similarity, MMR, threshold) and the retriever
    prompt.py              builds the multimodal prompt, chooses and labels the figures
    generate.py            retrieves, streams the answer, counts tokens, prices, traces
api/
  main.py                FastAPI app: CORS, lifespan, routers
  routers/               endpoints by purpose: system, retrieval, query, providers, documents, images
  service.py             owns the database engine and embedding client, picks the default LLM
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
  requirements.txt       the dashboard's packages (used by Streamlit Community Cloud)
migrations/              Alembic migrations that create the Postgres tables (alembic.ini at the root)
scripts/                 ingestion (ingest.py) and the dev-to-production copy (copy_document.py)
dbv2/images/             the figure files whose paths the database stores
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
- A new technical term goes in [docs/GLOSSARY.md](docs/GLOSSARY.md) in the same change.

## Troubleshooting

- **`uvicorn` starts the wrong Python.** Some machines have a stale Homebrew `uvicorn`. Activate `.venv` first, or call `.venv/bin/uvicorn`.
- **A plain `uv sync` removed `mlx-vlm`.** `uv sync` makes the environment match the lock exactly, so it drops extras you did not ask for and packages installed by hand. Use `uv sync --extra local`.
- **The Docker container cannot authenticate.** A value in `.env` has quotes; `docker run --env-file` keeps them. Remove them.
