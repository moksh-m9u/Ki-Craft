# AGENTS.md

Instructions for any coding agent (Claude Code, Codex, Cursor, and others) working in this repository. Read this file before making a change. The companion documents are:

- `docs/TRADEOFFS.md`: why each component is built the way it is, the alternatives, and the current recommendation.
- `docs/PRODUCT_RESEARCH.md`: what users who upload datasheets and PCB documents actually need, and where this project can stand out.
- `docs/GLOSSARY.md`: plain-language explanations of the technical terms used in the code and the docs.
- `CONTRIBUTING.md`: setup, running each part, ingestion, Docker, configuration, and which dependency file a new package goes in. The README is the product page: the vision (an open-source, AI-assisted PCB design and EDA toolkit next to KiCad), who it is for, status by phase, what works today, and links to `CONTRIBUTING.md`, `docs/TRADEOFFS.md` and `docs/PRODUCT_RESEARCH.md` only; the other docs are internal. Keep its Status table in step with the checkpoints below.

## Why this repository exists

This is a multimodal RAG system over electronics documents (datasheets, application notes, and later schematics). A user uploads a PDF and asks questions about it. The answer draws on the text, the tables, and the figures in the document.

The long-term direction is an agent that assists PCB designers through the whole design process. That is explicitly out of scope for now. Do not add agent frameworks, MCP integrations, or web search until the current phase is finished.

**Current phase: make the codebase explainable.** The owner must be able to explain every module, every function, how data travels from PDF to answer, and defend every engineering decision. Much of the code grew quickly and is hard to follow. Every change in this phase should make the code smaller, clearer, or more correct. Behaviour stays the same unless a change fixes a real architectural mistake, and that fix is agreed first.

## How we work together

This is AI-assisted coding, not autonomous coding. The owner stays in control of every decision and every commit.

### The change loop

Follow these steps for every change, however small:

1. **Propose.** Say what you want to change, why, and which files it touches. If there is any decision to make (a library, an API shape, deleting something, a schema), ask first and wait for an answer. Do not pick silently.
2. **Implement one small change.** One concern per change. Prefer several small changes over one large one.
3. **Check it yourself.** Run the test suite and any direct check that applies (an import, a curl against the local API, a script). Report failures honestly with the output.
4. **Hand over.** Give the owner:
   - the exact commands to test the change, each in its own code block;
   - what they should see if it works;
   - a **navigation walkthrough**: where to start reading, the order of files and functions to follow, and the shape of the data at each step. Explain from first principles, assuming no prior context about the change;
   - anything not verified.
5. **Answer questions.** The owner will read the code and ask about it. Answer directly, quoting `file:line` locations.
6. **Commit only when asked.** When the owner says to commit, make one commit for that change.
7. **Push only when asked.** Pushes go straight to `main`, and only after the owner has tested and approved.

If something fails twice in a row, stop and explain the problem rather than looping on fixes. If you need an API key, a Docker image, a model download, or any new software installed, stop and explain exactly what it is, why it is needed, and how large it is, then wait.

### Navigation walkthrough format

After each change, write the walkthrough like this:

```
Start: api/routers/query.py:query()          the HTTP entry point
  -> api/service.py:RAGService.answer()      loads the store once, retrieves
  -> src/retrieval/search.py:retrieve_chunks()  returns list[Document]
  -> src/retrieval/generate.py:generate_answer_stream() builds the prompt (prompt.py), streams the LLM
Data: QueryRequest -> list[Document] (metadata: raw_text, tables_html, image_paths) -> str answer -> QueryResponse
Try: set a breakpoint in X, or run the command below, and look at Y.
```

The owner wants depth here. At each hop, also explain from first principles what the library underneath does for us (FastAPI, LangChain, SQLAlchemy, pgvector, unstructured, and so on): what it receives, what it returns, and what would break without it. Assume no prior knowledge of that library.

### Glossary

`docs/GLOSSARY.md` holds a short plain-language explanation of every technical term the project uses (SSE, HNSW, MMR, RRF, lifespan, and so on). When a change or a document introduces a term that is not there yet, add it in the same change and link to it rather than explaining it again elsewhere.

### Commits and branches

- The owner is the sole author. Never add a `Co-Authored-By` trailer or any other agent attribution, in commits or PR descriptions.
- Use Conventional Commits: `type(scope): summary` in the imperative mood, with a body that explains why. Types: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `build`, `ci`, `chore`. Mark breaking API changes with `!` after the type and a `BREAKING CHANGE:` footer.
- Branch names describe the feature (`feat/<feature>`, `refactor/<area>`, `fix/<bug>`) and never contain the word "claude" or any agent name.
- Commit new documents and files promptly once the owner approves them. Untracked files cannot be recovered with git if they are deleted.

## Code style

- Every module starts with a docstring that says what the module is for and where it sits in the data flow.
- Every public function and class has a docstring: what it does, its arguments, what it returns, and what it raises. Private helpers get a one-line docstring.
- No banner comments (`# ----` or `# ====` dividers). If a file needs section dividers, it should probably be split into modules instead.
- No emojis anywhere: code, comments, logs, UI text, commit messages, and docs.
- Comments explain why, not what. Delete comments that restate the code.
- Use type hints on all function signatures.
- Keep things simple enough to explain. Prefer a plain function to a class, and a class to a framework. Do not add an abstraction until there are two real callers.
- Configuration lives in `config/settings.py` and is read from environment variables. No hardcoded secrets or URLs elsewhere.

## Architecture rules

- **Frontend and backend are strictly separate.** `app/` (Streamlit) talks to the backend only over HTTP through `app/api_client.py`. It must not import from `src/`, `api/`, or `config/` and must not read `json/` or `dbv2/` directly. Removing the remaining imports is part of the current phase.
- **One FastAPI app, many routers.** `api/main.py` only creates the app, sets up middleware and the lifespan, and includes routers. Each router file in `api/routers/` groups endpoints by purpose and stays thin: validate input, call a service function, shape the response.
- **Business logic lives in `src/`**, not in route handlers. `src/` must not import from `api/` or `app/`.
- **Heavy objects load once.** The embedding client, the vector store, and LLM clients are created at startup (FastAPI lifespan) or lazily once, never per request.
- **Blocking work stays off the event loop.** Synchronous LangChain calls inside async endpoints run through `asyncio.to_thread` or use the async LangChain methods (`ainvoke`, `astream`).
- **Errors are real HTTP errors.** Failures return 4xx/5xx with a JSON body, never a 200 with an apology string. A failing LLM provider returns 502.

## Repository map

```
config/settings.py      every tunable and environment variable
config/providers.py     catalogue of LLM providers and models the query llm object accepts
src/embed.py            the one embedding factory (IBM watsonx Granite, 768 dims)
src/llm.py              the one chat model factory: catalogue entry -> LangChain client; default_llm
src/ingestion/          offline pipeline: PDF -> elements -> chunks -> summaries -> Postgres
  extract.py              unstructured partition_pdf (hi_res, tables, images); drops headers, footers, logos
  chunk.py                chunk_by_title
  enrich.py               vision LLM summary per chunk (HF router GLM-4.5V, or local mlx_vlm Qwen3.5-4B)
  export.py               chunks -> JSON export
  pipeline.py             export -> rows with figure files; runs the steps and writes Postgres
src/retrieval/
  search.py               pgvector search (similarity, MMR, threshold), PostgresRetriever
  prompt.py               build the multimodal prompt, label chunks (C1) and figures (IMG1)
  citations.py            read the finished answer's [C1] and [IMG1] citations and check them
  generate.py             retrieve, stream the answer from the model, count tokens, price, trace
src/db/connection.py    Neon URL -> SQLAlchemy URL for psycopg 3
src/db/store.py         one document's rows: is_stored, write_document (one transaction), read_document, copy_document
src/db/schema.py        Postgres tables (documents, chunks with vector(768), images), SQLAlchemy Core
migrations/             Alembic migrations that create those tables (alembic.ini at the root)
src/documents.py        turns stored chunk metadata back into text, tables, images
src/browse.py           lists stored documents and a document's chunks in order (no search)
api/main.py             FastAPI app: CORS, lifespan, include_router calls
api/routers/            endpoints by purpose: system, retrieval, query, providers, documents, images
api/openapi_examples.py docs text and example payloads shown at /docs
api/service.py          RAGService: owns the store and retriever, picks the default LLM
api/schemas.py          Pydantic request and response models
api/requirements.txt    the packages the Docker image installs for the API
app/dashboard.py        Streamlit entry point: page config, styles, st.navigation (one URL per page)
app/views/              one module per page: home (/), explorer, compare, analytics, chat
app/components/         pieces several pages draw: document picker, filters, model settings, chunk and answer views
app/chunk_data.py       cached API reads and helpers over chunk dicts
app/state.py            session state defaults, kept widget values, the loaded document
app/api_client.py       the only place the UI talks to the backend
app/requirements.txt    the packages Streamlit Community Cloud installs for the dashboard
scripts/                CLI ingestion: PDFs or json/ exports into Postgres (scripts/ingest.py);
                        copy_document.py copies documents from dev to production unchanged
dbv2/images/            the figure files whose paths the database stores
json/                   per-document chunk exports (summaries included); Postgres can be rebuilt from them
evals/                  evaluation questions and notes
tests/                  pytest suite (no network)
```

Data flow at a glance:

```
Ingestion (offline):  PDF -> partition_pdf -> chunk_by_title -> vision summary per chunk
                      -> embed summary (watsonx) -> Postgres rows (summary, vector(768),
                      raw_text, tables_html, figure paths) + figure files on disk
Query (online):       question -> embed -> pgvector MMR (k=5 of 20) -> rows as Documents
                      -> multimodal prompt -> Gemini -> answer
```

## Commands

Use the project virtual environment at `/Users/moksh/Multimodal-Rag/.venv` (Python 3.13). A stale Homebrew `uvicorn` exists on the machine, so always call the venv binaries or activate the venv first. `pyproject.toml` is the one dependency list and `uv.lock` pins it; `uv sync` installs both, and `uv lock` refreshes the lock after `pyproject.toml` changes. `api/requirements.txt` (the Docker image) and `app/requirements.txt` (Streamlit Community Cloud) are deliberate subsets; the Dependencies section of `CONTRIBUTING.md` says which list a new package goes in.

```bash
source .venv/bin/activate
```

```bash
python -m pytest -q -p no:cacheprovider
```

```bash
uvicorn api.main:app --reload --port 8001
```

```bash
API_BASE_URL=http://127.0.0.1:8001 streamlit run app/dashboard.py
```

```bash
python -m scripts.ingest path/to/document.pdf
```

```bash
python -m scripts.ingest data/ --backend local
```

Notes:

- Tests must not call the network. Mock external clients (see `tests/test_embed.py`).
- `.env` points `DATABASE_URL` and `DATABASE_URL_DIRECT` at the Neon `dev` branch. Live checks and ingestion runs use `dev`; `production` is what the deployed API reads, and new documents reach it only through `python -m scripts.copy_document <source>` (no re-embedding, same record ids).
- Live checks load credentials from `.env` with `python-dotenv`. Never print, log, or read secret values. Refer to them only by variable name.
- Streamlit copies root-level keys from `.streamlit/secrets.toml` into `os.environ` and overwrites existing values. Keep that file out of git (it is ignored).

## Environment variables

`GEMINI_API_KEY`, `HF_TOKEN`, `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, `DATABASE_URL` (Neon, pooled), `DATABASE_URL_DIRECT` (Neon, direct; migrations and bulk copies), `PRODUCTION_DATABASE_URL_DIRECT` (read only by `scripts/copy_document.py`), `API_BASE_URL`, `CORS_ALLOWED_ORIGINS`, `GENERATION_MAX_TOKENS`, and the LangSmith variables (`LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, `LANGSMITH_PROJECT`). See `config/settings.py` for defaults. `LANGSMITH_TRACING` is the one tracing switch: when it is unset the legacy `LANGCHAIN_TRACING_V2` is used, and `config/settings.py` writes the result to `LANGSMITH_TRACING_V2`, which the langsmith library checks first.

Ingestion summaries: `ENHANCEMENT_BACKEND` (`huggingface` or `local`) and the `ENHANCEMENT_*` and `LOCAL_ENHANCEMENT_*` tunables; see the ingestion section of `CONTRIBUTING.md`.

Also present in `.env` for the hosted parser: `UNSTRUCTURED_API_KEY` and `UNSTRUCTURED_URL` (the v2 Transform API). A planned `PARSER_BACKEND` (`local` in development, `hosted` in production) will choose between local hi_res and the hosted Parse API; see `docs/TRADEOFFS.md`, section 1.

## Decisions made

Recorded so no agent re-opens them. Dates are when the owner decided.

- 2026-11-12: **Postgres + pgvector** replaces Chroma and also holds application data (documents, chats, cache, and later users). Existing chunks, summaries, and embeddings are copied across; nothing is re-summarized.
- 2026-11-12: **API shape changes** (Checkpoint 2):
  - remove `images_base64` from responses;
  - serve figures from disk at stable URLs `/v1/images/{record_id}/{position}`, where `record_id` is the Chroma record UUID. The server looks the file path up in the store and never takes a path from the URL. Changed on 2026-11-14 from `{chunk_id}`: the stored `chunk_id` restarts at 1 in every document (78 distinct values across 270 records), so it cannot identify a figure;
  - merge `/query` and `/query/custom` into one `POST /query` with an optional `llm: {provider, model, api_key, temperature?, max_tokens?}` object. `api_key` is required whenever `llm` is present, so a caller cannot run the server's own key against a model of their choice. No reasoning field for now: the `thinking` option was dropped on 2026-11-15 and native reasoning control is deferred (see Checkpoint 2);
  - add `POST /query/stream` for streaming; it replaces `GET /query/stream`, takes the same body as `POST /query` (including `max_images` and `llm`), and the two routes are the whole query API;
  - return real HTTP errors, with 502 when the LLM provider fails.
- 2026-11-15: **Two query routes, not one with a stream flag.** After Checkpoint 2 the query API is `POST /v1/query` (JSON) and `POST /v1/query/stream` (SSE), with one shared body; model choice is the `llm` object in that body, never a path (`/query/custom` goes). OpenAI and Anthropic use one route with `stream: true`, but Google Gemini (`generateContent` and `streamGenerateContent`), AWS Bedrock (`Converse` and `ConverseStream`), and LangServe (`/invoke` and `/stream`) use separate routes, so both patterns are mainstream. Two routes keep one response type per route, fully described in `/docs`, and a short handler each. POST, not GET, because the body can carry an API key, which must never be in a URL (URLs land in access and proxy logs), and because answers cost tokens and vary, so they must not be cached or retried like a GET. If OpenAI compatibility is ever wanted, add a separate `/v1/chat/completions` adapter.
- 2026-11-22: **Neon hosts Postgres; the API runs on Render.** The backend is already deployed on Render, so Railway's one advantage (private networking next to the API) no longer applies, and Railway's free offer is a one-month trial. Neon has a lasting free plan, pgvector, scale to zero and database branches, which suit the migration check (load a branch, compare it with Chroma, discard it if wrong). Costs accepted: a cold start of about a second after five idle minutes, and the API reaches the database over the public internet (TLS), so the Neon region should match the Render region. Use Neon's pooled connection string; with psycopg 3 that means turning prepared statements off. Tests stay offline. Render's own Postgres was not chosen: its free database expires after a short period.
- 2026-11-13: **`/v1` prefix** on every API endpoint except `/` and `/health`, which stay unversioned for uptime probes. The planned endpoint list is in `docs/TRADEOFFS.md`, section 8.
- 2026-11-13: **Component choices.** `docs/TRADEOFFS.md` marks every option in its tables as (chosen), (future option), (to research), or (not pursued). Alternatives are never deleted. In short:
  - extraction: local unstructured hi_res in development, Unstructured's hosted Parse API in production, switched by one environment variable; free Render or Railway tiers cannot run hi_res;
  - chunking: by title only;
  - enrichment: keep the existing summaries; research a hosted multimodal model that takes 5 to 6 images per call, a local vision model (framework and model), and multi-vector embeddings;
  - embeddings: watsonx Granite 278M; no local sentence-transformers;
  - retrieval: hybrid search once on Postgres; metadata filtering by document and document type still to design;
  - generation: Gemini Flash-Lite stays the default;
  - frontend: React with TypeScript in a later phase; Next.js or Vite decided then;
  - database host: Railway Postgres, tentatively; replaced on 2026-11-22 by Neon, with the API on Render (see above).
- 2026-11-13: **First upload target is datasheets.** Privacy (local mode) is a later add-on. The demo user is a mix of student and working engineer.
- 2026-11-21: **Postgres before the evaluation baseline.** Checkpoint 6 runs before Checkpoint 5; the numbers stay as they are because docs and commits refer to them. The migration copies the same embeddings under the same record ids, so it needs a check that pgvector returns what Chroma returns, not a measure of quality: about 50 real questions, comparing the record ids of the 20 nearest chunks and of the final 5 after MMR. The golden set refers to record ids, which the migration keeps, so it is built on Postgres afterwards and still comes before any change meant to alter results (hybrid search, re-ranking, chunking, a new summarizer or embedding model, the cache).
- 2026-11-13: **Working style.** Keep the checkpoint order (changed once, on 2026-11-21: Checkpoint 6 before 5). There is no fixed deadline; understanding comes before speed. For the evaluation set, the agent drafts questions with expected answers and pages, and the owner reviews every one.
- A chosen option is not frozen. When working on an area, give your own opinion and ask if you see a better path, then proceed with the chosen option unless the owner changes it.
- Auth is deferred to a later phase.

## Checkpoints for the current phase

Each checkpoint is a set of small commits. Finish one, let the owner test and push it, then start the next. Items marked "decision" still need the owner's answer before work starts. Status lives here, so update it when a checkpoint is done.

**Checkpoint 0. Ground rules and docs.** Status: done.
`AGENTS.md`, `docs/TRADEOFFS.md`, `docs/PRODUCT_RESEARCH.md`.

**Checkpoint 1. Repository hygiene.** Status: done (2026-11-14).
- Done: removed `chunks_export.json`, `rag_results.json`, `notebooks/raw.md`, `scripts/test_single_image_trace.py`, and the `last_prompt.txt` debug write. The owner chose to keep `parsed_pdf/` and the notebook; `queries.txt` was removed on 2026-11-16.
- Done: dropped `torch`, `transformers`, and the OpenCV system libraries from the API image; a local `docker build` and `/health` check passed on 2026-11-13. `docker run --env-file` does not strip quotes, so `.env` values must be unquoted.
- Done: `.dockerignore` is an allowlist (`api/`, `src/`, `config/`, `dbv2/`, `requirements-api.txt`). The image went from 1.34 GB to 1.17 GB. The list moved to `api/requirements.txt` on 2026-11-17, so `api/` now brings it in.
- Done: fixed the LM555 data bug on 2026-11-14. `json/LM555TImer.json` was a copy of the TLV1117 export, and the store held TLV1117 twice (the second copy under the source `chunks_huggingface`) and no LM555. The real LM555 export (36 chunks, formerly the root `chunks_huggingface.json`) now lives at `json/LM555TImer.json`, its 36 summaries were embedded with watsonx, and the 78 duplicate records and their 74 figure files were removed. The store holds 270 records and 267 figures.
- Done: rewrote the README for the current stack (watsonx, Gemini 3.5 Flash-Lite, k=10 of 20) with the setup, API, Docker, and test commands, and corrected the storage label in the dashboard's Home page diagram.
- Source names cleaned up on 2026-11-16: every document's `source` is its PDF's file name in `data/` without `.pdf` (`AN699`, `lm555`, `lm2596`, `lm317`, `ne555`, `tlv1117`), as `scripts/ingest.py` names new documents. The exports (`json/<source>.json`), the figure files (`dbv2/images/<source>_c<n>_img<i>.jpg`), `data/datasheet.pdf` (now `data/lm555.pdf`) and each record's `source` and `image_paths` were renamed in place; record ids, summaries and embeddings did not change, so figure URLs still work.
- The amd64 image was built with `docker buildx build --platform linux/amd64` and tested by the owner on 2026-11-14.

**Checkpoint 2. Backend structure.** Status: done (2026-11-16); tested and merged by the owner.
- Split `api/main.py` into `api/routers/system.py`, `retrieval.py`, `query.py`, `images.py`. `main.py` keeps only app creation, CORS, the lifespan, and `include_router` calls. Done 2026-11-14.
- Merge the default and custom generation paths into one function that receives an LLM. Done 2026-11-15: `api/custom_service.py` and the non-streaming copy are deleted; `RAGService.stream_answer` picks the default model.
- Remove the hidden 5-chunk limit. Moved here from Checkpoint 3 on 2026-11-14. Decided 2026-11-15: `top_k` is what the model gets (every retrieved chunk's text and tables), default 5 (was 10); figures are capped at an absolute count, `GENERATION_MAX_IMAGES` = 3, taken in chunk order; both are settings a request can override. The search type (`similarity`, `mmr`, `similarity_score_threshold`) and its parameters are settings too. Re-ranking (for example, pool 20 and re-rank to 5) waits until the evaluation results show it is needed. Done 2026-11-15.
- Apply the agreed API changes, including the `/v1` prefix (see Decisions made). These are breaking, so update `app/api_client.py` and `docs/API.md` in the same checkpoint. `images_base64` and `response_id` are removed from responses and the stream's `done` event (2026-11-16); building a response reads no image file (`src/documents.py:chunk_text_and_tables`), and `image_count` counts image URLs. Stable image URLs are done (2026-11-16): `/v1/images/{record_id}/{position}` reads the path from the record's `image_paths` in the store, serves it only if it lies inside `IMAGES_DIR`, and `ImageRegistry` is deleted; each chunk carries `record_id`. Both ingestion scripts write figure files (2026-11-16): `scripts/ingest.py` exports to `json/<pdf name>.json` and stores records through the same `src/ingestion/pipeline.py:documents_from_export` as `ingest_colab.py` (deleted later that day), so `src/documents.py` reads only the flat layout. The `llm` object is done (2026-11-16): `/v1/query/custom` is merged into `POST /v1/query`, `api_key` is a `SecretStr`, 422 bodies no longer echo the rejected value, and the unused `api_key_env` field is gone from `config/providers.py` (no provider has a server-side fallback key). The `/v1` prefix is done (2026-11-15): `api/main.py` adds it when including the retrieval, query and image routers, the old unprefixed paths were removed rather than kept as aliases, and `/` and `/health` stay unversioned.
- Replace `_create_llm_for_model` in `api/custom_service.py` with a factory in `src/` that chooses the client by provider id, not by model-name prefix. Every provider's `base_url` lives in `config/providers.py`, including Groq's. An unknown provider is rejected instead of silently going to Groq. Done 2026-11-15 (`src/llm.py:create_llm`).
- Prune the model catalogue. On 2026-11-13, 5 of the 9 Groq models listed (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3.6-27b`, `groq/compound`, `groq/compound-mini`) returned 404 for the owner's key. Consider checking against each provider's `/models` endpoint. Done 2026-11-15 (the 5 models removed).
- Drop the `thinking` option. Done 2026-11-15. It appended "show your reasoning in `<think>` tags" to the prompt, which does not change how a model works, and made `create_llm` send Groq's `reasoning_effort` as `none` or `default` to every Groq model, although Groq GPT-OSS accepts only `low`, `medium`, `high`. The request field, the prompt text, the factory argument, and the dashboard's Thinking Mode checkbox and `<think>` parsing are removed; each provider's default reasoning applies (Gemini 3.5 Flash-Lite defaults to `minimal`). Native reasoning control is deferred. Research kept for when it returns:
  - Both LangChain clients accept one parameter, `reasoning_effort`: `ChatGoogleGenerativeAI` maps it to Gemini's `thinking_level`, and `ChatOpenAI` sends it as is (OpenAI, Groq, and the Hugging Face router, which passes it through, though not every provider honours it).
  - Allowed values differ per model: Gemini 3.5 Flash-Lite and 3.6 Flash take `minimal`, `low`, `medium`, `high`; Gemini 3.1 Pro takes `low`, `medium`, `high`; Groq GPT-OSS takes `low`, `medium`, `high`; Groq Qwen 3.8 takes `none`, `default`, `low`, `medium`, `high`; GPT-5.4 takes `none` to `xhigh`; GPT-4o and GPT-4.1 have none. They would be recorded per model in `config/providers.py`.
  - Groq Qwen returns its reasoning inside `<think>` tags in the answer unless `reasoning_format` is `hidden` or `parsed`.
  - Reasoning tokens count against `max_tokens`, so the default of 512 can leave no room for the answer.
  - Open questions then: whether to show the reasoning text, and how `max_tokens` behaves when reasoning is on.
- Send `temperature` only to models that accept it. OpenAI's o-series reject a custom value, and Gemini's fixed-sampling models (`gemini-3.5-flash-lite`, `gemini-3.6-flash`) and GPT-5 ignore it. Recorded per model as `accepts_temperature` in `config/providers.py`. Done 2026-11-15.
- Check the model can read images before sending them. Done 2026-11-15: the catalogue marks the Groq and Hugging Face models `supports_vision=False`, but the prompt always included figures. Now `src/llm.py:accepts_images` reads that flag, and a text-only model gets the chunks' text and tables with no image blocks and no mention of figures in the instructions; `usage.prompt_images` reports 0. The response still lists the chunks' figures.
- Per-model output budget. Done 2026-11-15: Groq GPT-OSS spent all 512 output tokens reasoning and returned an empty answer with status 200. `default_max_tokens` in `config/providers.py` is 2048 for models that reason by default (Groq and Hugging Face GPT-OSS, OpenAI o-series and GPT-5.4, Gemini 3.6 Flash, 3.1 Pro, 2.5 Pro and 2.5 Flash) and 512 otherwise; a request's `llm` object uses it when `max_tokens` is left out, and the dashboard's Max Tokens 0 means the same.
- An empty answer is an error. Done 2026-11-15 with real HTTP errors: a model that writes no answer text is a 502, and when `finish_reason` is `length` or `MAX_TOKENS` the detail says the output budget ran out.
- Real HTTP errors. Done 2026-11-15: `src/llm.py` defines `LLMProviderError` (502) and `UnknownModelError` (422 with an unknown provider or model). `api/service.py` defines `StoreUnavailableError` (503, also on `/health`). Unexpected errors before a response starts are a 500 with `"Internal server error"` as the detail.
- Streaming POST migration. Done 2026-11-16. Both query routes take `QueryRequest`, including the optional `llm` object. `POST /v1/query/stream` waits for the first answer fragment, so a provider failure before it returns HTTP 502. Failures after the headers are sent end the stream with an `error` event carrying `status` and `message`. Changed 2026-11-19 by the owner: the `retrieval` event is sent as soon as the chunks are found, before the model is called. Measured that day, retrieval took under 2 s but Gemini 3.5 Flash-Lite took 40 to 55 s to its first fragment (and then sent the answer in 2 or 3 pieces), so the chunks waited for nothing. Store and `llm` errors are still HTTP 503 and 422; every model failure on the stream is now an `error` event (`status` 502), and `POST /v1/query` keeps the real HTTP 502.
- Trace structure (done 2026-11-15). `answer_query_stream` and `generate_answer_stream` are real generators, so LangSmith keeps each run open until the last token and records the answer as its output. `_StreamWrapper` is gone, `GenerateAnswerStream` is a `chain` run, and LangChain's own model run is the only `llm` run. Trace inputs hold only the question and settings; the model id is run metadata, and the chat model object (which holds the caller's key) is never recorded.
- Return the figures the model actually saw. Add `figures` to the `/v1/query` response (and the `done` stream event): the image URLs sent in the prompt, in prompt order, each with its chunk and source document. Label each image in the prompt (`Figure F1, from <document>, chunk <n>`) so the model can cite `[F1]`. This is the baseline for citations; see Later phases for highlighted text. Done 2026-11-16: `src/retrieval/prompt.py:select_figures` chooses the figures once (an unreadable figure takes no label or slot), and that one choice builds the prompt, the trace attachments (only the sent figures, named `F1`, `F2`, ...) and the `figures` list (`label`, `url`, `chunk_id`, `source`) in `/v1/query` and the `done` event. Figures carry their file path, not bytes: the bytes are read only for the model request and the trace attachment, and the prompt, like the image endpoint, uses only files inside `IMAGES_DIR` (`src/documents.py:image_file`).

**Checkpoint 3. Pipeline readability and known bugs.** Status: done (2026-11-17); tested and merged by the owner (pull requests 3 and 4).
- Docstrings everywhere, remove all banner comments. Done 2026-11-17: the 55 banners left (not 75) became plain comments or were deleted, and every function in `api/`, `src/`, `config/` and `app/api_client.py` has a docstring. The two dashboards still miss about 50 function docstrings; they are written during the Checkpoint 4 split, which rewrites those functions.
- Split `src/retrieval/generate.py` into prompt building, the LLM factory, and streaming. Done 2026-11-17: `src/retrieval/prompt.py` builds the messages and chooses figures (`build_messages`, `select_figures`, `prompt_stats`), `default_llm` moved to `src/llm.py` next to `create_llm`, and `generate.py` keeps retrieval-then-stream, token counting, pricing and tracing (454 lines to 302).
- Fix:
  - only the first 5 chunks' text reaches the prompt while images come from all chunks (moved to Checkpoint 2 on 2026-11-14, done 2026-11-15);
  - streamed usage tokens reporting 0. Done 2026-11-17, two causes. `ChatOpenAI` asks for streamed token counts (`stream_options.include_usage`) only when it talks to api.openai.com, so Groq and Hugging Face streams reported none; `create_llm` now sets `stream_usage=True`. And `ChatGoogleGenerativeAI` gives each chunk the tokens that chunk added (Gemini sends running totals and LangChain subtracts), but `_capture_usage` kept only the last chunk's numbers, so every multi-chunk Gemini stream reported 0 input tokens; it now sums the chunks. Checked against the provider docs on 2026-11-17: the Hugging Face router documents `include_usage` (one final chunk with empty `choices` and the whole request's `usage`), and the Groq SDK's chunk type has a top-level `usage`, which `ChatOpenAI` reads. Not checked live: no Hugging Face credit;
  - cost estimates hardcoded to Gemini prices. Done 2026-11-17: `_estimate_cost` uses the answering model's prices from `config/providers.py`, and `GENERATION_INPUT_PRICE_PER_1M` and `GENERATION_OUTPUT_PRICE_PER_1M` are deleted. Those settings held Gemini 2.5 Flash-Lite's prices (0.10 and 0.40 per 1M), so even the default model, 3.5 Flash-Lite (0.30 and 2.50), was priced too low. A model whose catalogue prices are both 0 reports a null cost, not $0.
- Corpus re-summarized 2026-11-16 with the local Qwen3.5-4B backend after page cleanup (AN699, lm2596, lm317, lm555, tlv1117; ne555 was already done) and replaced in `dbv2` with `--replace`: 310 records and 276 figures, longest summary 474 words (was 1,523, past the embedder's 512-token limit).
- Answer prompt rewritten 2026-11-16 (`src/retrieval/prompt.py:build_messages`). A system message holds the rules: answer only from the context, say when the documents do not state it, never mix values from different documents, keep units and conditions, cite figures as `[F1]`. The user message gives each chunk's document name, summary, original text (`raw_text`, previously never sent) and tables, each figure right after its chunk, and the question last. On three live questions it stopped presenting the LM555 diagram as the NE555's and gave the LM317 reference voltage range (1.2, 1.25, 1.3 V) where it had invented a 4.95 V maximum; prompts are about 40 to 70 percent longer.
- Dead code and dependency cleanup, 2026-11-17. Deleted `src/tracing.py` (it built a 7-field summary per chunk, of which callers read two counts, for log lines only; `retrieve_chunks` now counts directly and generation logs the prompt size instead of repeating the counts), the unused re-exports in `src/retrieval/__init__.py`, `src/ingestion/__init__.py` and `config/__init__.py`, `SUPPORT_STACK` in the dashboard, the `HUGGINGFACEHUB_API_TOKEN` setting, and the catalogue fields nothing read (`context_window`, `supports_streaming`, `description`). Dependencies: `requirements.txt` is deleted (Streamlit Cloud already used `uv.lock`, which it reads first), `pyproject.toml` no longer pretends to be an installable package (`[tool.uv] package = false`, and its `packages.find` pointed at the wrong folders), `pytest` and `httpx` are a `dev` group, Pillow is out of `requirements-api.txt` (checked with Pillow blocked: the API imports, a Gemini message with a figure converts, 103 backend tests pass), and the dashboard gets `app/requirements.txt`, which Streamlit Cloud reads before the root files. That list has no unstructured and so no OpenCV, which made `packages.txt` (`libgl1`) unnecessary; it is deleted. The deployed dashboard was not redeployed to confirm.
- Install docs, 2026-11-17: `requirements-api.txt` moved to `api/requirements.txt`, next to the code it serves, as `app/requirements.txt` is (a root `requirements.txt` would read as the full install but hold only the API). It now lists `langsmith` and `pydantic`, which our code imports directly; an import trace from `api.main` found nothing else outside the list. The docs were then split the usual open-source way: the README is the product page (why, who it is for, what it does today, how it works, quick start, roadmap, license), and `CONTRIBUTING.md` holds the full setup, the Dependencies section (which file each deployment installs, where a new package goes), ingestion, Docker, configuration, conventions and troubleshooting. Added `LICENSE` (Apache 2.0, the owner's choice) and `.env.example`. A `docker compose` quick start waits until Checkpoint 4 moves the dashboard fully behind the API. The amd64 image built from it (`docker buildx build --platform linux/amd64`) was pushed to Docker Hub and passed `/health` on 2026-11-18.
- Tables in prompts are Markdown since 2026-11-16 (`src/tables.py:table_to_markdown`, used by the answer and summary prompts). Unstructured has no Markdown table output, so tables stay stored as HTML (lossless, rendered by the dashboard) and are converted only when a prompt is built: 32 percent fewer tokens across the 118 stored tables. Merged cells are repeated, a rowspan stops at the end of its section, and empty tables are skipped.

**Checkpoint 4. Frontend separation.** Status: done (2026-11-19); tested by the owner, in review.
- `app/` reads its own settings from the environment. Done 2026-11-19: `API_BASE_URL` was read only by the dashboard, so it moved out of `config/settings.py` into `app/api_client.py`, which loads `.env` itself; the dashboard logs with the standard `logging` module instead of `src/logger.py`. No settings module: one variable has one reader. Frontend and backend each read only what their own environment needs, and share nothing but the HTTP API.
- Add a `GET /providers` endpoint so the UI stops importing `config/providers.py`. Done 2026-11-19: `GET /v1/providers` (`api/routers/providers.py`) returns `items` (each provider's `id`, `display_name` and `models`, each model with `supports_vision`, `accepts_temperature`, `default_max_tokens` and prices) and `total`, built from the same catalogue `src/llm.py` checks, so every listed model is accepted by `/v1/query` (a test sends each one). `protocol` and `base_url` stay server-side, and the route needs no store. The dashboard's model picker reads it, `list_models_for_provider` (no caller left) is deleted, and the dashboard no longer puts the repository root on `sys.path`, so `app/` cannot import the backend at all. Found the same day, not changed: the sidebar's Custom Model ID field says any model id works, but `/v1/query` rejects a model outside the catalogue with 422.
- Serve dataset browsing through the API instead of reading `json/` directly. Done 2026-11-19: `GET /v1/documents` lists each document (`id` is its `source` name) with chunk, figure and table counts, and `GET /v1/documents/{document_id}/chunks?offset=&limit=` returns its chunks in reading order (by stored `chunk_id`) with `items`, `total`, `offset`, `limit` (limit 1 to 200, default 50; an unknown document is 404, an offset past the end an empty page). The reads are in `src/browse.py` and use no embedding or vector search: Chroma's `get` with a `where` filter. Chroma cannot sort, so a document's chunks are read whole, sorted and sliced; the largest has 77. The Chunk Explorer, Compare and Analytics pages read them through `app/api_client.py:document_chunks`, which follows `total` across pages, and show figures from their image URLs; filters stay in the dashboard. The sidebar's Upload JSON box is removed (uploading through the API is Checkpoint 7). Same day: the Analytics page crashed on every document because plotly rejects numpy integers for `nbins`; fixed with `int(...)`.
- **Future: Postgres and pgvector serve these same two endpoints** (Checkpoint 6). Only `src/browse.py` changes: `list_documents` becomes a `GROUP BY source` count, and `document_chunks` an `ORDER BY chunk_id LIMIT ... OFFSET ...` query, so the database sorts and pages instead of Python. The paths, the response shapes and the dashboard stay as they are. Postgres keeps the `source` name as a unique column so `document_id` URLs do not change.
- Split `app/dashboard.py` into Streamlit pages plus shared components. Done 2026-11-19: `app/dashboard.py` (2,097 lines) is now an 80-line entry point that sets up each run and calls `st.navigation`, which serves each page at its own URL (`/`, `/explorer`, `/compare`, `/analytics`, `/chat`) so pages can be bookmarked and linked. Pages are callables in `app/views/` (the folder is not `pages/`, which Streamlit treats as its older automatic multipage mode). The sidebar shows each part only where it applies: the document picker on Explorer, Compare and Analytics, the filters on Explorer, the model settings on the Query page. Streamlit deletes a widget's value when a run does not draw it, so `app/state.py:keep_widget_values` re-assigns those keys every run, and widgets take their first value from the session defaults instead of `value=`. `log_tracing_status` was deleted: the dashboard calls no model, so it has no LangSmith runs to report. `app/rag_eval_dashboard.py` stays a separate app until Checkpoint 5.
- Home page as a product page. Done 2026-11-19: `app/views/home.py` presents KI-CRAFT to someone who reads datasheets (what it does, who it is for, the documents ready to query read live from `GET /v1/documents`, example questions, how it works, the roadmap), with links to the Query and Explorer pages. The old landing page described an earlier pipeline (Tesseract OCR, JSON exports feeding the inspector, k=10) and showed an 89.8/100 judge score from an earlier evaluation; both are gone until Checkpoint 5 measures a new baseline. The diagram (`app/components/diagram.py`) now shows the current pipeline. Reworked 2026-11-20 at the owner's request: a hero with the three actions as buttons that fit their labels, the pipeline written out step by step with its real numbers (sections of at most 3,000 characters, MMR 5 of 20, up to 3 figures, the citation check as its own step), the indexed parts listed by name and kind instead of chunk, figure and table counts, and a Current limits section. The example questions live in `app/components/example_questions.py` and are shown on the home page and the Query page; clicking one asks it on the Query page at once.
- Query page layout and live updates. Done 2026-11-19: `app/views/chat.py` reserves one slot per section and fills each as its stream event arrives: the retrieved chunks at once (about 1 to 2 s, since the API now sends them before calling the model), the answer as it streams, then the cited figures above the answer, the other figures below it, the chunks marked as cited, and token usage with running totals for the session. The empty "Retrieved N chunks" status box and the separate Referenced Chunks list are gone. Both model modes stream (the custom-model path used `POST /v1/query` and showed nothing until the end). The sidebar shows Retrieved Chunks and Max Images always, and the provider, model and key only when "Use my own model" is ticked. `api_client.query` and `api_client.retrieve` had no caller left and were deleted. On 2026-11-19 Gemini 3.5 Flash-Lite answered in 8 s on one try and failed mid-answer with its own 503 (high demand) on another; the page kept the chunks and the partial answer and showed the error.
- Show the `figures` strip under each answer, with the figures the answer cites (`[IMG1]`) shown first. Done 2026-11-19: `app/dashboard.py:_render_figures` shows "Figures the model saw" under the answer, ordered by `cited_first` (cited figures in the order the answer first cites them, then the rest in prompt order), each captioned with its label, cited or not, document and chunk; a non-empty `citations.unknown` shows a warning. `app/api_client.py` makes each `figures[].url` absolute, and the figures and citations are kept in session state so they survive reruns.
- Prompt labels that cannot collide with the datasheet's own figure numbers. Done 2026-11-19: chunk text quotes captions such as "Figure 3. Output Voltage ...", and our label "Figure F3" sat right next to them, so the model mixed the two numberings. Rewriting "Figure N" in the chunk text to our label was measured and rejected: of 144 chunks with figures only 38 have as many captions as images, 73 have none, and many mentions point to figures in other chunks or not sent, so a find-and-replace would assert wrong mappings. Instead chunks are labelled `C1`, `C2`, ... and images `IMG1`, `IMG2`, ... (`src/retrieval/prompt.py:chunk_label`, `figure_label`); the rules ask for `[C1]` after each statement and `[IMG1]` for an image, say the image labels are not the document's figure numbers, and allow "Figure 3 [IMG1]" only when that image is the figure. The API's `figures[].label` values changed from `F1` to `IMG1`. Storing each image's real caption at ingestion (unstructured's `FigureCaption` next to `Image`) belongs to Checkpoint 7.
- Check citations in code. Done 2026-11-19: a prompt can ask for correct citations but cannot guarantee them, so after generation `src/retrieval/citations.py:find_citations` reads the answer's bracketed labels and sorts them into cited chunks that exist, cited figures that were sent, and `unknown` labels that match nothing. `/v1/query` and the stream's `done` event carry them as `citations` (`chunks`, `figures`, `unknown`). This is deterministic; whether a valid citation is the right one (the image really shows the claim) is measured by the Checkpoint 5 evaluation.
- Cleanup, 2026-11-20, after a scan of the whole repository found no function without a caller but several duplicates. Ingestion has one write path: a PDF becomes an export and is stored by `src/ingestion/pipeline.py:store_export`, like an existing export, so it gets the same checks and refuses a document already stored; `create_vector_store` is deleted, and the store is opened only by `src/retrieval/search.py:load_vector_store`. The dashboard draws a chunk's content with one function, `app/components/chunk_view.py:render_content_tabs`, on the Explorer, Compare and Query pages, and the Explorer and Analytics pages share one per-chunk table (`build_chunk_df`) and one totals row (`components/dataset.py:render_totals`). `tests/test_api.py` is split into `tests/api/`, one file per router, with the fakes in `fakes.py` and the fixtures in `conftest.py`. `src/logger.py` lost its unused options (81 lines to 42), and `scripts/ingest.py` its `sys.path` line, since it runs as `python -m scripts.ingest`.
- Send the sidebar's Max Images in default mode too. Found 2026-11-15: without a custom provider the dashboard calls `/query/stream` with `top_k` only (`app/dashboard.py`, `api_stream(...)`), so the Max Images input is ignored and the server default `GENERATION_MAX_IMAGES` (3) applies. Only custom-provider mode sends it. Done 2026-11-16 in the POST stream migration: default mode passes `max_images` (`app/dashboard.py`, `api_stream(...)`); on 2026-11-19 the request body was captured and carries the sidebar value, 0 included.

**Checkpoint 6. Postgres + pgvector.** Status: done (2026-11-24), not yet tested by the owner. Runs before Checkpoint 5 (decided 2026-11-21). Host: Neon (decided 2026-11-22), in the same region as the Render API.
- Schema for documents, chunks (with `vector(768)`), images, queries, retrievals, answers, and feedback, with Alembic migrations.
  Started 2026-11-22: `src/db/schema.py` and the first migration (`migrations/versions/0001_documents_chunks_images.py`) create `documents`, `chunks` and `images` and enable pgvector; `alembic upgrade head`, `alembic check` (no drift), `downgrade base` and `upgrade` again ran on the Neon `dev` branch. The other tables come with the code that writes them. The Chroma collection uses cosine distance, so pgvector searches with `<=>`. Packages added: `sqlalchemy` (Core only, no ORM), `psycopg[binary]`, `alembic`, `pgvector`.
- A one-off script copies ids, summaries, embeddings, and metadata out of Chroma into Postgres. No summarizer and no re-embedding.
  Done 2026-11-23: `scripts/chroma_to_postgres.py` copied 6 documents, 310 chunks and 276 figures into the Neon `dev` branch in one transaction, then read every row back and found it identical to Chroma (ids, text, tables, figure paths, embeddings bit for bit). It refuses to run on a non-empty `chunks` table; `--check-only` repeats the comparison. `src/db/connection.py:sqlalchemy_url` (shared with `migrations/env.py`) points Neon's `postgresql://` URLs at psycopg 3.
- Each Chroma record id becomes the chunk's primary key, stored as the Postgres `uuid` type, unchanged. Figure URLs (`/v1/images/{record_id}/{position}`) contain this id, so generating new ids would break every figure link a client holds.
- Swap the retriever to pgvector and check it returns what Chroma returns: for about 50 real questions, the same record ids among the 20 nearest chunks and the same final 5 after MMR.
  Switched 2026-11-23: the API reads only Postgres. `RAGService` holds the embedding client and one engine (`src/db/connection.py:create_db_engine` on `DATABASE_URL`, pooled), checked with `SELECT 1` at startup (503 when it fails, detail `Database unavailable: ...`); each request gets a `PostgresRetriever`; `src/browse.py` lists documents, pages a document's chunks and looks up one figure's path with plain SQL, as planned in Checkpoint 4; the figure endpoint still serves only files inside `IMAGES_DIR` (`src/documents.py:figure_file`). Checked live against Neon `dev`: `/health`, the per-document chunk, figure and table counts (identical to Chroma's), a chunk page, a figure (200, image/jpeg), 404 for an unknown document and position, and `/v1/retrieve`. The `/health` body said `vector_store` until 2026-11-24; it is now `{status, database: "connected"}` (owner's choice). The API tests fake the engine-backed reads (`tests/api/fakes.py:FakeStore`). The API image installs `sqlalchemy`, `psycopg[binary]` and `pgvector`.
  Checked 2026-11-23 with `scripts/compare_search.py` on the 76 golden-set questions (one watsonx embedding per question, sent to both stores): the final 5 after MMR match on all 76, in the same order; the nearest 20 match on 75, and on the other one two chunks tie at exactly the same distance at ranks 20 and 21 and the stores break the tie differently. Postgres now breaks ties by record id. Found on the way: LangChain's Chroma returns MMR's picks in nearest-first order, not in the order MMR picks them, and that order sets the `C1`..`C5` labels and which figures fit the image cap, so `search_postgres` sorts its picks the same way. That run took watsonx to 90 percent of its monthly embedding quota (271,140 of 300,000 tokens).
- Ingestion writes Postgres. Done 2026-11-23: `src/ingestion/pipeline.py:store_export` checks the export, refuses a stored document unless `replace`, embeds every summary, writes figure files (`rows_from_export`), then `write_document` deletes the old document (cascade) and inserts the new rows in one transaction; stale figure files go last. `scripts/ingest.py` lost `--store`; it writes to `DATABASE_URL_DIRECT`. A new database is filled with `alembic upgrade head` and `python -m scripts.ingest json/*.json` (re-embeds about 100,000 tokens, new record ids); `CONTRIBUTING.md` has a Database section for contributors (Neon, the `pgvector/pgvector` Docker image, or any Postgres with pgvector). Checked live on Neon `dev` with a two-chunk test document: stored, refused without `--replace`, replaced with new record ids, then deleted.
- Remove Chroma once they match. Done 2026-11-24:
  1. The Neon `production` branch was filled by copying, not re-ingesting (`alembic upgrade head`, then the copy script with `DATABASE_URL_DIRECT` pointed at `production` for those two commands): 6 documents, 310 chunks, 276 figures, same record ids, so figure URLs stay the same; the read-back check found it identical to Chroma.
  2. Deleted: `scripts/chroma_to_postgres.py`, `scripts/compare_search.py` and their test, `load_vector_store`, `dbv2/chroma_db`, the `CHROMA_*` settings, `langchain-chroma` (and with it about 600 lines of `uv.lock`), and the `git checkout -- dbv2/chroma_db` notes. Git history keeps the scripts.
  3. Left for deployment: on Render, set `DATABASE_URL` to the `production` pooled string before the new image goes live (Checkpoint 9 lists it too).
  The last Chroma leftover, `src/documents.py:_parse_json_value` (it accepted the JSON strings Chroma stored), was deleted the same day; Postgres returns lists.
  Added 2026-11-24: `scripts/copy_document.py` copies documents from `dev` to `production` row for row (`src/db/store.py:copy_document`, one transaction per document, refuses an existing one without `--replace`), so new datasheets are embedded once. `is_stored` and `write_document` moved from `src/ingestion/pipeline.py` to `src/db/store.py`, which both use. Checked live with a throwaway two-chunk document: copied identical (ids, text, tables, figures, vectors), skipped on a second run, replaced with `--replace`, an unknown name failed, then deleted from both branches. A connection that drops during a request is now a 503, not a 500 (`api/main.py:database_lost`).
  Cleanup the same day, chosen by the owner after a scan found no function without a caller: deleted `notebooks/` (the Chroma-era notebook), `parsed_pdf/` (42 MB of annotated PDFs it drew), `app/rag_eval_dashboard.py` with `evals/testing.json`, and the unused `assets/dashboard-compare.png`.
  The work is on the branch `feat/postgres-pgvector`.

**Checkpoint 5. Evaluation baseline.** Runs after Checkpoint 6, on Postgres.
- Turn the old `evals/testing.json` (deleted 2026-11-24 with `app/rag_eval_dashboard.py`, its viewer) into a small golden set covering text, table, figure, and "not stated" questions per document. The agent drafts each question with its expected answer and page; the owner reviews every one.
  Drafted 2026-11-21 as `evals/golden_set.json` (76 questions, on `main`). The review page `evals/review.py` and its doc lines are parked on the local branch `feat/golden-set-review` (2026-11-22); merge or rebase it when Checkpoint 5 resumes.
- One script reports retrieval hit rate and answer correctness.
- Record the baseline in `docs/TRADEOFFS.md` so later changes (chunking, embeddings, hybrid search, cache) are measured, not guessed.

**Checkpoint 7. Upload and ingest through the API.** This is the product goal of the phase.
- Keep each element's page number and bounding box from `unstructured` (`metadata.page_number`, `metadata.coordinates`) on the chunk. Highlighted citations depend on this, and it cannot be recovered later without re-extracting.
- `POST /documents` uploads a PDF and starts an ingestion job.
- `GET /documents/{id}` reports the job's status.
- Queries can be scoped to one or more documents.
- Research, then decide: the summarizer for new uploads (the Hugging Face credit is gone). Candidates are a hosted multimodal model taking 5 to 6 images per call, a local vision model, and multi-vector embeddings (`docs/TRADEOFFS.md`, section 3).
- Merge the two ingestion entry points into one. Done 2026-11-16: `scripts/ingest.py` takes PDFs or `json/` exports and stores records through `src/ingestion/pipeline.py:documents_from_export` (figure files in `dbv2/images`, flat metadata); `scripts/ingest_colab.py`, which added duplicates when re-run, and `scripts/query.py` are deleted. Since 2026-11-16 `scripts/ingest.py <export>.json --replace` replaces a stored document safely (`src/ingestion/pipeline.py:store_export`): it checks the export, embeds every summary, and only then deletes the document's old records and figure files and writes the new ones, so a failed embedding call leaves the store as it was. Re-storing a document gives its chunks new record ids, so its old figure URLs return 404.
- Parser backend switch (`PARSER_BACKEND=local|hosted`), one adapter that turns local and hosted elements into the same shape, and figure cropping from page renders, because the hosted Parse API returns image boxes but no image bytes (`docs/TRADEOFFS.md`, section 1).

**Checkpoint 8. Semantic cache.**
- Cache answers keyed by question embedding similarity plus an exact match on part numbers. The key also includes the model, `top_k`, and the corpus version, and failures are never cached.
- Optional: a small router model that decides between a cache hit, retrieval, or a direct answer.

**Checkpoint 9. Deployment.**
- Set the watsonx and database (Neon) variables on the Render service.
- Set `API_BASE_URL` in Streamlit Cloud secrets.
- Document the deploy in `CONTRIBUTING.md`.

## Later phases (not now)

Recorded so the current work does not block them. Do not start these until the checkpoints above are done.

- **React frontend.** A TypeScript and React web app (Next.js or Vite, decided then) replaces Streamlit as the product UI, and Streamlit stays as an internal inspection tool. It is possible only because the UI talks to the backend over HTTP alone, which Checkpoint 4 finishes. The owner will also use it to practise JavaScript fundamentals.
- **Highlighted citations.** The UI renders the source PDF page (for example with pdf.js) and highlights the exact passage or table cell an answer came from, using the page numbers and bounding boxes kept in Checkpoint 7. This goes further than citing a file name.
- **PCB design agent.** The long-term direction described at the top of this file.

## Things not to do

- Do not start the PCB design agent, MCP servers, web search, or LangGraph work in this phase.
- Do not change public API shapes, delete files, add dependencies, or change models without asking.
- Do not commit `.env`, `.streamlit/secrets.toml`, or anything containing a key.
- Do not re-index or re-summarize the corpus without saying so first. It costs API calls and rewrites stored data.
- Do not leave dead code "for later". Delete it; git history keeps it.
