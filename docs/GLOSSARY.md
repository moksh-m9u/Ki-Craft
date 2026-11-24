# Glossary

Plain-language explanations of the technical terms used in this repository. Each entry says what the term means and where it shows up here. Add a term in the same change that first uses it.

## Retrieval and embeddings

**Embedding.** A list of numbers (here 768) that represents the meaning of a piece of text. Texts with similar meaning get vectors that point in similar directions. Here: `src/embed.py`, watsonx Granite.

**Cosine similarity.** How close two embeddings are, measured by the angle between them: 1 means the same direction, 0 means unrelated. pgvector ranks chunks by it here (`<=>` is cosine distance, 1 minus the similarity), as Chroma did before.

**Vector store.** A database that stores embeddings and finds the nearest ones to a query embedding. Here: Postgres with pgvector (Chroma until 2026-11-23).

**HNSW (Hierarchical Navigable Small World).** The index that makes nearest-neighbour search fast. It builds a layered graph of vectors and walks it towards the query instead of comparing against every vector. The result is approximate but very close to exact. pgvector offers it; with a few hundred chunks this project compares every row instead, which is exact and fast enough.

**MMR (Maximal Marginal Relevance).** A way to pick results that are relevant to the question but different from each other. Here it fetches the 20 nearest chunks and keeps 5 (`top_k`), so the prompt does not get five copies of the same table. `src/retrieval/search.py`.

**top_k.** How many chunks retrieval returns for a question. Here every returned chunk's text and tables go into the prompt, so `top_k` is also the model's context size. Default `RETRIEVAL_K` = 5; a request can override it.

**Image cap (`max_images`).** The most figures sent to the model per question, taken in chunk order. Images cost far more input tokens than text, so this is the main cost control. Default `GENERATION_MAX_IMAGES` = 3; 0 sends every image.

**Search type.** How the vector store picks chunks for one question, set by `RETRIEVAL_SEARCH_TYPE`. `similarity` returns the k nearest; `mmr` returns k relevant but varied chunks from a larger pool; `similarity_score_threshold` returns only chunks scoring above a cutoff. These are LangChain's three `as_retriever` options. `src/retrieval/search.py`.

**lambda_mult.** MMR's balance between relevance and variety: 1 ranks by relevance alone, 0 by difference from the chunks already picked. LangChain's default is 0.5.

**Score threshold.** The minimum relevance score a chunk must reach to be returned. With cosine distance the score is 1 minus the distance. Unlike top-k it can return nothing, which is useful for "not in the documents" answers but needs tuning.

**Re-ranker.** A second, slower model that rescores retrieved candidates against the question, reading both together (a cross-encoder). The usual pattern retrieves 20 to 100 candidates and keeps the best 3 to 10. Not used here yet.

**Hybrid search.** Running a keyword search and a vector search together and merging the results. Keyword search catches exact tokens such as `TLV1117-33` or `VREF` that embeddings blur. Planned once the data is on Postgres.

**RRF (Reciprocal Rank Fusion).** A simple way to merge two ranked lists: each result scores `1 / (k + rank)` in every list it appears in, and the scores are added. It needs no tuning of score scales, which is why hybrid search usually uses it.

**Multi-vector embeddings.** Storing more than one vector per chunk (for example one for the raw text and one for the summary) and matching a question against any of them. Better recall, more storage.

**Vision (multimodal) model.** A model that accepts images in the prompt as well as text. `supports_vision` in `config/providers.py` records it per model; a text-only model is sent no figures (`src/llm.py:accepts_images`).

**Temperature.** How random a model's word choice is. At 0 it always picks the most likely next token, so answers repeat; higher values vary more. Some models fix it: Gemini 3.5 Flash-Lite, the default here, ignores it, which is why `accepts_temperature` exists in `config/providers.py`.

**OpenAI-compatible API.** A provider that accepts the same HTTP requests as OpenAI's chat API at its own address. Groq and the Hugging Face router are, so one client (`ChatOpenAI` with a different `base_url`) serves all three. The provider's `protocol` field records this.

**Generator.** A Python function that contains `yield`. Calling it runs nothing; it returns an iterator, and each `next()` runs the body up to the next `yield` and hands back that value. Here `generate_answer_stream` yields answer tokens one at a time, so the API can forward each token as soon as the model produces it, and `answer_query_stream` yields events: the chunks first, then tokens, then usage.

**Dataclass.** A Python class declared with `@dataclass`, which writes the constructor, equality and printing from a list of typed fields. `src/chunk.py:Chunk` is one: every step of the query path passes chunks as `Chunk(record_id, source, chunk_id, summary, raw_text, tables_html, image_paths)`, so a reader sees what a chunk holds and a typo in a field name fails at once, where a string key in a metadata dict would silently read nothing. `frozen=True` makes the fields read-only after creation.

**LangSmith trace (run).** LangSmith records each call to a function decorated with `@traceable` as a run, with its inputs, outputs and timing; runs started inside another run nest under it, and the whole tree is one trace. Here every query is one `AnswerQuery` run with `RetrieveChunks` and `GenerateAnswerStream` under it, and LangChain's own model run (for example `ChatGoogleGenerativeAI`, with tokens and cost) under that. When the decorated function is a generator, the run stays open until the last item and `reduce_fn` turns the yielded items into its output. `src/retrieval/generate.py`.

**Run type.** The kind LangSmith files a run under: `chain` for a step of our own, `retriever` for retrieval, `llm` for a model call. Only `llm` runs get model, token and cost columns, so only the real model call (LangChain's run) uses it here.

**reduce_fn.** An argument to `@traceable` for generator functions: it receives everything the generator yielded and returns what the run records as its output. Here it joins the tokens into the answer.

**Factory.** A function whose job is to build and configure an object, so callers ask for what they need instead of assembling it. Here `src/embed.py` builds the embedding client and `src/llm.py` builds chat clients.

## Ingestion

**Partitioning.** Splitting a PDF into typed elements (`Title`, `NarrativeText`, `Table`, `Image`, and so on). Here: `unstructured`, `src/ingestion/extract.py`.

**hi_res strategy.** The `unstructured` mode that renders each page as an image and runs a layout detection model on it to find tables and figures. Accurate but slow and heavy, which is why production uses the hosted API.

**Page decoration.** What is printed on every page rather than written about the content: the running header (part numbers, document number, revision date), the footer, page numbers and the vendor logo. `src/ingestion/extract.py:remove_page_decoration` drops it before chunking. Headers and footers go by the type `unstructured` gives them; an image is a logo when its OCR text is short (6 words or fewer) and either repeats on 3 or more pages or contains a vendor name from `PDF_LOGO_WORDS` (OCR sometimes reads only part of a logo, such as "instruments").

**Merged cells (rowspan, colspan).** An HTML table cell can cover several rows (`rowspan`) or columns (`colspan`), as a "MIN/TYP/MAX" header often does in a datasheet. Markdown tables cannot merge cells, so `src/tables.py:table_to_markdown` repeats a merged value in every cell it covers, and stops a `rowspan` at the end of its section (`thead`, `tbody`), as a browser does. Tables are stored as HTML and turned into Markdown only for prompts, where they take about 30 percent fewer tokens.

**Chunking by title.** Grouping elements into chunks that start at each section heading, so a chunk follows the document's own sections. `src/ingestion/chunk.py`.

**Coordinates (bounding box).** The rectangle an element occupies on its page. `unstructured` reports it as four corner points plus the coordinate system (for example `PixelSpace` with the page rendered at 2125 x 2750). Needed for cropping figures and for highlighted citations.

**Enrichment (summary).** A vision model's written description of a chunk, including its tables and figures. The summary is what gets embedded; the original is what the answer is built from.

**Summary backend.** Which server writes the summaries: `huggingface` (a hosted model) or `local` (a model on this machine). Chosen with `ENHANCEMENT_BACKEND` or `scripts/ingest.py --backend`. See [OpenAI-compatible API](#api-and-serving).

**MLX and mlx-vlm.** MLX is Apple's array library for Apple silicon; it runs models on the Mac's GPU using the memory the CPU and GPU share. `mlx-vlm` runs vision language models on it, and `python -m mlx_vlm.server` serves one over the OpenAI chat protocol at `http://127.0.0.1:8080/v1`.

**Quantization (4-bit).** Storing a model's weights in fewer bits than the 16 they were trained in. A 4-bit, 4-billion-parameter model needs about 2.5 GB instead of 8 GB, at a small cost in quality; this is what lets it run on a 16 GB laptop.

**Retry with exponential backoff.** After a failed call, wait and try again, doubling the wait each time (for example 0.5 s, 1 s, 2 s), so a busy or restarting server gets time to recover. The OpenAI client does this itself for timeouts, connection errors, 429 and 5xx.

**Optional dependency (extra).** A group of packages installed only on request, declared under `[project.optional-dependencies]` in `pyproject.toml`. `uv sync --extra local` adds `mlx-vlm`; a plain `uv sync` leaves it out (and removes it if present). An environment marker (`sys_platform == 'darwin' and platform_machine == 'arm64'`) also keeps it off machines that cannot run it.

**Dependency group.** Packages needed to work on the project but not to run it, declared under `[dependency-groups]` in `pyproject.toml`. The `dev` group holds `pytest` and `httpx`; `uv sync` installs it by default.

**Lock file (`uv.lock`).** The exact version of every package, direct and indirect, that `uv` resolved from the ranges in `pyproject.toml`. `uv sync` installs exactly those versions, so every machine gets the same environment; `uv lock` updates the file after `pyproject.toml` changes.

**Repetition penalty.** A sampling setting above 1.0 that makes a token less likely once it has appeared, which stops small models looping on the same line.

## Database

**Postgres and pgvector.** Postgres is a relational database: tables of rows with typed columns, queried with SQL. pgvector is an extension that adds a `vector(n)` column type and distance operators, so the same database stores chunks and searches their embeddings. Here `<=>` (cosine distance) ranks chunks, matching the Chroma collection. `src/db/schema.py`.

**SQLAlchemy Core.** The part of SQLAlchemy that describes tables as Python `Table` objects and builds SQL from them, without the ORM's classes and sessions. Queries stay close to the SQL they produce. `src/db/schema.py`.

**Migration (Alembic).** A versioned script that changes the database schema, kept in git like code. Alembic records the applied version in an `alembic_version` table; `alembic upgrade head` runs the scripts not yet applied, `downgrade` runs their `downgrade()` to undo them, and `alembic check` compares the database with the tables in `src/db/schema.py`. `migrations/`.

**Connection pooler.** A server between clients and Postgres that lets many clients share a few database connections, because each Postgres connection costs memory. Neon's pooler (PgBouncer, the host with `-pooler`) works per transaction, so session features such as prepared statements do not carry over; migrations and bulk copies use the direct host instead. `DATABASE_URL` and `DATABASE_URL_DIRECT`.

**Engine (connection pool).** SQLAlchemy's object for one database: it opens connections when code asks for one, keeps a few open for reuse, and hands them out again, so a request does not pay for a new TLS connection each time. Created once per process (`src/db/connection.py:create_db_engine`).

**Transaction.** A group of SQL statements that succeed or fail together: `COMMIT` keeps all of them, `ROLLBACK` (or any error) keeps none. `engine.begin()` opens one and commits when its block ends. Ingestion replaces a document inside one, so a failure never leaves half a document.

**Foreign key and ON DELETE CASCADE.** A foreign key says a column must hold the id of a row in another table (`chunks.document_id` points at `documents.id`). `ON DELETE CASCADE` makes deleting that row also delete the rows that point at it, so deleting a document removes its chunks and their figure rows.

**Prepared statement.** A query the server parses once and runs many times. psycopg 3 prepares repeated queries automatically, but a transaction pooler such as PgBouncer may run the next transaction on another server connection that never saw it, so `create_db_engine` turns this off (`prepare_threshold=None`).

**Neon branch.** A copy of a Neon database that shares storage with its parent until either changes, so it is created instantly. Here `dev` is where migrations, live checks and ingestion changes are tried; `production` is what the deployed API reads.

## API and serving

**Lifespan.** The FastAPI hook that runs code once when the server starts and once when it stops. Here it builds the embedding client and the database engine a single time instead of per request. `api/main.py`.

**Router (`APIRouter`).** A FastAPI object that collects endpoints in a module that does not have the app. `@router.get(...)` only records a route; `app.include_router(router)` copies the routes into the app, adding the router's `prefix` and `tags`. Here: one router per file in `api/routers/`.

**API versioning.** Putting a version in every path (`/v1/query`) so the API can change shape later under `/v2` while existing clients keep using `/v1`. Here the prefix is added once, in `api/main.py`'s `include_router(..., prefix="/v1")`; `/` and `/health` stay unversioned so probes never need a new URL.

**Path parameter.** A `{name}` part of a route's path, such as `{position}` in `/v1/images/{record_id}/{position}`. FastAPI passes it to the function parameter with the same name and converts it with that parameter's type hint, returning 422 when the conversion fails.

**Query parameter.** A `name=value` pair after the `?` in a URL, such as `?offset=50&limit=50`. FastAPI passes it to the function parameter with the same name, converts it with its type hint, and checks its limits (`Query(ge=1, le=200)`), returning 422 when either fails.

**Pagination (offset and limit).** Returning a long list in pages. The client sends `offset` (how many items to skip) and `limit` (how many to return), and the response says `total`, so the client knows when it has everything. A page past the end is simply empty. The alternative, cursor pagination, hands back an opaque "continue from here" token instead; it suits huge or fast-changing lists, where an offset can skip or repeat items when rows are inserted. Here a document's chunks are few and change only on re-ingestion, so offset and limit are enough. `GET /v1/documents/{document_id}/chunks`.

**Record id.** The id each stored chunk gets when it is stored (the first 310 came from Chroma, new ones from ingestion), kept as `chunks.id` in Postgres, a random UUID such as `004ca22b-a528-444e-9c0e-27c8b5858ff5`. It never changes while the record exists, so figure URLs contain it (`/v1/images/{record_id}/{position}`). Not the same as `chunk_id`, which is only the chunk's position in one response.

**Path traversal.** An attack where a request smuggles `..` or an absolute path into something the server turns into a file path, to read files outside the intended folder. The image route avoids it twice: the URL names a record, not a file, and the stored path must still resolve inside `IMAGES_DIR` (`src/figures.py:figure_file`).

**Cache-Control.** A response header that tells browsers and proxies how long they may reuse a response without asking again. Figures are sent with `public, max-age=86400` (anyone may cache them for a day), which is safe because a record's figures never change. `FileResponse` also sends `ETag` and `Last-Modified`, so a client can check cheaply whether its copy is still current.

**MIME type (media type).** A standard label for a kind of content, such as `image/jpeg` or `application/json`, registered with IANA. The server sends it in the `Content-Type` header so the client knows how to read the bytes.

**OpenAPI.** A standard JSON description of an HTTP API: its paths, parameters, request and response shapes, and examples. FastAPI generates it from the routes and Pydantic models and serves it at `/openapi.json`.

**Swagger UI.** The interactive page at `/docs` that FastAPI builds from the OpenAPI schema, where each endpoint can be tried in the browser. Its Examples dropdown comes from `Body(openapi_examples=...)`; here those examples live in `api/openapi_examples.py`.

**SSE (Server-Sent Events).** A way for a server to send a stream of small messages over one HTTP response. Used to stream the answer token by token from `/v1/query/stream`.

**Event loop.** The single thread that runs all `async` endpoints in FastAPI. Blocking work on it stalls every request, which is why synchronous LangChain calls run in `asyncio.to_thread`.

**Multipart form data.** The HTTP body format for uploading files together with form fields. The hosted Parse API and the planned `POST /v1/documents` both use it.

**202 Accepted.** An HTTP status meaning "the request was accepted and will finish later". Used for long jobs such as ingestion, with a status URL to check.

**HTTP status code.** The three-digit number at the start of every HTTP response that says how it went, before anyone reads the body: 2xx succeeded, 4xx means the caller must change the request, 5xx means the server or something behind it failed. Clients, retries, proxies and uptime probes all act on it, which is why a failure must never be a 200. The codes this API uses are below; every error body is `{"detail": "..."}`.

**422 Unprocessable Content.** The request was well-formed JSON but its content is wrong. FastAPI returns it when a body fails the Pydantic model (for example `top_k` 31); here `/v1/query` also returns it when its `llm` object names a provider or model not in the catalogue. The body lists each failing field as `{type, loc, msg}`; the rejected value (`input`) is dropped, see [Request validation error](#request-validation-error).

**500 Internal Server Error.** Something failed that the code did not expect. Here the caller gets only `"Internal server error"` and the traceback goes to the log, because error text can expose file paths and library internals.

**502 Bad Gateway.** The server was acting for another server (here the model provider) and did not get a usable answer from it: the provider raised an error, or it returned no answer text. It tells the caller our service is fine but the upstream failed, so retrying later, or with another model, may work.

**503 Service Unavailable.** The server cannot handle requests right now. Here: the database could not be reached. Uptime probes such as Render's health check treat it as "not healthy".

**Exception handler.** A function registered with `@app.exception_handler(SomeError)` in `api/main.py`. When a route raises that error and nothing catches it, FastAPI calls the handler, which turns it into a response. Routes can then just raise, and the mapping from error to status code lives in one place.

**SecretStr.** A Pydantic string type for secrets. Printing, logging or dumping a model that holds one shows `**********`; only `.get_secret_value()` returns the real text. `llm.api_key` in `api/schemas.py:LLMSettings` is one, and `api/routers/query.py:query` unwraps it once, to build the chat model.

<a id="request-validation-error"></a>**Request validation error.** What FastAPI raises when a request body does not match its Pydantic model (a missing field, a number out of range). Its default handler answers 422 and, for each error, includes `input`, the value that failed. For a missing `llm.model` that value is the whole `llm` object, API key included, so `api/main.py:invalid_request` drops `input` and keeps `type`, `loc` (where in the body) and `msg`.

**finish_reason.** Why the model stopped writing, reported with the last streamed chunk: `stop` (done), `length` (OpenAI-style providers) or `MAX_TOKENS` (Gemini) when it hit `max_tokens`, or a safety reason. A reply with no text and `length` means reasoning used up the whole budget. `src/retrieval/generate.py:_empty_answer_message`.

## Docker and deployment

**Image.** A read-only snapshot of a filesystem plus a start command. `docker build` produces it from the `Dockerfile`; `docker run` starts a container from it. Here: `multimodal-rag-api`.

**Layer.** One step of the `Dockerfile` (`RUN`, `COPY`) stored as a diff on top of the previous step. Docker reuses a layer when its inputs did not change, which is why `api/requirements.txt` is copied and installed before the code: a code change does not reinstall the packages.

**Build context.** The folder `docker build` sends to the Docker engine (the `.` at the end of the command). `COPY . .` can only copy files that are in the context.

**.dockerignore.** A file listing paths to leave out of the build context. Here it is an allowlist: `*` ignores everything and the `!` lines re-include `api/`, `src/`, `config/`, and `dbv2/` (the API dependency list, `api/requirements.txt`, comes in with `api/`).

**Allowlist and denylist.** An allowlist names what is permitted and rejects the rest; a denylist names what is rejected and permits the rest. An allowlist is safer by default, because something new is left out until someone adds it.

**Container.** A running process started from an image, isolated from the host with its own filesystem, network, and process list. It is not a virtual machine: it shares the host's kernel. Anything written inside it is lost when it is removed, unless it is written to a volume.

**Dockerfile.** The recipe `docker build` follows to produce an image: one instruction per line, run top to bottom. Here: `Dockerfile` at the repository root.

**Base image (`FROM`).** The image a Dockerfile starts from. Here `python:3.13-slim`: Debian with Python 3.13 and few system packages. `slim` is smaller than the full image and, unlike `alpine`, can install ordinary prebuilt Python wheels.

**Tag.** The label after the colon in an image name (`multimodal-rag-api:latest`, `python:3.13-slim`). A tag can be moved to a newer image; a digest (`@sha256:...`) names one exact image forever.

**Registry.** A server that stores images so other machines can pull them. Docker Hub is the default one. `docker push` uploads, `docker pull` downloads, and only layers the other side lacks are transferred.

**Layer cache.** Docker reuses a layer from the previous build when the instruction and its input files are unchanged. The first changed layer and every layer after it are rebuilt, so instructions go from least often changed (system packages, dependencies) to most often changed (application code).

**`RUN`, `COPY`, `ENV`, `WORKDIR`.** Build-time instructions. `RUN` executes a command and saves the result as a layer. `COPY` brings files from the build context into the image. `ENV` sets an environment variable for later instructions and for the running container. `WORKDIR` sets the folder later instructions and the start command run in (`/app` here).

**`CMD` and `ENTRYPOINT`.** What the container runs when it starts. `CMD` is the default and is replaced by any arguments given to `docker run`; `ENTRYPOINT` is fixed and receives those arguments. Here `CMD` starts `uvicorn`.

**Exec form and shell form.** `CMD ["uvicorn", "api.main:app"]` (exec form) starts the program directly. `CMD uvicorn api.main:app` (shell form) starts `sh -c` with the program under it. Only a shell expands `${PORT}`, which is why the `CMD` here calls `sh -c` explicitly.

**PID 1.** The first process in a container. `docker stop` sends it `SIGTERM` and kills the container after 10 seconds if it has not exited. If PID 1 is a shell that does not pass the signal on, the server never gets the chance to shut down cleanly.

**`EXPOSE`.** A note in the image saying which port the server listens on. It opens nothing by itself; `docker run -p 8000:8000` (host port, then container port) does the publishing.

**`HEALTHCHECK`.** A command Docker runs inside the container on a timer; exit code 0 marks the container `healthy`. Here it requests `/health`. Hosting platforms often ignore it and use their own health check setting.

**Environment file (`--env-file`).** A file of `NAME=value` lines that `docker run` turns into environment variables in the container. Docker takes each line literally, so quotes become part of the value. It keeps secrets out of the image, which anyone who can pull the image could otherwise read.

**Volume.** A folder that lives outside the container's own filesystem and survives when the container is removed. Not used here: the data lives in Postgres on Neon, and the figure files are baked into the image.

**Multi-stage build.** A Dockerfile with several `FROM` lines. An early stage holds compilers and build tools, and the final stage copies only the finished result out of it, so the tools never reach the shipped image.

## Testing

**Fixture.** A pytest function that prepares something a test needs and cleans it up afterwards. A test asks for it by naming it as an argument. Here `backend` and `client` in `tests/api/conftest.py`.

**Monkeypatch.** pytest's built-in fixture for temporarily replacing an attribute, such as a function inside a module, for one test. It puts the original back when the test ends. Here it swaps the database reads, the embedding client and the LLM builders for fakes.

**Fake (test double).** A small stand-in object that behaves like the real dependency for the few methods the code calls. `_FakeStore` has only `as_retriever`, and `_FakeLLM` has only `stream`.

**Mock transport.** An `httpx` feature that answers HTTP requests from a Python function instead of the network. The OpenAI SDK sends its requests through `httpx`, so `tests/test_trace_structure.py` gives a real `ChatOpenAI` client a mock transport: the client, LangChain's tracing and the SDK's logs are all real, but nothing leaves the machine.

**TestClient.** FastAPI's in-process HTTP client, built on `httpx`. It sends real requests to the app without starting a server or opening a port. Used inside `with`, it also runs the app's lifespan.

**Characterization test.** A test that records what the code does now, including behaviour known to be wrong, so a refactor can prove it changed nothing. When the defect is fixed, the test is updated in the same change. The known-defect tests in the API tests (now `tests/api/`) (a 200 for a failing store, an apology for a failing provider, a 500 for an unknown model) were examples until commit 6a fixed those status codes.

**Marker.** A label on a test (`@pytest.mark.live`) that lets a run select it or leave it out with `-m`. Markers are registered in `pyproject.toml`. Here `live` tests are skipped unless the run uses `-m live`.
