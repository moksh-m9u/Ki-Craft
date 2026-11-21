# Multimodal RAG — API Guide for Frontend Developers

This guide explains how to talk to the Multimodal RAG backend so you can build a
chat UI, a search box, or a document viewer on top of it — without knowing
anything about the ML pipeline underneath.

The backend is a **FastAPI** service. It answers questions by searching a
vector database of PDFs (datasheets like the LM317 / LM2596) and generating an
answer with a multimodal LLM. Responses include the answer text **plus** the
supporting evidence: the exact document chunks, their tables, and their figures.

- Interactive docs: `GET /docs` (Swagger UI) — try every endpoint in the browser.
- Base URL: `http://localhost:8000` locally. In production, use the deployed URL
  (e.g. `https://your-app.up.railway.app`). Everything below is relative to it.
- Versioning: every endpoint is under `/v1` except `/` and `/health`, which stay
  unversioned so uptime probes keep a fixed URL. A breaking change would go to
  `/v2` while `/v1` keeps working. The old unprefixed paths (`/query`,
  `/retrieve`, ...) were removed on 2026-11-15 and return 404.

---

## 1. The endpoints at a glance

| Endpoint | Method | Purpose | Returns |
|---|---|---|---|
| `/` | GET | Index / endpoint list | `{name, version, endpoints}` |
| `/health` | GET | Is the service up? | `{status: "ok", vector_store: "loaded"}`, or **503** when the store cannot load |
| `/v1/retrieve` | POST | Get evidence **without** an answer | `RetrieveResponse` |
| `/v1/query` | POST | Get evidence **and** a full answer | `QueryResponse` |
| `/v1/query/stream` | POST | Same body as `/v1/query`, with a streamed answer | Server-Sent Events |
| `/v1/images/{record_id}/{position}` | GET | Fetch one figure as an image (stable URL) | raw JPEG bytes |
| `/v1/providers` | GET | The providers and models the `llm` object accepts | `ProviderList` |
| `/v1/documents` | GET | List the indexed documents | `DocumentList` |
| `/v1/documents/{document_id}/chunks` | GET | One page of a document's chunks, in reading order | `ChunkPage` |
| `/docs` | GET | Swagger UI (try-it-out docs) | HTML |

There are only **three shapes** you need to understand:

1. **Request** — `{"query": "...", "top_k": 5}` (`top_k` optional). The two query endpoints also take an optional `max_images` and an optional `llm` object to pick the model.
2. **`RetrieveResponse`** — the chunks + everything they contain.
3. **`QueryResponse`** — the same, plus `answer`, `answer_characters` and `figures`.

### Errors

A failure is a real HTTP error status, never a 200 with an apology in the
answer. Every error body has the same shape: `{"detail": "..."}`.

| Status | Meaning | Example `detail` |
|---|---|---|
| `422` | The body is invalid (FastAPI's own check, `detail` is a list of `{type, loc, msg}`; the rejected value is never echoed, so an `llm.api_key` cannot come back in it), or `llm` named a provider or model that is not in the catalogue | `"Model gpt-9 not found for provider openai"` |
| `500` | Unexpected server error; the real message is only in the server log | `"Internal server error"` |
| `502` | The model provider failed (bad key, quota, timeout), or it returned no answer text | `"The model provider failed: ..."`, or `"The model used its whole output budget (512 tokens, reasoning included) before writing any answer text. Raise max_tokens and try again."` |
| `503` | The vector store could not be loaded: the embedding client failed to start (missing credentials, or `EMBEDDING_BACKEND=local` without llama-cpp-python), or the store has no collection for the configured embedding model, or its collection was made by another model | `"Vector store unavailable: dbv2/chroma_db has no collection for embedding model baai/bge-m3 (EMBEDDING_BACKEND=openrouter). Build it from the stored summaries with: python -m scripts.reembed"` |

`/health` returns 503 with its usual body (`status: "error"` plus `detail`),
so an uptime probe can rely on the status code alone.

---

## 2. The request

`/v1/retrieve`, `/v1/query`, and `/v1/query/stream` accept the fields below.
Both query endpoints also accept an optional `llm` object.

```json
{
  "query": "Tell me about the pin configuration of the LM317.",
  "top_k": 5,
  "max_images": 3
}
```

| Field | Type | Rules | Meaning |
|---|---|---|---|
| `query` | string | 1–2000 chars | The question you want answered |
| `top_k` | integer | 1–30, default 5 | How many document chunks to fetch. Every fetched chunk's text and tables go to the model, so more chunks = more evidence, but a longer prompt. |
| `max_images` | integer | 0–50, default 3 | Query endpoints only. Most figures sent to the model, taken in chunk order. 0 sends every image. Images cost far more tokens than text. A text-only model (`supports_vision` false in `config/providers.py`) is sent none, whatever this says. |

Start with the defaults; raise `top_k` if answers miss context, and raise
`max_images` (or set 0) for questions about graphs and diagrams.
`/v1/retrieve` never calls a model, so it takes only `query` and `top_k`.

### Choosing the model (`llm`)

Leave `llm` out and the server's default model (Gemini 3.5 Flash-Lite) answers
with the server's own key. Send it to answer with another catalogued model and
**your own** key:

```json
{
  "query": "Tell me about the pin configuration of the LM2596.",
  "top_k": 5,
  "max_images": 3,
  "llm": {
    "provider": "groq",
    "model": "openai/gpt-oss-120b",
    "api_key": "YOUR_GROQ_API_KEY",
    "temperature": 0.0,
    "max_tokens": 2048
  }
}
```

| Field | Type | Rules | Meaning |
|---|---|---|---|
| `provider` | string | required | `gemini`, `groq`, `huggingface` or `openai` |
| `model` | string | required | A model id listed for that provider in `config/providers.py`; anything else is a 422 |
| `api_key` | string | required, non-empty | Your key for that provider. Used for this request only: never stored, logged, traced or echoed back. There is no fallback to a server key |
| `temperature` | number | 0–2, default 0 | Not sent to models that reject or ignore it (Gemini 3.5 Flash-Lite and 3.6 Flash, OpenAI o-series and GPT-5.4) |
| `max_tokens` | integer | 1–8192, optional | Output budget, reasoning included. Left out, the catalogue default applies: 2048 for models that reason by default, 512 otherwise |

`GET /v1/providers` returns this catalogue, so a model picker never offers a
model the server would reject:

```json
{
  "items": [
    {
      "id": "gemini",
      "display_name": "Google Gemini",
      "models": [
        {
          "id": "gemini-3.5-flash-lite",
          "display_name": "Gemini 3.5 Flash-Lite",
          "supports_vision": true,
          "accepts_temperature": false,
          "default_max_tokens": 512,
          "input_price_per_1m": 0.3,
          "output_price_per_1m": 2.5
        }
      ]
    }
  ],
  "total": 4
}
```

`/docs` has request examples for Gemini, Groq GPT-OSS, OpenAI GPT-4o mini and
the Hugging Face router on both query endpoints.

---

## 3. The response — what comes back

This is what `POST /v1/query` returns (identical top-level shape for `/v1/retrieve`,
minus `answer`, `answer_characters` and `figures`):

```jsonc
{
  "query": "Tell me about the pin configuration of the LM317.",
  "num_chunks": 2,                 // how many chunks were retrieved
  "total_images": 3,               // sum of image_count across chunks
  "total_tables": 2,               // sum of tables across chunks
  "answer": "The LM317 is an adjustable linear voltage regulator...",  // /v1/query only
  "answer_characters": 1086,       // /v1/query only
  "figures": [                     // /v1/query only: the figures the model saw
    {"label": "IMG1", "url": "/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/0",
     "chunk_id": 1, "source": "lm317"}
  ],
  "citations": {                   // /v1/query only: what the answer cites, checked
    "chunks": [1, 2], "figures": ["IMG1"], "unknown": []
  },
  "chunks": [
    {
      "chunk_id": 1,               // 1-based position within this response
      "record_id": "004ca22b-a528-444e-9c0e-27c8b5858ff5", // stable id in the store
      "enhanced_content": "**SEARCHABLE DESCRIPTION: LM317 Voltage Regulator...", // markdown
      "raw_text": "Pin 1 (ADJ): adjustment pin...", // plain OCR text
      "tables_html": ["<table><tr><th>Pin</th><th>Name</th>...</table>"],
      "image_count": 2,            // = len(image_urls)
      "image_urls": ["/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/0",
                     "/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/1"],
      "has_table": true,
      "has_image": true,
      "source": "lm317",
      "metadata": { }
    }
  ]
}
```

### What each chunk field means for your UI

| Field | Type | How to render it |
|---|---|---|
| `enhanced_content` | string | Markdown text. Render with a markdown parser (it uses `**bold**`, headings, lists). This is the search-friendly summary. |
| `raw_text` | string | Plain OCR text from the PDF. Use for a "raw source" toggle. |
| `tables_html` | string[] | Each entry is a full `<table>...</table>` HTML string. Inject as HTML (sanitise first) or parse into your own table component. |
| `record_id` | string \| null | The chunk's id in the vector store. Stable, and part of its image URLs. |
| `image_urls` | string[] | Relative HTTP path of each figure. **Prefix with the base URL** (see §5). |
| `has_table` / `has_image` | boolean | Quick flags to skip rendering empty sections. |
| `source` | string \| null | Source filename, if the chunk knows it. |

### The figures the model saw

`chunks[].image_urls` lists every figure of every retrieved chunk. `figures`
lists only the ones sent to the model, in prompt order, capped by `max_images`
(empty for a text-only model). The prompt labels each one, and the answer
cites it in square brackets, so `[IMG1]` in `answer` means the entry whose
`label` is `IMG1`. `chunk_id` points to its chunk in `chunks`, and `url` is the
same URL that chunk lists in `image_urls`.

The answer also cites chunks: `[C2]` means the chunk whose `chunk_id` is 2.
The image labels are the server's own numbering, not the datasheet's: a
datasheet's "Figure 3" is not `IMG3`. When the answer names a datasheet
figure it adds the image label only when that image is the figure, as in
"Figure 3 [IMG1]".

### What the answer cites (`citations`)

The model is asked to cite, but nothing forces it to cite correctly. After the
answer is written the server reads it for `[C1]` and `[IMG1]` labels (several
may share one bracket, as `[C1, IMG2]`) and checks each against what the
prompt held:

| Field | Type | Meaning |
|---|---|---|
| `chunks` | integer[] | `chunk_id` of each cited chunk, in the order the answer first cites it |
| `figures` | string[] | `label` of each cited entry in `figures`, in first-cited order |
| `unknown` | string[] | Cited labels that match nothing sent, such as `IMG4` when three figures were sent; show a warning |

Show the cited figures first: order `figures` by their position in
`citations.figures`, then the rest. Other bracketed text (`[1]`, a Markdown
link) is not a citation and is ignored.

| Field | Type | Meaning |
|---|---|---|
| `label` | string | `IMG1`, `IMG2`, ... as the prompt names it |
| `url` | string | Relative URL of the figure; prefix with the base URL |
| `chunk_id` | integer | The `chunk_id` of its chunk in this response |
| `source` | string | Source document of the figure |

### Getting the figures

Responses carry no image bytes. Each figure is an HTTP URL in `image_urls`,
e.g. `/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/1`. Fetch it with
`GET {base}` plus that path and you get **raw JPEG bytes**
(`Content-Type: image/jpeg`). Use it directly in `<img src>`:

```js
`<img src="${base}${chunk.image_urls[i]}">`
```

The URL is stable and cacheable for a day (`Cache-Control: public,
max-age=86400`), so the JSON stays small and the browser fetches each figure
once.

---

### Browsing the stored documents

Two read-only endpoints show what is indexed, with no search and no model call.

`GET /v1/documents` lists every document:

```json
{
  "items": [
    {"id": "lm317", "chunk_count": 51, "image_count": 49, "table_count": 20}
  ],
  "total": 6
}
```

`GET /v1/documents/{document_id}/chunks?offset=0&limit=50` returns a document's
chunks in reading order, one page at a time:

```json
{
  "items": [
    {
      "record_id": "004ca22b-a528-444e-9c0e-27c8b5858ff5",
      "chunk_id": 1,
      "source": "lm317",
      "enhanced_content": "The LM317 is an adjustable ...",
      "raw_text": "LM317 3-Terminal Adjustable Regulator ...",
      "tables_html": [],
      "image_urls": ["/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/0"],
      "has_table": false,
      "has_image": true
    }
  ],
  "total": 51,
  "offset": 0,
  "limit": 50
}
```

| Parameter | Rules | Meaning |
|---|---|---|
| `document_id` | path | A document's `id`, as `/v1/documents` lists it; an unknown one is a **404** |
| `offset` | integer, 0 or more, default 0 | Chunks to skip from the start of the document. Past the end gives an empty `items`, not an error |
| `limit` | integer, 1 to 200, default 50 | Most chunks to return; anything else is a **422** |

Here `chunk_id` is the chunk's position in its document, counting from 1. In
query responses `chunk_id` is the chunk's position in that response instead.
To read a whole document, request pages until you have `total` chunks:

```js
let chunks = [];
do {
  const r = await fetch(`${base}/v1/documents/lm317/chunks?offset=${chunks.length}&limit=200`);
  const page = await r.json();
  chunks = chunks.concat(page.items);
  if (page.items.length === 0) break;
} while (chunks.length < page.total);
```

---

## 4. Streaming the answer (`/v1/query/stream`)

`POST /v1/query/stream` takes the same JSON body as `POST /v1/query`. The
response uses **[Server-Sent Events](GLOSSARY.md#api-and-serving)**
(`Content-Type: text/event-stream`). The chunks are sent as soon as they are
found, before the model is called, so you can show the evidence while the
model is still working (that can take tens of seconds).

| Event | `data` payload | When |
|---|---|---|
| `retrieval` | a full `RetrieveResponse` (the chunks) | first, as soon as retrieval finishes |
| `token` | `{"delta": "The "}` | many times, one fragment of the answer each |
| `done` | `{"answer": "...", "answer_characters": 1086, "num_chunks": 2, "figures": [...], "citations": {...}, "usage": {...}}` | after the last token; `figures` and `citations` as in `/v1/query` |
| `error` | `{"status": 502, "message": "..."}` | replaces `done` if the provider fails, before or after the first `token`. Unexpected server errors carry `status: 500` |

```bash
curl -N http://localhost:8000/v1/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"query": "Tell me about the pin configuration of the LM317.", "top_k": 5}'
```

In a browser, use `fetch` with `method: "POST"`. Check the HTTP status before
reading `response.body`. Decode the body as text and buffer partial frames
until the blank line that ends each event. Append each `token` event's
`delta` to the displayed answer and use `done.answer` as the final text.

An invalid `llm` returns HTTP `422` and an unavailable store HTTP `503`,
before the stream opens. Every model failure comes after the `retrieval`
event, when the HTTP status is already `200`, so it arrives as an `error`
event carrying the failure status (`502` for the provider). If you need a
provider failure as a real HTTP `502`, use `POST /v1/query`.

---

## 5. Working example (plain JS)

```js
const BASE = "http://localhost:8000"; // or your deployed URL

async function ask(query, top_k = 10) {
  const res = await fetch(`${BASE}/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function absolutize(url) {
  return url.startsWith("/") ? `${BASE}${url}` : url;
}

const data = await ask("Tell me about the pin configuration of the LM317.");
console.log(data.answer);                       // the generated answer
for (const chunk of data.chunks) {
  console.log(chunk.enhanced_content);          // markdown evidence
  console.log(chunk.tables_html);               // HTML table strings
  chunk.image_urls.forEach((u) => console.log(absolutize(u))); // absolute <img> URLs
}
```

**Always call `absolutize` on `image_urls`** — they are returned as relative
paths (`/v1/images/...`) and must be prefixed with the base URL before a browser
can load them.

---

## 6. Dependencies & gotchas (read this)

- **Image URLs are stable.** Every chunk's `image_urls` is
  `/v1/images/{record_id}/{position}`: the chunk's id in the vector store (also
  returned as the chunk's `record_id`) and the figure's 0-based position
  **within that chunk**. The same figure has the same URL in every response,
  after a restart and on any server, so you can store it or cache it. Take each
  chunk's `image_urls` as-is rather than building URLs yourself.

- **The URL names a record, not a file.** The server reads the file path from
  the store; a `record_id` that is not a UUID or a negative `position` is a
  `422`, and an unknown record or position is a `404`.

- **Handle image load errors.** Show a broken-image fallback on a `404`.

- **`raw_text` and `enhanced_content` are different.** `enhanced_content` is
  the AI-summarised, markdown version (nice to read). `raw_text` is the raw OCR
  (exact source wording, uglier). Prefer `enhanced_content` for display.

- **`tables_html` is raw HTML.** The strings come straight from the PDF
  parser. If you inject them with `innerHTML`, sanitise them first.

- **Streaming and the chunks arrive separately.** In `/v1/query/stream`, the
  `retrieval` event contains the chunks; the answer tokens follow. If you want
  both in one place, combine the `retrieval` event's data with the `done`
  event's `answer`.

---

## 7. Prerequisites before you point a browser at this

- **CORS:** the API **allows browser access** from a configured allow-list of
  origins (currently `http://localhost:5173`, `http://localhost:3000`,
  `http://127.0.0.1:5173` — your local Vite dev server is covered). For a
  production frontend URL, the backend must add it to
  `CORS_ALLOWED_ORIGINS` (a comma-separated env var), e.g.
  `CORS_ALLOWED_ORIGINS=https://app.example.com,http://localhost:5173`.
  Non-browser clients are unaffected.

- **Health check:** always `GET /health` before hammering the API — the vector
  store loads lazily, so the very first request can take a few seconds.

- **Non-browser clients** (curl, Node, mobile apps) are unaffected by CORS.

---

## 8. Choosing an endpoint

| You want… | Use |
|---|---|
| A ready-to-display answer + evidence | `POST /v1/query` |
| Evidence only (build your own answer elsewhere) | `POST /v1/retrieve` |
| A typewriter/streaming answer effect | `POST /v1/query/stream` |
| The figures of a response | `image_urls` → `GET /v1/images/...` |
| The figures an answer cites (`[IMG1]`) | `figures` → match `label` |
| The chunks an answer cites (`[C2]`) | `chunks` → match `chunk_id` |
| The providers and models for a model picker | `GET /v1/providers` |
| What is indexed, and every chunk of one document | `GET /v1/documents` → `GET /v1/documents/{id}/chunks` |
| Just a health/probe check | `GET /health` |
