# KI-CRAFT

**An open-source, AI-assisted PCB design and EDA toolkit, built to work alongside KiCad.**

KI-CRAFT aims to bring the same help to open tools: read the datasheets for you, check your schematic against them, and, step by step, draft schematics and assist with layout.

It starts where every design starts: the datasheet. Today it answers questions about datasheets and application notes from their text, tables and figures, and shows the figure each answer relied on.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F61?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![watsonx](https://img.shields.io/badge/watsonx_Granite-0F62FE?style=for-the-badge&logo=ibm&logoColor=white)](https://www.ibm.com/watsonx)
[![OpenRouter](https://img.shields.io/badge/OpenRouter_embeddings-6566F1?style=for-the-badge)](https://openrouter.ai/)
[![Gemini](https://img.shields.io/badge/Gemini_3.5_Flash--Lite-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

| Chunk Explorer | Dataset Analytics | Query & Retrieve |
|---|---|---|
| ![chunk-explorer](assets/dashboard-chunk-explorer.png) | ![analytics](assets/dashboard-analytics.png) | ![query](assets/dashboard-query.png) |

## Who it is for

- **KiCad users tired of drawing schematics by hand** and cross-checking every pin, capacitor value and footnote against a 60-page PDF.
- **Hardware engineers** checking a design: dropout at a given load, the output voltage formula, the recommended layout, with the source figure to verify against.
- **Students and makers** learning to read datasheets: what a pin does, "typical" versus "maximum", what a graph shows.
- **Teams who want an open alternative** to closed AI EDA tools, one they can run, inspect and extend.

## Status

| Phase | Goal | Progress |
|---|---|---|
| 1. Datasheet assistant | Trustworthy answers from datasheets: text, tables, figures, cited | `████░░░░░░` 40%, in progress |
| 2. Schematic review | Check a KiCad schematic against the datasheets of its parts: pins, feedback dividers, decoupling | `░░░░░░░░░░` planned |
| 3. Design agent | Draft schematics in KiCad from requirements, and assist with placement and layout | `░░░░░░░░░░` planned |

Phase 1 so far: a cleaned-up, explainable codebase, an API with streaming answers and labelled figure citations, and a pipeline that ingests any PDF. Next: an evaluation set that measures accuracy, Postgres with hybrid search, uploading your own PDFs through the API with page-level citations, and a hosted demo.

## What it does today

- **Answers from text, tables and figures.** The model gets each relevant section's original text, its tables and its figures, not just the paragraphs.
- **Shows its evidence.** Every answer lists the figures the model saw, and cites them as `[IMG1]`, `[IMG2]`, next to the chunks it used (`[C1]`).
- **Refuses to guess.** It says when the documents do not state something, keeps units and test conditions, and never mixes values from different parts.
- **Brings your own model.** Gemini by default; any model from Gemini, OpenAI, Groq or Hugging Face with your own key.
- **Ingests new PDFs** with a hosted vision model or one that runs locally on a Mac.
- **Runs without a credit card.** Search uses a free OpenRouter embedding model, IBM watsonx, or a model on your own machine; each model keeps its own index.

Six datasheets and application notes ship ready to query: LM317, LM2596, LM555, NE555, TLV1117 and Microchip AN699.

<details>
<summary><b>How it works</b></summary>

A vision model writes a plain-text summary of each section of a PDF, covering its prose, tables and figures. The summaries are embedded and searched; the original text, tables and figures are stored alongside, so each answer is written from the source material, not from the summary.

```mermaid
flowchart TB
    PDF[("PDF")]

    subgraph Ingestion["Ingestion, offline (src/ingestion)"]
        direction TB
        PART["extract.py<br/>unstructured partition_pdf, hi_res"]
        CHUNK["chunk.py<br/>chunk_by_title, max 3000 chars"]
        ENRICH["enrich.py<br/>vision LLM summary per chunk<br/>(GLM-4.5V hosted, or Qwen3.5-4B local)"]
        BUILD["pipeline.py (scripts/ingest.py)<br/>embed each summary (OpenRouter, watsonx<br/>or local Granite), write figure files"]
        PART --> CHUNK --> ENRICH
    end

    subgraph Storage["Storage (dbv2, json)"]
        CHROMA[("Chroma<br/>summary vector + raw_text,<br/>tables_html, image_paths")]
        IMAGES[("dbv2/images<br/>figure JPEGs")]
        JSON[("json/*.json<br/>chunk exports")]
    end

    subgraph Query["Query, online (api, src/retrieval)"]
        direction TB
        Q["question"]
        EMBED_Q["src/embed.py<br/>embed the question"]
        MMR["search.py<br/>MMR, keep 5 of 20"]
        REBUILD["src/documents.py<br/>rebuild text, tables, figures"]
        GEN["prompt.py, generate.py<br/>multimodal prompt, Gemini 3.5 Flash-Lite"]
        ANSWER["answer"]
        Q --> EMBED_Q --> MMR --> REBUILD --> GEN --> ANSWER
    end

    subgraph UI["Streamlit (app)"]
        CHAT["Query and Retrieve"]
        EXPLORE["Chunk Explorer, Compare, Analytics"]
    end

    PDF --> PART
    ENRICH --> JSON
    JSON --> BUILD
    BUILD --> CHROMA
    BUILD --> IMAGES
    CHROMA --> MMR
    IMAGES --> REBUILD
    CHAT -- "HTTP" --> Q
    EXPLORE -- "HTTP, /v1/documents" --> CHROMA
```

</details>

## Getting started

You need [uv](https://docs.astral.sh/uv/), a [Gemini key](https://aistudio.google.com/apikey) for answers, and one way to embed questions (below). The six documents are already indexed in the repository.

```bash
git clone https://github.com/moksh-m9u/Ki-Craft.git && cd Ki-Craft
uv sync && source .venv/bin/activate
cp .env.example .env              # fill in GEMINI_API_KEY and OPENROUTER_API_KEY
python -m pytest -q -p no:cacheprovider
uvicorn api.main:app --port 8001
```

Ask a question from a second terminal, or open the dashboard with `API_BASE_URL=http://127.0.0.1:8001 streamlit run app/dashboard.py`:

```bash
curl -X POST http://127.0.0.1:8001/v1/query -H 'Content-Type: application/json' -d '{"query": "What is the output voltage range of the LM317?"}'
```

### Choose how to embed

Set `EMBEDDING_BACKEND` in `.env`:

| Backend | Pick it when | You need |
|---|---|---|
| `openrouter` (the template's choice) | you want to start in minutes, with no card | a free key from [openrouter.ai](https://openrouter.ai/settings/keys), with free endpoints allowed in its privacy settings; about 45 questions a day on the free tier |
| `watsonx` (used when unset) | you have an IBM Cloud account | `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL` |
| `local` | documents must stay on your machine | `uv sync --extra local-embeddings`, a 303 MB model file and about 630 MB of RAM |

The API backends send your questions to the provider. Each embedding model has its own index; `python -m scripts.reembed --status` shows which one you are using. To use another OpenRouter model, build its index from the stored summaries (5 requests):

```bash
EMBEDDING_BACKEND=openrouter OPENROUTER_EMBEDDING_MODEL=baai/bge-m3 python -m scripts.reembed
```

### Docker

The image embeds through OpenRouter or watsonx only: 1.17 GB, and about 230 MB of RAM in use.

```bash
docker build -t ki-craft-api . && docker run --rm -p 8000:8000 --env-file .env ki-craft-api
```

Ingesting your own PDFs, every setting and troubleshooting are in [CONTRIBUTING.md](CONTRIBUTING.md); why each part is built the way it is, with the measurements, is in [docs/TRADEOFFS.md](docs/TRADEOFFS.md).

## Documentation

| Document | What it holds |
|---|---|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Setup, running each part, ingestion, Docker, configuration, and conventions for changes |
| [docs/TRADEOFFS.md](docs/TRADEOFFS.md) | Why each component is built the way it is, and what was considered instead |
| [docs/PRODUCT_RESEARCH.md](docs/PRODUCT_RESEARCH.md) | What people designing with datasheets need, existing tools, and where this project can stand out |

## Contributing

Issues and pull requests are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), or open a [GitHub issue](https://github.com/moksh-m9u/Multimodal-Rag/issues) to discuss an idea.

## License

[Apache 2.0](LICENSE)
