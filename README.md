# AI Case Management Pipeline

A four-capability document processing pipeline built for the Rodriguez foreclosure case (2025-FC-08891). Processes messy legal documents, retrieves grounded evidence, generates structured draft outputs, and improves from operator edits.

## What this does

| Capability | Description |
|---|---|
| Document Processing | Cleans OCR noise and extracts structured data from title searches, servicer emails, and court orders |
| Grounded Retrieval | Chunks and embeds documents locally, retrieves relevant passages, and makes citations inspectable |
| Draft Generation | Produces Title Review Summary, Case Status Memo, Document Checklist, and Action Item Extract grounded in retrieved evidence |
| Improvement from Edits | Analyzes operator edit pairs, extracts writing patterns, applies them to future generations |

## Requirements

- Python 3.10 or later
- A Google Gemini API key

## Setup

```
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

## Run the pipeline

Processes all four documents, builds the index, generates drafts, learns from operator edits, and saves everything to `sample_outputs/`.

```
python run_pipeline.py
```

## Run the REST API

Starts a FastAPI server on port 8000 with a browser UI at `http://localhost:8000/ui`.

```
uvicorn api:app --reload --port 8000
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Liveness check, shows API key status and indexed doc count |
| POST | /process | Upload .txt files, runs OCR cleanup + extraction, builds retrieval index |
| POST | /retrieve | Run a query against the indexed documents, returns grounded chunks |
| POST | /generate | Generate a named draft type (optionally with style guide) |
| POST | /learn | Build a style guide from operator edit pairs |
| POST | /compare | Compare two draft versions and get an LLM evaluation |

Interactive API docs are at `http://localhost:8000/docs`.

## Run the tests

Tests mock the LLM and the embedding model so they run offline and without API keys.

```
pytest tests/ -v
```

## Docker

Build and run both the pipeline and the API in containers:

```
docker compose up
```

- `pipeline` service: runs `python run_pipeline.py` and saves outputs to `sample_outputs/`
- `api` service: runs the FastAPI server on port 8000

The `sample_outputs/` directory is mounted as a volume, so generated files are accessible on the host.

## Project structure

```
pipeline/
  __init__.py        package marker
  config.py          paths, model names, API key loading
  models.py          ProcessedDocument, Chunk, RetrievedChunk, DraftOutput
  processor.py       OCR cleanup + LLM structured extraction (Part 1)
  retriever.py       chunking, local embeddings, numpy cosine-sim index (Part 2)
  generator.py       four draft generators with retrieved evidence grounding (Part 3)
  learner.py         edit pair analysis, style guide building, improvement loop (Part 4)
api.py               FastAPI REST endpoints + UI static file serving
run_pipeline.py      CLI entry point
tests/
  test_processor.py
  test_retriever.py
  test_generator.py
  test_learner.py
ui/
  index.html         browser UI for upload / retrieval / generation / edit capture
sample_outputs/      all generated outputs from the pipeline run
Dockerfile
docker-compose.yml
EVALUATION.md        how grounding and improvement were measured
APPROACH.md          architecture overview, assumptions, tradeoffs
README.md            this file
```

## Output files

| File | Contents |
|---|---|
| `01_clean_*.txt` | OCR-cleaned version of each source document |
| `01_extraction_results.json` | Structured data extracted from all documents |
| `02_retrieval_demo.txt` | Sample retrieval query result |
| `03_baseline_title_review_summary.txt` | Generated Title Review Summary |
| `03_baseline_case_status_memo.txt` | Generated Case Status Memo |
| `04_learned_style_guide.json` | Patterns learned from operator edit pairs |
| `04_baseline_document_checklist.txt` | Document Checklist before style guide |
| `04_improved_document_checklist.txt` | Document Checklist after style guide |
| `04_improvement_evaluation.txt` | LLM comparison of baseline vs improved |

## Swapping the LLM provider

All LLM calls go through `genai.GenerativeModel` in each module. To switch to OpenAI, replace those calls with `openai.chat.completions.create` — the prompt text is the same. Update `GEMINI_MODEL` in `config.py` and add `OPENAI_API_KEY` to `.env`.

## Notes

The embedding model (`all-MiniLM-L6-v2`) downloads ~80 MB on first run and caches locally. Subsequent runs use the cache.

Adding a new document to a case: drop a `.txt` file in `ai_engineer_assignment_data/sample_documents/` and re-run. No code changes needed as long as the filename matches one of the known patterns (title_search_*, servicer_email, court_order).
