"""
REST API — optional FastAPI layer over the pipeline.

Endpoints:
  POST /process            — process one or more uploaded documents
  POST /retrieve           — run a retrieval query against the indexed docs
  POST /generate           — generate a named draft type
  POST /learn              — build a style guide from edit pairs in the request body
  GET  /health             — liveness check

The index is held in application state so it persists for the lifetime of the
server process. Re-POSTing to /process rebuilds the index from scratch.

Run with:
  uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.config import GEMINI_API_KEY, OUTPUT_DIR
from pipeline.generator import GENERATORS, generate_all_drafts
from pipeline.learner import (
    analyze_edit_pair,
    build_style_guide,
    compare_drafts,
    format_style_guide_for_prompt,
    load_edit_pairs,
)
from pipeline.models import ProcessedDocument
from pipeline.processor import _clean_ocr_noise, _detect_doc_type, _extract_with_llm
from pipeline.retriever import DocumentIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Case Management Pipeline",
    description="Document processing, retrieval, and draft generation for legal cases.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve the UI from the ui/ directory
_UI_DIR = Path(__file__).parent / "ui"
if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

# in-process state — index + docs rebuilt by /process
_state: dict = {
    "docs": [],
    "index": None,
    "style_guide": {},
}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_type: Optional[str] = None


class GenerateRequest(BaseModel):
    draft_type: str
    use_style_guide: bool = True


class LearnRequest(BaseModel):
    # each item must have system_draft, operator_edited_version, draft_type, key_edits
    edit_pairs: list[dict]


class CompareRequest(BaseModel):
    baseline: str
    improved: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {
        "status": "ok",
        "api_key_set": bool(GEMINI_API_KEY),
        "docs_indexed": len(_state["docs"]),
    }


@app.post("/process")
async def process_documents(files: list[UploadFile] = File(...)):
    """
    Upload one or more .txt documents, process them (OCR cleanup + extraction),
    and rebuild the retrieval index. Returns a summary of extracted fields.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    docs: list[ProcessedDocument] = []
    for upload in files:
        raw = (await upload.read()).decode("utf-8", errors="replace")
        filename = upload.filename or "unknown.txt"
        stem = Path(filename).stem
        doc_type = _detect_doc_type(stem)
        clean = _clean_ocr_noise(raw)
        extracted = _extract_with_llm(doc_type, clean)
        docs.append(
            ProcessedDocument(
                source_file=filename,
                doc_type=doc_type,
                raw_text=raw,
                clean_text=clean,
                extracted=extracted,
            )
        )

    # rebuild index
    index = DocumentIndex()
    index.index(docs)
    _state["docs"] = docs
    _state["index"] = index

    return {
        "processed": len(docs),
        "documents": [
            {
                "source_file": d.source_file,
                "doc_type": d.doc_type,
                "chunks": index.chunk_count,
                "extracted_keys": list(d.extracted.keys()) if d.extracted else [],
            }
            for d in docs
        ],
    }


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """
    Run a retrieval query against the indexed documents.
    Returns the top matching chunks with source attribution.
    """
    index: Optional[DocumentIndex] = _state["index"]
    if index is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet. POST to /process first.")

    if req.doc_type:
        results = index.retrieve_for_doc_type(req.query, req.doc_type, top_k=req.top_k)
    else:
        results = index.retrieve(req.query, top_k=req.top_k)

    return {
        "query": req.query,
        "results": [
            {
                "chunk_id": r.chunk.chunk_id,
                "source_file": r.chunk.source_file,
                "doc_type": r.chunk.doc_type,
                "score": round(r.score, 4),
                "text": r.chunk.text,
            }
            for r in results
        ],
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    """
    Generate a draft of the specified type, grounded in the indexed documents.

    Available types: title_review_summary, case_status_memo,
                     document_checklist, action_item_extract
    """
    index: Optional[DocumentIndex] = _state["index"]
    docs: list[ProcessedDocument] = _state["docs"]

    if index is None:
        raise HTTPException(status_code=400, detail="No documents indexed yet. POST to /process first.")

    if req.draft_type not in GENERATORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown draft_type '{req.draft_type}'. "
                   f"Valid options: {list(GENERATORS.keys())}",
        )

    style_guide_text = ""
    if req.use_style_guide and _state["style_guide"]:
        style_guide_text = format_style_guide_for_prompt(
            _state["style_guide"],
            draft_type=req.draft_type,
        )

    gen_fn = GENERATORS[req.draft_type]
    draft = gen_fn(index, docs, style_guide=style_guide_text)  # type: ignore[call-arg]

    return {
        "draft_type": draft.draft_type,
        "content": draft.content,
        "sources": [
            {
                "source_file": rc.chunk.source_file,
                "score": round(rc.score, 4),
                "text_preview": rc.chunk.text[:120] + "...",
            }
            for rc in draft.sources
        ],
    }


@app.post("/learn")
def learn(req: LearnRequest):
    """
    Analyze operator edit pairs and update the in-memory style guide.
    The guide is also saved to sample_outputs/learned_style_guide.json.
    """
    if not req.edit_pairs:
        raise HTTPException(status_code=400, detail="edit_pairs list is empty")

    style_guide = build_style_guide(req.edit_pairs)
    _state["style_guide"] = style_guide

    pattern_counts = {
        dt: len(patterns)
        for dt, patterns in style_guide.get("patterns_by_draft_type", {}).items()
    }

    return {
        "message": "Style guide updated",
        "patterns_extracted": pattern_counts,
        "style_guide_path": str(OUTPUT_DIR / "learned_style_guide.json"),
    }


@app.post("/compare")
def compare(req: CompareRequest):
    """Ask the LLM to evaluate the improvement between two draft versions."""
    evaluation = compare_drafts(req.baseline, req.improved)
    return {"evaluation": evaluation}


@app.get("/")
def root():
    """Redirect root to the UI if it exists, otherwise show API info."""
    if _UI_DIR.is_dir():
        return FileResponse(str(_UI_DIR / "index.html"))
    return {
        "message": "AI Case Management API",
        "docs": "/docs",
        "health": "/health",
    }
