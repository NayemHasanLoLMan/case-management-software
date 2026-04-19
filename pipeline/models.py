"""
Shared data types for the pipeline.
Keeping this simple — plain dataclasses, no ORM or fancy serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessedDocument:
    """Result of running a raw source document through the processor."""

    source_file: str           # original filename, e.g. "title_search_page1.txt"
    doc_type: str              # "title_search", "servicer_email", "court_order"
    raw_text: str              # original text as-is from disk
    clean_text: str            # after OCR noise removal
    extracted: dict[str, Any]  # structured fields pulled out by the LLM


@dataclass
class Chunk:
    """A passage from a processed document, ready to be embedded and retrieved."""

    chunk_id: str              # unique id, e.g. "title_search_page1_0"
    source_file: str
    doc_type: str
    text: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A chunk that came back from a retrieval query, with its similarity score."""

    chunk: Chunk
    score: float               # cosine similarity, 0-1


@dataclass
class DraftOutput:
    """A generated draft — the text plus the evidence that backs it up."""

    draft_type: str                 # "title_review_summary", "case_status_memo", etc.
    content: str                    # the actual draft text
    sources: list[RetrievedChunk]   # chunks used as evidence
