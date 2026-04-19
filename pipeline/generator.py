"""
Generator — Part 3 of the pipeline.

Responsibilities:
  - Accept processed documents + a built retrieval index
  - Retrieve relevant evidence for each draft type
  - Generate structured drafts grounded in that evidence
  - Include source citations so every claim is traceable
  - Instruct the LLM not to fabricate — say "not found" instead

Four draft types are supported:
  title_review_summary   — liens, tax, ownership from title search docs
  case_status_memo       — cross-document summary with prioritized actions
  document_checklist     — what's filed, what's missing, upcoming deadlines
  action_item_extract    — prioritized task list from email + court order
"""

from __future__ import annotations

import json
import logging

import google.generativeai as genai

from pipeline.config import CASE_CONTEXT_FILE, GEMINI_API_KEY, GEMINI_MODEL
from pipeline.models import DraftOutput, ProcessedDocument, RetrievedChunk
from pipeline.retriever import DocumentIndex, format_retrieved_context

logger = logging.getLogger(__name__)

# configure once at module level
genai.configure(api_key=GEMINI_API_KEY)

# load case context once so generators don't hit disk on every call
with open(CASE_CONTEXT_FILE, "r", encoding="utf-8") as _f:
    _CASE_CONTEXT = json.load(_f)
_CASE_CONTEXT_JSON = json.dumps(_CASE_CONTEXT, indent=2)

# system-level constraint injected into every generation prompt
_GROUNDING_RULE = (
    "Use ONLY information present in the retrieved evidence. "
    "If a fact is not in the evidence, write \"not found in documents\" rather than guessing."
)


def _call_llm(prompt: str) -> str:
    """Call Gemini and return the response text. Returns an error string on failure."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        return f"[ERROR: generation failed — {exc}]"


def _citation_block(chunks: list[RetrievedChunk]) -> str:
    """Build a numbered source list for appending to draft output."""
    lines = [
        f"  [{i}] {rc.chunk.source_file} (relevance: {rc.score:.2f})"
        for i, rc in enumerate(chunks, start=1)
    ]
    return "\n".join(lines)


def _style_section(style_guide: str) -> str:
    if style_guide.strip():
        return f"\n\nSTYLE GUIDE — apply these patterns learned from operator edits:\n{style_guide}"
    return ""


def _build_draft(
    draft_type: str,
    query: str,
    instructions: str,
    index: DocumentIndex,
    top_k: int,
    style_guide: str,
) -> DraftOutput:
    """
    Common scaffold for all draft generators:
      1. Retrieve relevant chunks
      2. Build a grounded prompt
      3. Call the LLM
      4. Append the citation block
    """
    chunks = index.retrieve(query, top_k=top_k)
    context = format_retrieved_context(chunks)

    prompt = (
        f"You are a legal case management assistant.\n\n"
        f"CASE CONTEXT:\n{_CASE_CONTEXT_JSON}\n\n"
        f"RETRIEVED EVIDENCE:\n{context}\n"
        f"{_style_section(style_guide)}\n\n"
        f"GROUNDING RULE: {_GROUNDING_RULE}\n\n"
        f"INSTRUCTIONS:\n{instructions}\n"
    )

    content = _call_llm(prompt)
    content += f"\n\nSOURCES USED:\n{_citation_block(chunks)}"

    return DraftOutput(draft_type=draft_type, content=content, sources=chunks)

def generate_title_review_summary(
    index: DocumentIndex,
    docs: list[ProcessedDocument],  # noqa: ARG001 — kept for consistent signature
    style_guide: str = "",
) -> DraftOutput:
    """Generate a Title Review Summary grounded in the title search documents."""
    return _build_draft(
        draft_type="title_review_summary",
        query="liens encumbrances mortgage HOA lis pendens easement taxes ownership chain vesting instrument number",
        top_k=6,
        index=index,
        style_guide=style_guide,
        instructions=(
            "Write a Title Review Summary for the Rodriguez case.\n"
            "Organize into labeled sections: LIENS & ENCUMBRANCES, TAX STATUS, OWNERSHIP, OTHER MATTERS, REVIEWER NOTES.\n"
            "Include instrument numbers, recording dates, and dollar amounts where available.\n"
            "Flag items requiring attorney action with 'ACTION REQUIRED:'.\n"
            "Write in plain professional prose — no markdown decorations."
        ),
    )


def generate_case_status_memo(
    index: DocumentIndex,
    docs: list[ProcessedDocument],  # noqa: ARG001
    style_guide: str = "",
) -> DraftOutput:
    """Generate a Case Status Memo pulling evidence from all four documents."""
    return _build_draft(
        draft_type="case_status_memo",
        query="deadlines action items servicer transfer borrower counsel HOA complaint filing case management conference payoff",
        top_k=7,
        index=index,
        style_guide=style_guide,
        instructions=(
            "Write a Case Status Memo for the Rodriguez foreclosure case.\n"
            "Start with a header block: case number, court, judge, plaintiff, borrower, borrower's counsel.\n"
            "Then: ACTION ITEMS (numbered, labeled URGENT / HIGH / NORMAL by urgency).\n"
            "Then: UPCOMING DEADLINES (specific dates and what must be filed or done by each).\n"
            "Then: TITLE CONCERNS (from title search documents).\n"
            "Then: NOTES (servicer contact, payoff amount, other details).\n"
            "Write in plain professional prose — no markdown decorations."
        ),
    )


def generate_document_checklist(
    index: DocumentIndex,
    docs: list[ProcessedDocument],  # noqa: ARG001
    style_guide: str = "",
) -> DraftOutput:
    """Generate a Document Checklist — on file, missing, upcoming deadlines."""
    return _build_draft(
        draft_type="document_checklist",
        query="documents on file required missing deadline proof of service complaint case management report",
        top_k=6,
        index=index,
        style_guide=style_guide,
        instructions=(
            "Write a Document Checklist for the Rodriguez case.\n"
            "Three sections: DOCUMENTS ON FILE, DOCUMENTS REQUIRED / MISSING, UPCOMING FILING DEADLINES.\n"
            "For missing documents, cite which source (court order or servicer email) requires them.\n"
            "Include instrument numbers and dates where relevant.\n"
            "Write in plain professional prose."
        ),
    )


def generate_action_item_extract(
    index: DocumentIndex,
    docs: list[ProcessedDocument],  # noqa: ARG001
    style_guide: str = "",
) -> DraftOutput:
    """Generate a prioritized action item list from the servicer email and court order."""
    return _build_draft(
        draft_type="action_item_extract",
        query="action required must shall deadline billing resubmit servicer counsel HOA proof of service",
        top_k=6,
        index=index,
        style_guide=style_guide,
        instructions=(
            "Extract and prioritize all action items from the source documents.\n"
            "Label each one: URGENT (< 2 weeks or immediate risk), HIGH (within 30 days), NORMAL (no pressing deadline).\n"
            "For each item state: who is responsible, exactly what must be done, and by when.\n"
            "Write in plain professional prose."
        ),
    )


# registry so callers can look up a generator by name
GENERATORS: dict[str, object] = {
    "title_review_summary": generate_title_review_summary,
    "case_status_memo": generate_case_status_memo,
    "document_checklist": generate_document_checklist,
    "action_item_extract": generate_action_item_extract,
}


def generate_all_drafts(
    index: DocumentIndex,
    docs: list[ProcessedDocument],
    style_guide: str = "",
    draft_types: list[str] | None = None,
) -> list[DraftOutput]:
    """
    Run the specified draft generators (all four by default).
    Unknown draft type names are logged and skipped.
    """
    if draft_types is None:
        draft_types = list(GENERATORS.keys())

    outputs: list[DraftOutput] = []
    for name in draft_types:
        gen_fn = GENERATORS.get(name)
        if gen_fn is None:
            logger.warning("Unknown draft type '%s' — skipping", name)
            continue
        logger.info("Generating %s ...", name)
        outputs.append(gen_fn(index, docs, style_guide=style_guide))  # type: ignore[call-arg]

    return outputs
