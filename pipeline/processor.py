"""
Document processor — Part 1 of the pipeline.

Responsibilities:
  - Load raw document text from disk
  - Clean obvious OCR noise (character substitutions from scanning)
  - Use the LLM to extract structured fields from each document type
  - Return ProcessedDocument objects for the rest of the pipeline

OCR noise notes for this dataset:
  - '1' (digit one) is substituted for 'l' (lowercase L) in words
  - 'O' (capital O) is substituted for '0' (zero) in numbers
  - 'fi' ligature sometimes breaks into separate characters
  The regex approach handles the most common patterns. The LLM extraction
  handles any ambiguity the regex misses.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import google.generativeai as genai

from pipeline.config import GEMINI_API_KEY, GEMINI_MODEL, DOCS_DIR
from pipeline.models import ProcessedDocument

logger = logging.getLogger(__name__)

# configure genai once at module load so we don't repeat it on every call
genai.configure(api_key=GEMINI_API_KEY)

# maps filename stem to a document type label used throughout the system
_DOC_TYPE_MAP: dict[str, str] = {
    "title_search_page1": "title_search",
    "title_search_page2": "title_search",
    "servicer_email": "servicer_email",
    "court_order": "court_order",
}

# extraction schemas per document type — returned as JSON by the LLM
_EXTRACTION_SCHEMAS: dict[str, str] = {
    "title_search": """
{
  "property_address": "string",
  "effective_date": "string",
  "file_number": "string",
  "liens": [
    {
      "type": "string",
      "party": "string",
      "amount": "string or null",
      "date": "string or null",
      "instrument_number": "string or null",
      "notes": "string or null"
    }
  ],
  "tax_status": {
    "year_2024": "string",
    "year_2025": "string",
    "parcel_number": "string or null",
    "special_assessment": "string or null"
  },
  "ownership_chain": ["list of string descriptions in chronological order"],
  "current_vesting": "string",
  "judgment_search": {
    "judgments_found": false,
    "federal_tax_liens": false,
    "state_tax_liens": false,
    "notes": "string or null"
  }
}""",
    "servicer_email": """
{
  "from": "string",
  "date": "string",
  "subject": "string",
  "borrower_name": "string",
  "loan_number": "string",
  "action_items": [
    {
      "priority": "URGENT | HIGH | NORMAL",
      "description": "string",
      "deadline": "string or null"
    }
  ],
  "servicer_transfer": {
    "from_servicer": "string",
    "to_servicer": "string",
    "effective_date": "string",
    "new_address": "string",
    "new_phone": "string"
  },
  "borrower_counsel": {
    "name": "string",
    "firm": "string",
    "phone": "string",
    "email": "string"
  },
  "payoff_amount": "string",
  "payoff_as_of": "string",
  "hoa_note": "string or null"
}""",
    "court_order": """
{
  "court": "string",
  "case_number": "string",
  "plaintiff": "string",
  "defendants": ["list of strings"],
  "order_type": "string",
  "judge": "string",
  "order_date": "string",
  "deadlines": [
    {
      "description": "string",
      "date": "string",
      "consequences": "string or null"
    }
  ],
  "conference": {
    "date": "string",
    "time": "string",
    "location": "string",
    "courtroom": "string"
  },
  "filing_requirements": ["list of strings"]
}""",
}


def _detect_doc_type(stem: str) -> str:
    """Return a document type label for the given filename stem."""
    return _DOC_TYPE_MAP.get(stem, "unknown")


def _clean_ocr_noise(text: str) -> str:
    """
    Fix common OCR substitutions found in the title search scans.

    Only applies transformations where the substitution is unambiguous
    based on surrounding characters. Anything uncertain is left for
    the downstream LLM extraction to interpret.
    """
    cleaned = text

    # digit-one replacing lowercase-L
    # covers: mid-word (tit1e->title), word-start (1ien->lien), word-end (origina1->original)
    cleaned = re.sub(r'(?<=[a-zA-Z])1(?=[a-zA-Z])', 'l', cleaned)
    cleaned = re.sub(r'\b1(?=[a-z]{2,})', 'l', cleaned)
    cleaned = re.sub(r'(?<=[a-zA-Z])1\b', 'l', cleaned)  # trailing: "origina1" -> "original"

    # capital-O replacing zero in amounts and instrument numbers
    # e.g. "$445,OOO.OO" -> "$445,000.00", "2O21-O123456" -> "2021-0123456"
    # replace O in dollar amounts first (handles OOO sequences)
    cleaned = re.sub(r'\$([0-9O,\.]+)', lambda m: '$' + m.group(1).replace('O', '0'), cleaned)
    cleaned = re.sub(r'(?<=\d)O(?=\d)', '0', cleaned)
    cleaned = re.sub(r'(?<=\d)O(?=[,.])', '0', cleaned)
    cleaned = re.sub(r'(?<=[,.])O(?=\d)', '0', cleaned)
    cleaned = re.sub(r'(?<=-)O(?=\d)', '0', cleaned)

    # fi ligature sometimes surfaces as "fi1e" after combined OCR errors
    cleaned = cleaned.replace('fi1e', 'file')

    return cleaned


def _build_extraction_prompt(doc_type: str, clean_text: str) -> str:
    """Build a JSON extraction prompt for the given document type."""
    schema = _EXTRACTION_SCHEMAS.get(doc_type, '{"notes": "string"}')
    instructions = (
        f"Extract the following fields as JSON from this document.\n"
        f"Return ONLY valid JSON with no explanation or code fences.\n"
        f"Use null for fields not present in the document.\n\n"
        f"Schema:\n{schema}"
    )

    return (
        "You are a document processing assistant for a legal case management system.\n\n"
        f"{instructions}\n\n"
        f"Document text:\n---\n{clean_text}\n---\n"
    )


def _extract_with_llm(doc_type: str, clean_text: str) -> dict:
    """
    Call Gemini to extract structured data. Returns an empty dict on failure
    so downstream steps degrade gracefully instead of crashing.
    """
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = _build_extraction_prompt(doc_type, clean_text)

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # strip markdown code fences if the model wrapped the JSON
        raw = re.sub(r'^```[a-z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw)

        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed for %s: %s", doc_type, exc)
        return {}
    except Exception as exc:
        logger.error("LLM extraction failed for %s: %s", doc_type, exc)
        return {}


def process_document(filepath: Path | str) -> ProcessedDocument:
    """Process a single document file and return a ProcessedDocument."""
    filepath = Path(filepath)
    doc_type = _detect_doc_type(filepath.stem)

    raw_text = filepath.read_text(encoding="utf-8")
    clean_text = _clean_ocr_noise(raw_text)

    logger.info("Extracting structured data from %s...", filepath.name)
    extracted = _extract_with_llm(doc_type, clean_text)

    return ProcessedDocument(
        source_file=filepath.name,
        doc_type=doc_type,
        raw_text=raw_text,
        clean_text=clean_text,
        extracted=extracted,
    )


def process_all_documents(docs_dir: Path | None = None) -> list[ProcessedDocument]:
    """
    Process all .txt files in docs_dir (defaults to config.DOCS_DIR).
    Files are processed in alphabetical order for reproducibility.
    """
    target = Path(docs_dir) if docs_dir else DOCS_DIR
    document_files = sorted(target.glob("*.txt"))

    if not document_files:
        raise FileNotFoundError(f"No .txt files found in {target}")

    results = []
    for f in document_files:
        logger.info("Processing %s", f.name)
        results.append(process_document(f))

    return results
