"""
Pipeline entry point.

Runs all four capabilities in sequence:
  1. Document processing — OCR cleanup + structured extraction
  2. Index building — chunk, embed, build retrieval index
  3. Baseline draft generation — Title Review Summary + Case Status Memo
  4. Edit learning + improved draft — learn from sample_edits.json, regenerate

All outputs are saved to sample_outputs/.
Run with: python run_pipeline.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from pipeline.config import GEMINI_API_KEY, OUTPUT_DIR
from pipeline.generator import generate_all_drafts, generate_document_checklist
from pipeline.learner import (
    build_style_guide,
    compare_drafts,
    format_style_guide_for_prompt,
    load_edit_pairs,
)
from pipeline.processor import process_all_documents
from pipeline.retriever import DocumentIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_api_key() -> None:
    if not GEMINI_API_KEY:
        print(
            "ERROR: GEMINI_API_KEY is not set.\n"
            "Create a .env file in the project root with:\n"
            "  GEMINI_API_KEY=your_key_here\n"
            "or export the variable directly before running."
        )
        sys.exit(1)


def _save(name: str, content: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(content, encoding="utf-8")
    logger.info("Saved %s", path.name)
    return path


def _separator(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _check_api_key()

    print("\n" + "=" * 60)
    print("  AI Case Management Pipeline")
    print("  Rodriguez Case — 2025-FC-08891")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Part 1: Document Processing
    # -----------------------------------------------------------------------
    _separator("Part 1: Document Processing")
    docs = process_all_documents()

    # save individual cleaned documents
    for doc in docs:
        _save(f"01_clean_{doc.source_file}", doc.clean_text)

    # save combined extraction results
    extraction_summary = [
        {
            "source_file": doc.source_file,
            "doc_type": doc.doc_type,
            "extracted_fields": doc.extracted,
        }
        for doc in docs
    ]
    _save("01_extraction_results.json", json.dumps(extraction_summary, indent=2, ensure_ascii=False))
    print(f"Processed {len(docs)} documents.\n")

    # -----------------------------------------------------------------------
    # Part 2: Index Building + Retrieval Demo
    # -----------------------------------------------------------------------
    _separator("Part 2: Retrieval Index")
    index = DocumentIndex()
    index.index(docs)
    print(f"Index ready: {index.chunk_count} chunks\n")

    demo_query = "HOA lis pendens amount instrument number"
    demo_results = index.retrieve(demo_query, top_k=3)
    demo_lines = [
        f"  [{r.chunk.source_file} | {r.score:.2f}] {r.chunk.text[:100]}..."
        for r in demo_results
    ]
    _save(
        "02_retrieval_demo.txt",
        f"Query: {demo_query}\n\nTop results:\n" + "\n".join(demo_lines),
    )
    print(f"Demo retrieval — '{demo_query}':")
    for line in demo_lines:
        print(line)
    print()

    # -----------------------------------------------------------------------
    # Part 3: Baseline Draft Generation
    # -----------------------------------------------------------------------
    _separator("Part 3: Baseline Draft Generation")
    baseline_drafts = generate_all_drafts(
        index, docs,
        draft_types=["title_review_summary", "case_status_memo"],
    )
    baseline_texts: dict[str, str] = {}
    for draft in baseline_drafts:
        _save(f"03_baseline_{draft.draft_type}.txt", draft.content)
        baseline_texts[draft.draft_type] = draft.content
        print(f"Generated {draft.draft_type} ({len(draft.content)} chars)")
    print()

    # -----------------------------------------------------------------------
    # Part 4: Learning from Operator Edits
    # -----------------------------------------------------------------------
    _separator("Part 4: Learning from Operator Edits")

    edit_pairs = load_edit_pairs()
    print(f"Loaded {len(edit_pairs)} edit pair(s) from sample_edits.json\n")

    style_guide = build_style_guide(edit_pairs)
    _save("04_learned_style_guide.json", json.dumps(style_guide, indent=2, ensure_ascii=False))

    style_guide_text = format_style_guide_for_prompt(style_guide)

    # generate baseline + improved document checklist to show measurable change
    print("Generating baseline Document Checklist (no style guide)...")
    baseline_checklist = generate_document_checklist(index, docs, style_guide="")
    _save("04_baseline_document_checklist.txt", baseline_checklist.content)

    print("Generating improved Document Checklist (with style guide)...")
    improved_checklist = generate_document_checklist(index, docs, style_guide=style_guide_text)
    _save("04_improved_document_checklist.txt", improved_checklist.content)

    print("Evaluating improvement...\n")
    evaluation = compare_drafts(baseline_checklist.content, improved_checklist.content)
    _save("04_improvement_evaluation.txt", evaluation)
    print("Improvement evaluation:\n")
    print(evaluation)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Pipeline complete. Outputs in sample_outputs/:")
    print("=" * 60)
    all_outputs = sorted(OUTPUT_DIR.glob("0*.txt")) + sorted(OUTPUT_DIR.glob("0*.json"))
    for path in all_outputs:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name:<50} {size_kb:>5.1f} KB")
    print()


if __name__ == "__main__":
    main()
