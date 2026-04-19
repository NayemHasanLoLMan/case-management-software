# AI Engineer — Take-Home Assessment

## Document Understanding, Grounded Drafting, and Improvement from Edits

---

## Context

You are an AI Engineer at a company building intelligent case management software for the professional services industry. The system processes high volumes of documents — title searches, servicer communications, court orders, property records — many of which arrive in messy formats: scanned PDFs, noisy OCR output, inconsistently structured emails, and formal legal filings.

Your job is to build part of the internal AI pipeline that takes these documents, extracts usable information from them, and produces **grounded draft outputs** that attorneys and processors can review, edit, and act on.

The system generates first-pass drafts — title review summaries, case status memos, document checklists — that must be **grounded in the actual source documents**, not based on unsupported assumptions. When operators edit these drafts, the system should learn from those edits to improve future outputs.

---

## What to Build

A pipeline with four capabilities:

1. **Process** messy documents — extract usable text and structured information from noisy, inconsistent inputs
2. **Retrieve** relevant context — use retrieval over processed documents so outputs are grounded in source material
3. **Generate** draft outputs — produce structured, evidence-backed drafts (title review summaries, case status memos, etc.)
4. **Improve** from operator edits — capture how operators correct the defaults, learn patterns, and produce better outputs over time

---

## Technical Stack

- **Required:** Python 3.10+
- **Required:** LLM usage for extraction, retrieval, and generation. Use your own API keys with any provider.
- **Your choice:** Any additional libraries, vector stores, or frameworks.

---

## Sample Data

We provide a complete set of documents for one case. Your system should process these and demonstrate all four capabilities against them.

All data is in `ai_engineer_assignment_data/`.

### Case Context

> **See:** `case_context.json`

The Rodriguez case (2025-FC-08891) — a foreclosure case in Miami-Dade County, Florida with several active complications: a pending servicer transfer, an HOA lis pendens, borrower has retained counsel, and a case management conference is approaching with multiple court deadlines.

### Source Documents

> **See:** `sample_documents/` (4 files)

| Document | What It Is | Key Challenges |
|---|---|---|
| `title_search_page1.txt` | Schedule B exceptions from a title search | **Noisy OCR** — character substitutions (`1` for `l`, `O` for `0`), degraded text. Contains liens, mortgage info, assignments, ownership chain. |
| `title_search_page2.txt` | Legal description, tax info, judgment search | Cleaner text but structured differently — tables, mixed formats, cross-references to Schedule B. |
| `servicer_email.txt` | Email from loan servicer with multiple instructions | Unstructured email with 5+ action items buried in prose — servicing transfer, borrower counsel info, payoff update, HOA concern. |
| `court_order.txt` | Court order setting a case management conference | Formal legal document with specific deadlines, filing requirements, and appearance obligations. |

These documents are realistic — the title search has actual OCR noise, the email mixes formal instructions with casual asides, and the court order has structured legal language.

### Sample Operator Edits

> **See:** `sample_edits.json`

2 before/after pairs showing how an operator improved system-generated drafts:

1. **Title Review Summary** — the system produced a flat list of liens; the operator reorganized into labeled sections, added instrument numbers, flagged action items, and added reviewer notes.
2. **Case Status Memo** — the system produced a passive summary; the operator added prioritized action items, cross-referenced deadlines across documents, and connected information from title search + email + court order into a unified picture.

Each pair includes `key_edits` — a list explaining what the operator changed and why. Use these to build your improvement mechanism.

---

## Requirements

### 1. Document Processing (25 points)

Your system should accept the sample documents and extract useful content from them.

This includes:
- Handling OCR noise in the title search (character substitutions, formatting artifacts)
- Extracting structured data: liens with amounts/dates/instrument numbers, deadlines from the court order, action items from the email
- Producing clean, structured output that can be used for retrieval and drafting

We evaluate: how well the system handles messy inputs, extraction quality, usefulness of structured outputs, whether the extracted data is accurate.

### 2. Grounded Retrieval (25 points)

Your system should use retrieval over the processed documents so that generated outputs are based on actual source material.

We expect the system to:
- Index the processed documents for retrieval
- When generating a draft, retrieve relevant sections from the source documents
- Make it possible to **inspect what evidence** each part of the output is based on (citations, source references, or retrieved chunks)
- Control unsupported generation — if the source documents don't contain information, the system should say so rather than fabricate

We evaluate: retrieval quality, relevance of retrieved context, whether outputs are grounded, whether evidence is inspectable, how well hallucination is controlled.

### 3. Draft Generation (10 points)

Using the processed content and retrieved evidence, generate draft outputs for the Rodriguez case.

Generate at least TWO of the following:
- **Title Review Summary** — structured summary of liens, encumbrances, ownership, and tax status from the title search documents
- **Case Status Memo** — cross-document summary pulling deadlines, action items, and status from all four documents
- **Document Checklist** — what's been filed, what's missing, what deadlines are approaching
- **Action Item Extract** — prioritized list of tasks extracted from the servicer email and court order

The drafts should be:
- Grounded in the source documents (not generic boilerplate)
- Reasonably structured and useful as a first pass
- Cite or reference the source material they're based on

We are not evaluating legal correctness. We are evaluating whether the output is well-supported by the provided materials and whether the system architecture is sound.

### 4. Improvement from Operator Edits (25 points)

After the system generates a default draft, an operator reviews it and makes edits. Your system should learn from those edits.

Using the 2 sample edit pairs provided in `sample_edits.json`:
- **Capture** the structured differences between the system draft and operator version
- **Identify** what patterns can be learned (section organization, information completeness, action item prioritization, cross-document synthesis)
- **Apply** those learnings to improve future outputs — generate a new draft for a different document type and show that the improvements carry over

We want to see a **meaningful improvement loop**, not just version comparison. Demonstrate that the system produces measurably better output after ingesting the operator edits.

### 5. Code Quality and System Design (10 points)

We evaluate:
- Code organization and modularity
- Clean separation between processing, retrieval, generation, and learning components
- Error handling and edge cases
- Whether the architecture could scale to handle hundreds of cases and document types

### 6. Documentation and Clarity (5 points)

We evaluate:
- Ease of setup and running
- Quality of the architecture overview
- Clarity of assumptions and tradeoffs explanation

---

## What to Submit

**Required:**
- Source code
- `README.md` with setup and run instructions
- `APPROACH.md` — short architecture overview, assumptions, tradeoffs, and how the improvement loop works
- Sample inputs and outputs (your system's actual output on the provided documents)
- Evaluation approach — how you measured whether outputs are grounded and whether the improvement loop works

**Optional (but encouraged):**
- REST API endpoints
- Tests
- Docker setup
- Simple UI for document upload → draft review → edit capture

---

## Constraints

- **Time budget:** 8-12 hours. This is a broader assessment than a single-feature task. Prioritize the components you're strongest in — a well-built document processor + retrieval system with basic generation is better than a shallow pass across all four areas.
- **LLM usage is required.** Use your own API keys. Any provider is fine.
- **You may create additional mock documents** if you want to demonstrate your system handling more variety. The provided documents are the minimum required input.
- **No domain expertise needed.** The documents are self-explanatory. You don't need to know foreclosure law.

---

## Evaluation Rubric (100 Points)

| Area | Points | What We Evaluate |
|---|---|---|
| **Document Processing** | 25 | OCR noise handling, extraction quality, structured output usefulness |
| **Retrieval & Grounding** | 25 | Retrieval quality, evidence traceability, hallucination control |
| **Draft Quality** | 10 | Usefulness, structure, consistency with source documents |
| **Improvement from Edits** | 25 | Edit capture, pattern learning, measurable improvement in future outputs |
| **Code Quality & Design** | 10 | Organization, modularity, error handling, scalability |
| **Documentation & Clarity** | 5 | Setup ease, architecture explanation, reviewer experience |

---

## Reviewer Focus

A strong submission clearly shows:
- How messy documents are processed into clean, structured data
- How retrieval grounds the generated output in actual source evidence
- How the draft stays factual — citing sources, not fabricating
- How operator edits produce measurably better future outputs
- That the system could handle a different case with different documents without code changes

Good luck. We look forward to reviewing your work.
