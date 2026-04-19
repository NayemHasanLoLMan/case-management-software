# Architecture Overview, Assumptions, and Tradeoffs

## What the system does

This pipeline processes legal case documents, retrieves grounded evidence, and generates structured drafts that attorneys and processors can review. It then learns from operator corrections to produce better outputs over time.

## Architecture

The system is split into four clean modules, each with a single responsibility.

### processor.py — Document Processing

The processor has two phases. First it applies regex-based OCR cleanup. The title search documents in this case have systematic character substitutions typical of low-quality scans: digit-one replacing lowercase-L in words, capital-O replacing zero in numbers. These are handled with targeted regexes that only fire in contexts where the substitution is unambiguous (digit surrounded by letters, for instance). Anything the regex cannot confidently fix is left for the LLM to interpret.

The second phase sends each cleaned document to Gemini with a type-specific extraction prompt. Each prompt asks for a specific JSON schema — liens with instrument numbers and amounts, deadlines with dates and consequences, action items with priority levels. Returning structured JSON rather than free text means the extracted data can be queried programmatically and does not lose its meaning when passed to other components.

### retriever.py — Grounded Retrieval

The retriever chunks each processed document into overlapping 300-word windows (with a 50-word overlap to avoid cutting sentences across a boundary). Each chunk carries its source file name and document type so citations are always traceable.

Embeddings are computed locally using the all-MiniLM-L6-v2 sentence-transformer model. This was chosen for three reasons: it runs on CPU without any API cost, the model weights are small (~80MB), and it performs well on short factual passages. Similarity is computed with a numpy dot product on normalized vectors, which is equivalent to cosine similarity and avoids a FAISS dependency.

For production at scale, these chunks would be persisted to a vector database (pgvector on Postgres, or Pinecone) and the embedding calls would be batched. The interface `index.retrieve(query, top_k)` is the same regardless of backing store, so swapping is straightforward.

### generator.py — Draft Generation

Each draft generator retrieves relevant chunks for a query, then builds a prompt that includes those chunks as labeled evidence blocks. The prompt explicitly instructs the LLM to cite sources and to write "not found in documents" rather than fabricate missing information. The final output appends a SOURCES USED section with the source file name and relevance score for each retrieved chunk.

This design means every claim in a generated draft can be traced back to a specific passage in a specific source file. A reviewer can open the source file, find the passage, and verify the claim. That is the core grounding guarantee.

### learner.py — Improvement from Operator Edits

The improvement loop works in three steps.

First, each operator edit pair (system draft vs operator-edited version) is analyzed by the LLM. The LLM reads both versions and the operator-provided key_edits list, then extracts a set of concrete, reusable patterns. For example: "always include instrument numbers with every lien", "flag items requiring attorney action with ACTION REQUIRED:", "organize into labeled sections rather than a flat list".

Second, those patterns are stored as a style guide JSON file. The style guide is keyed by draft type so patterns for title reviews do not bleed into case memos.

Third, the style guide text is injected into the system prompt of future generation calls as an additional instruction block. This is a lightweight but effective approach — the patterns are written in plain language that the LLM can follow directly.

The improvement is demonstrated by generating a Document Checklist twice: once without the style guide (baseline) and once with it (improved). A third LLM call then compares the two and explains what improved and what is still missing. This gives a concrete, inspectable measure of progress.

## Assumptions

- The documents are in English and have consistent structure within each type (title search, court order, etc.).
- OCR noise is systematic, not random. The regex patterns are tuned to the specific substitutions seen in this dataset. A production system would need a broader cleanup pass for arbitrary OCR output.
- The Gemini API is available and the key has sufficient quota. All LLM calls are synchronous for simplicity — a production system would batch or parallelize extraction calls.
- The style guide accumulates improvements from every edit pair it sees. There is no weighting or recency bias applied — all patterns are treated equally.

## Tradeoffs

### In-memory index vs. persistent vector DB

The current index lives in memory and is rebuilt on every run. This is acceptable for a few hundred documents but would be slow at hundreds of cases with dozens of documents each. The fix is straightforward: serialize the chunk list and embedding matrix to disk, or switch to pgvector/Pinecone. The retriever interface does not change.

### Local embeddings vs. API embeddings

Using a local model means the first run is slower (model download) and embedding quality is decent but not state-of-the-art. Using the Gemini Embedding API or OpenAI's text-embedding-3-large would be higher quality but adds per-call cost and latency. For a production system processing high volumes, the tradeoff favors a hosted embedding API for quality, with local caching to manage cost.

### Pattern injection vs. fine-tuning

Injecting patterns as a prompt string is fast to implement and easy to inspect. The patterns are human-readable and can be reviewed or edited without any ML expertise. The downside is that very long style guides can crowd out other context, and the LLM may not follow all instructions equally. Fine-tuning on operator-edited pairs would produce stronger improvements but requires training data volume and compute that is not practical for a small dataset.

### Schema-per-document-type

Giving each document type its own extraction schema (court order schema vs. title search schema) produces cleaner structured output than a single generic schema. The tradeoff is that adding a new document type requires writing a new schema and extraction prompt. For this use case — a fixed set of legal document types — this is a reasonable choice. A more dynamic alternative would use an LLM to infer the schema from the document itself, but that produces less consistent output.

## How the improvement loop works end to end

1. Run `python run_pipeline.py` on a new case.
2. The pipeline processes documents, builds the index, and generates baseline drafts (Title Review Summary, Case Status Memo).
3. An operator reviews those drafts, makes corrections, and saves the before/after pair to sample_edits.json.
4. On the next run (or as an explicit step), `learner.build_style_guide()` analyzes the new pairs and updates the style guide.
5. Subsequent draft generation calls include the updated style guide, producing outputs closer to what the operator expects.
6. Each new edit pair makes the guide more specific. Over time, the baseline drafts require fewer corrections.

The key measurement is whether the improved draft contains the elements the operator consistently adds: labeled sections, instrument numbers, action flags, cross-document references. The `compare_drafts` function provides a text evaluation of this, and the output files provide a direct side-by-side for human review.

## Scaling to multiple cases

The pipeline is designed to handle any case without code changes. The only case-specific file is `case_context.json`, which is passed as context to the LLM. Swapping in a different case means pointing the config at a different data directory. The document type detection is based on filename conventions, so as long as new cases follow the same naming pattern the processors work as-is.

For a production system handling hundreds of cases, the architecture would add a case-level metadata store (Postgres), a queue for async processing jobs (Celery or a cloud queue), and a shared vector index partitioned by case. The module interfaces remain the same — only the infrastructure backing them changes.
