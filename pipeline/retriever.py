"""
Retriever — Part 2 of the pipeline.

Responsibilities:
  - Chunk processed documents into overlapping passages
  - Embed them using a local sentence-transformer model (no API cost)
  - Store in memory as a numpy matrix for cosine similarity search
  - Retrieve relevant chunks for a query with full source attribution

Design notes:
  Using sentence-transformers locally so no data leaves the machine and
  there is no per-call embedding cost. For production with large document
  volumes, swap the backing store for pgvector or Pinecone — the retrieve()
  interface stays the same regardless.

  The index is rebuilt each run. Persisting it to disk is a one-line change
  (numpy.save / numpy.load) if startup speed becomes a concern.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, TOP_K
from pipeline.models import Chunk, ProcessedDocument, RetrievedChunk

logger = logging.getLogger(__name__)


class DocumentIndex:
    """
    In-memory document index backed by numpy cosine similarity.

    Usage:
        index = DocumentIndex()
        index.index(processed_docs)
        results = index.retrieve("HOA lien amount", top_k=5)
    """

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None
        self._chunks: list[Chunk] = []
        self._matrix: Optional[np.ndarray] = None  # (n_chunks, embed_dim)

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model so startup is fast when not needed."""
        if self._model is None:
            logger.info(
                "Loading embedding model %s (first run downloads ~80 MB)...",
                EMBEDDING_MODEL,
            )
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _split_into_chunks(self, doc: ProcessedDocument) -> list[Chunk]:
        """
        Split document text into overlapping word-level windows.

        Word count is used as a proxy for token count — close enough for
        retrieval purposes with these document sizes. The overlap prevents
        relevant sentences from being split across a chunk boundary.
        """
        words = doc.clean_text.split()
        step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
        chunks: list[Chunk] = []

        for chunk_num, start in enumerate(range(0, len(words), step)):
            window = words[start: start + CHUNK_SIZE]
            if not window:
                break
            chunk_id = f"{doc.source_file.replace('.txt', '')}_chunk{chunk_num}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_file=doc.source_file,
                    doc_type=doc.doc_type,
                    text=" ".join(window),
                )
            )

        return chunks

    def index(self, documents: list[ProcessedDocument]) -> None:
        """Build the index from a list of processed documents."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self._split_into_chunks(doc)
            logger.info("  %s -> %d chunk(s)", doc.source_file, len(chunks))
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No chunks produced — check that documents have content")

        model = self._get_model()
        texts = [c.text for c in all_chunks]
        logger.info("Embedding %d chunks with %s ...", len(all_chunks), EMBEDDING_MODEL)

        # normalize_embeddings=True means dot product == cosine similarity
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embedding = emb.tolist()

        self._chunks = all_chunks
        self._matrix = np.array(embeddings, dtype=np.float32)
        logger.info("Index built: %d chunks ready", len(self._chunks))

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks for the given query string."""
        if self._matrix is None or not self._chunks:
            raise RuntimeError("Index is empty — call index() before retrieve()")

        model = self._get_model()
        q_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

        # dot product on normalized vectors == cosine similarity
        scores = self._matrix @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def retrieve_for_doc_type(
        self, query: str, doc_type: str, top_k: int = TOP_K
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks filtered to a specific document type.
        Fetches top_k * 3 broadly then filters, to ensure enough results survive.
        """
        broad = self.retrieve(query, top_k=top_k * 3)
        filtered = [r for r in broad if r.chunk.doc_type == doc_type]
        return filtered[:top_k]

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a labeled evidence block for LLM prompts.
    Each chunk includes its source file so the model can cite it.
    """
    parts = [
        f"[Source {i}: {rc.chunk.source_file} | relevance={rc.score:.2f}]\n{rc.chunk.text}"
        for i, rc in enumerate(chunks, start=1)
    ]
    return "\n\n".join(parts)
