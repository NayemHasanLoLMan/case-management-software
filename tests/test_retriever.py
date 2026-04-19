"""
Tests for pipeline/retriever.py

Tests cover:
  - Chunking behavior (size, overlap, edge cases)
  - Index building
  - Retrieval (returns correct number, ordered by score)
  - retrieve_for_doc_type filtering
  - format_retrieved_context output format

Embedding is mocked so tests run without downloading the model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.models import Chunk, ProcessedDocument, RetrievedChunk
from pipeline.retriever import DocumentIndex, format_retrieved_context


def _make_doc(source_file: str, doc_type: str, text: str) -> ProcessedDocument:
    return ProcessedDocument(
        source_file=source_file,
        doc_type=doc_type,
        raw_text=text,
        clean_text=text,
        extracted={},
    )


def _mock_encode(texts, **kwargs):
    """Return deterministic normalized vectors with the real model's 384 dimensions."""
    n = len(texts)
    embs = np.zeros((n, 384), dtype=np.float32)
    for i in range(n):
        embs[i, i % 384] = 1.0
    return embs

class TestChunking:

    def test_short_doc_produces_one_chunk(self):
        doc = _make_doc("a.txt", "court_order", "short text")
        index = DocumentIndex()
        chunks = index._split_into_chunks(doc)
        assert len(chunks) == 1

    def test_chunk_ids_are_unique(self):
        # long enough document to produce multiple chunks
        words = ["word"] * 1000
        doc = _make_doc("title_search_page1.txt", "title_search", " ".join(words))
        index = DocumentIndex()
        chunks = index._split_into_chunks(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_all_chunks_carry_source_info(self):
        doc = _make_doc("servicer_email.txt", "servicer_email", "word " * 400)
        index = DocumentIndex()
        chunks = index._split_into_chunks(doc)
        for chunk in chunks:
            assert chunk.source_file == "servicer_email.txt"
            assert chunk.doc_type == "servicer_email"

    def test_empty_document_produces_no_chunks(self):
        doc = _make_doc("empty.txt", "unknown", "")
        index = DocumentIndex()
        chunks = index._split_into_chunks(doc)
        assert len(chunks) == 0


class TestIndexBuilding:

    def test_index_empty_before_build(self):
        index = DocumentIndex()
        assert index.chunk_count == 0

    def test_index_raises_on_empty_documents(self):
        index = DocumentIndex()
        with pytest.raises(ValueError):
            with patch.object(index, "_split_into_chunks", return_value=[]):
                index.index([_make_doc("a.txt", "unknown", "")])

    def test_chunk_count_after_indexing(self):
        docs = [
            _make_doc("court_order.txt", "court_order", "word " * 50),
            _make_doc("servicer_email.txt", "servicer_email", "word " * 50),
        ]
        index = DocumentIndex()
        with patch.object(index, "_get_model") as mock_model:
            mock_model.return_value.encode = _mock_encode
            index.index(docs)
        assert index.chunk_count >= 2

class TestRetrieval:

    @pytest.fixture
    def built_index(self):
        docs = [
            _make_doc("title_search_page1.txt", "title_search", "lien mortgage amount instrument"),
            _make_doc("servicer_email.txt", "servicer_email", "servicer transfer deadline billing"),
            _make_doc("court_order.txt", "court_order", "conference deadline filing proof service"),
        ]
        mock_model = MagicMock()
        mock_model.encode = _mock_encode
        index = DocumentIndex()
        # patch _get_model so both index() and retrieve() use the same mock
        with patch.object(index, "_get_model", return_value=mock_model):
            index.index(docs)
        # keep the patch alive for the fixture consumer
        index._model = mock_model
        return index

    def test_retrieve_before_index_raises(self):
        index = DocumentIndex()
        with pytest.raises(RuntimeError):
            index.retrieve("test query")

    def test_retrieve_returns_correct_count(self, built_index):
        index = built_index
        results = index.retrieve("lien mortgage", top_k=2)
        assert len(results) == 2

    def test_retrieve_results_are_ordered_by_score(self, built_index):
        index = built_index
        results = index.retrieve("any query", top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_for_doc_type_filters_correctly(self, built_index):
        index = built_index
        results = index.retrieve_for_doc_type("any", doc_type="court_order", top_k=5)
        for r in results:
            assert r.chunk.doc_type == "court_order"


class TestFormatRetrievedContext:

    def test_includes_source_file_name(self):
        chunk = Chunk(
            chunk_id="x_0",
            source_file="court_order.txt",
            doc_type="court_order",
            text="the conference is on April 22",
        )
        rc = RetrievedChunk(chunk=chunk, score=0.85)
        output = format_retrieved_context([rc])
        assert "court_order.txt" in output

    def test_includes_chunk_text(self):
        chunk = Chunk(
            chunk_id="x_0",
            source_file="servicer_email.txt",
            doc_type="servicer_email",
            text="payoff is $487,920.00",
        )
        rc = RetrievedChunk(chunk=chunk, score=0.5)
        output = format_retrieved_context([rc])
        assert "$487,920.00" in output

    def test_multiple_chunks_are_numbered(self):
        chunks = [
            RetrievedChunk(
                chunk=Chunk(chunk_id=f"x_{i}", source_file="a.txt", doc_type="x", text=f"text {i}"),
                score=0.9 - i * 0.1,
            )
            for i in range(3)
        ]
        output = format_retrieved_context(chunks)
        assert "Source 1" in output
        assert "Source 2" in output
        assert "Source 3" in output
