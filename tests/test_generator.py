"""
Tests for pipeline/generator.py

The LLM is mocked for all tests so the suite runs offline and without API keys.
We test that:
  - Each generator calls the LLM with the right content in the prompt
  - The grounding rule appears in every prompt
  - Source citations are appended to the output
  - The GENERATORS registry contains all expected types
  - generate_all_drafts() honours the draft_types filter
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.generator import (
    GENERATORS,
    generate_all_drafts,
    generate_case_status_memo,
    generate_document_checklist,
    generate_title_review_summary,
    generate_action_item_extract,
    _GROUNDING_RULE,
)
from pipeline.models import Chunk, ProcessedDocument, RetrievedChunk
from pipeline.retriever import DocumentIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(source_file: str, doc_type: str) -> ProcessedDocument:
    return ProcessedDocument(
        source_file=source_file,
        doc_type=doc_type,
        raw_text="raw",
        clean_text="clean content about liens and deadlines",
        extracted={},
    )


def _fake_retrieve(query, top_k=5):
    """Return a single fake chunk regardless of query."""
    chunk = Chunk(
        chunk_id="fake_0",
        source_file="title_search_page1.txt",
        doc_type="title_search",
        text="The mortgage is $445,000 recorded February 2021",
    )
    return [RetrievedChunk(chunk=chunk, score=0.75)]


def _make_index_with_fake_retrieve():
    index = MagicMock(spec=DocumentIndex)
    index.retrieve.side_effect = _fake_retrieve
    index.retrieve_for_doc_type.side_effect = _fake_retrieve
    return index


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestGenerateWithMockedLLM:

    @pytest.fixture
    def index(self):
        return _make_index_with_fake_retrieve()

    @pytest.fixture
    def docs(self):
        return [_make_doc("title_search_page1.txt", "title_search")]

    def _run_generator(self, gen_fn, index, docs):
        with patch("pipeline.generator._call_llm", return_value="Generated draft text.") as mock_llm:
            result = gen_fn(index, docs)
        return result, mock_llm

    def test_title_review_summary_calls_llm(self, index, docs):
        result, mock_llm = self._run_generator(generate_title_review_summary, index, docs)
        mock_llm.assert_called_once()

    def test_case_status_memo_calls_llm(self, index, docs):
        result, mock_llm = self._run_generator(generate_case_status_memo, index, docs)
        mock_llm.assert_called_once()

    def test_document_checklist_calls_llm(self, index, docs):
        result, mock_llm = self._run_generator(generate_document_checklist, index, docs)
        mock_llm.assert_called_once()

    def test_action_item_extract_calls_llm(self, index, docs):
        result, mock_llm = self._run_generator(generate_action_item_extract, index, docs)
        mock_llm.assert_called_once()

    def test_grounding_rule_in_prompt(self, index, docs):
        captured_prompt = []

        def capture(prompt):
            captured_prompt.append(prompt)
            return "some output"

        with patch("pipeline.generator._call_llm", side_effect=capture):
            generate_title_review_summary(index, docs)

        assert _GROUNDING_RULE in captured_prompt[0]

    def test_citation_block_appended(self, index, docs):
        with patch("pipeline.generator._call_llm", return_value="draft body"):
            result = generate_title_review_summary(index, docs)

        assert "SOURCES USED" in result.content
        assert "title_search_page1.txt" in result.content

    def test_result_has_correct_draft_type(self, index, docs):
        with patch("pipeline.generator._call_llm", return_value="text"):
            result = generate_title_review_summary(index, docs)
        assert result.draft_type == "title_review_summary"

    def test_result_sources_match_retrieved_chunks(self, index, docs):
        with patch("pipeline.generator._call_llm", return_value="text"):
            result = generate_title_review_summary(index, docs)
        assert len(result.sources) >= 1

    def test_style_guide_injected_into_prompt(self, index, docs):
        captured = []

        def capture(prompt):
            captured.append(prompt)
            return "output"

        with patch("pipeline.generator._call_llm", side_effect=capture):
            generate_case_status_memo(index, docs, style_guide="always include deadlines")

        assert "always include deadlines" in captured[0]


# ---------------------------------------------------------------------------
# GENERATORS registry
# ---------------------------------------------------------------------------

class TestGeneratorsRegistry:

    def test_all_four_types_registered(self):
        expected = {
            "title_review_summary",
            "case_status_memo",
            "document_checklist",
            "action_item_extract",
        }
        assert set(GENERATORS.keys()) == expected

    def test_all_registered_values_are_callable(self):
        for name, fn in GENERATORS.items():
            assert callable(fn), f"{name} is not callable"


# ---------------------------------------------------------------------------
# generate_all_drafts
# ---------------------------------------------------------------------------

class TestGenerateAllDrafts:

    @pytest.fixture
    def index(self):
        return _make_index_with_fake_retrieve()

    @pytest.fixture
    def docs(self):
        return [_make_doc("court_order.txt", "court_order")]

    def test_generates_requested_types_only(self, index, docs):
        with patch("pipeline.generator._call_llm", return_value="text"):
            results = generate_all_drafts(
                index, docs,
                draft_types=["title_review_summary", "case_status_memo"],
            )
        types = {r.draft_type for r in results}
        assert types == {"title_review_summary", "case_status_memo"}

    def test_unknown_type_skipped(self, index, docs):
        with patch("pipeline.generator._call_llm", return_value="text"):
            results = generate_all_drafts(
                index, docs,
                draft_types=["title_review_summary", "nonexistent_type"],
            )
        assert len(results) == 1
        assert results[0].draft_type == "title_review_summary"

    def test_default_runs_all_four(self, index, docs):
        with patch("pipeline.generator._call_llm", return_value="text"):
            results = generate_all_drafts(index, docs)
        assert len(results) == 4
