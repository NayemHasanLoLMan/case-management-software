"""
Tests for pipeline/processor.py

These tests cover the parts of the processor that don't require a live LLM call:
  - OCR noise cleanup regexes
  - Document type detection
  - Prompt building (structure only)
  - process_all_documents() with a temporary directory

LLM extraction is tested separately with a mock so the test suite runs without
an API key and without network access.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.processor import (
    _clean_ocr_noise,
    _detect_doc_type,
    _build_extraction_prompt,
    process_document,
    process_all_documents,
)

class TestCleanOcrNoise:

    def test_digit_one_inside_word(self):
        assert _clean_ocr_noise("tit1e") == "title"

    def test_digit_one_at_word_start(self):
        assert _clean_ocr_noise("1ien") == "lien"

    def test_capital_o_between_digits(self):
        assert _clean_ocr_noise("2O21") == "2021"

    def test_capital_o_in_dollar_amount(self):
        assert _clean_ocr_noise("$445,OOO.OO") == "$445,000.00"

    def test_capital_o_in_instrument_number(self):
        # instrument number pattern like "2O21-O123456"
        result = _clean_ocr_noise("Instrument No. 2O21-O123456")
        assert "2021-0123456" in result

    def test_fi_ligature(self):
        assert _clean_ocr_noise("fi1e") == "file"

    def test_clean_text_unchanged(self):
        clean = "This text has no OCR noise at all."
        assert _clean_ocr_noise(clean) == clean

    def test_realistic_title_search_line(self):
        noisy = "origina1 amount of $445,OOO.OO dated February 8, 2O21"
        result = _clean_ocr_noise(noisy)
        assert "original" in result
        assert "$445,000.00" in result
        assert "2021" in result

    def test_does_not_corrupt_normal_numbers(self):
        # The number 10 should not become "l0" — digit after letter fixes nothing
        result = _clean_ocr_noise("Section 10 applies")
        assert "10" in result

class TestDetectDocType:

    def test_title_search_page1(self):
        assert _detect_doc_type("title_search_page1") == "title_search"

    def test_title_search_page2(self):
        assert _detect_doc_type("title_search_page2") == "title_search"

    def test_servicer_email(self):
        assert _detect_doc_type("servicer_email") == "servicer_email"

    def test_court_order(self):
        assert _detect_doc_type("court_order") == "court_order"

    def test_unknown_file(self):
        assert _detect_doc_type("random_document") == "unknown"

class TestBuildExtractionPrompt:

    def test_contains_document_text(self):
        prompt = _build_extraction_prompt("court_order", "some court text")
        assert "some court text" in prompt

    def test_returns_json_instruction(self):
        prompt = _build_extraction_prompt("title_search", "text")
        assert "JSON" in prompt

    def test_unknown_type_still_returns_prompt(self):
        prompt = _build_extraction_prompt("unknown", "text")
        assert len(prompt) > 20

class TestProcessDocument:

    def test_returns_processed_document(self, tmp_path):
        doc_file = tmp_path / "servicer_email.txt"
        doc_file.write_text("From: test@example.com\nSubject: Test\n\nHello.", encoding="utf-8")

        mock_extracted = {"from": "test@example.com", "subject": "Test"}

        with patch("pipeline.processor._extract_with_llm", return_value=mock_extracted):
            result = process_document(doc_file)

        assert result.source_file == "servicer_email.txt"
        assert result.doc_type == "servicer_email"
        assert result.extracted == mock_extracted
        assert "Hello" in result.clean_text

    def test_ocr_cleanup_applied(self, tmp_path):
        doc_file = tmp_path / "title_search_page1.txt"
        doc_file.write_text("origina1 amount $445,OOO.OO", encoding="utf-8")

        with patch("pipeline.processor._extract_with_llm", return_value={}):
            result = process_document(doc_file)

        assert "original" in result.clean_text
        assert "$445,000.00" in result.clean_text

    def test_raw_text_preserved(self, tmp_path):
        raw = "origina1 amount $445,OOO.OO"
        doc_file = tmp_path / "title_search_page1.txt"
        doc_file.write_text(raw, encoding="utf-8")

        with patch("pipeline.processor._extract_with_llm", return_value={}):
            result = process_document(doc_file)

        # the raw text must be exactly as read from disk
        assert result.raw_text == raw

class TestProcessAllDocuments:

    def test_processes_all_txt_files(self, tmp_path):
        (tmp_path / "servicer_email.txt").write_text("email content", encoding="utf-8")
        (tmp_path / "court_order.txt").write_text("order content", encoding="utf-8")

        with patch("pipeline.processor._extract_with_llm", return_value={}):
            results = process_all_documents(docs_dir=tmp_path)

        assert len(results) == 2

    def test_raises_when_no_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            process_all_documents(docs_dir=tmp_path)

    def test_alphabetical_order(self, tmp_path):
        for name in ["court_order.txt", "servicer_email.txt", "title_search_page1.txt"]:
            (tmp_path / name).write_text("text", encoding="utf-8")

        with patch("pipeline.processor._extract_with_llm", return_value={}):
            results = process_all_documents(docs_dir=tmp_path)

        filenames = [r.source_file for r in results]
        assert filenames == sorted(filenames)
