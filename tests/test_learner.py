"""
Tests for pipeline/learner.py

All LLM calls are mocked. Tests cover:
  - load_edit_pairs reads the JSON correctly
  - analyze_edit_pair returns correct structure (LLM path and fallback path)
  - _strip_code_fences removes markdown fences
  - build_style_guide creates the expected structure and saves to disk
  - format_style_guide_for_prompt output is non-empty and contains instructions
  - compare_drafts calls the LLM with both inputs
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest

from pipeline.learner import (
    _strip_code_fences,
    analyze_edit_pair,
    build_style_guide,
    format_style_guide_for_prompt,
    compare_drafts,
)

class TestStripCodeFences:

    def test_strips_json_fence(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        result = _strip_code_fences(text)
        assert result == '{"key": "value"}'

    def test_strips_plain_fence(self):
        text = "```\n{\"key\": 1}\n```"
        result = _strip_code_fences(text)
        assert result == '{"key": 1}'

    def test_no_fences_unchanged(self):
        text = '{"key": "value"}'
        assert _strip_code_fences(text) == text

    def test_strips_leading_whitespace(self):
        text = '  {"key": 1}'
        result = _strip_code_fences(text)
        assert result == '{"key": 1}'


SAMPLE_PAIR = {
    "draft_type": "title_review_summary",
    "system_draft": "flat list of liens",
    "operator_edited_version": "organized sections with instrument numbers",
    "key_edits": ["add instrument numbers", "organize into sections"],
}

VALID_PATTERN_JSON = json.dumps({
    "draft_type": "title_review_summary",
    "patterns": [
        {"pattern_name": "instrument_numbers", "instruction": "always include instrument numbers"},
        {"pattern_name": "sections", "instruction": "organize into labeled sections"},
    ],
})


class TestAnalyzeEditPair:

    def test_returns_correct_draft_type(self):
        with patch("pipeline.learner._call_llm", return_value=VALID_PATTERN_JSON):
            result = analyze_edit_pair(SAMPLE_PAIR)
        assert result["draft_type"] == "title_review_summary"

    def test_returns_patterns_list(self):
        with patch("pipeline.learner._call_llm", return_value=VALID_PATTERN_JSON):
            result = analyze_edit_pair(SAMPLE_PAIR)
        assert isinstance(result["patterns"], list)
        assert len(result["patterns"]) > 0

    def test_patterns_have_required_keys(self):
        with patch("pipeline.learner._call_llm", return_value=VALID_PATTERN_JSON):
            result = analyze_edit_pair(SAMPLE_PAIR)
        for p in result["patterns"]:
            assert "pattern_name" in p
            assert "instruction" in p

    def test_falls_back_to_key_edits_on_bad_json(self):
        with patch("pipeline.learner._call_llm", return_value="not valid json at all"):
            result = analyze_edit_pair(SAMPLE_PAIR)
        # fallback should use key_edits
        assert result["draft_type"] == "title_review_summary"
        instructions = [p["instruction"] for p in result["patterns"]]
        assert "add instrument numbers" in instructions

    def test_llm_receives_both_drafts(self):
        captured = []

        def capture(prompt):
            captured.append(prompt)
            return VALID_PATTERN_JSON

        with patch("pipeline.learner._call_llm", side_effect=capture):
            analyze_edit_pair(SAMPLE_PAIR)

        assert "flat list of liens" in captured[0]
        assert "organized sections" in captured[0]


class TestBuildStyleGuide:

    def test_returns_correct_structure(self, tmp_path):
        with patch("pipeline.learner._call_llm", return_value=VALID_PATTERN_JSON):
            with patch("pipeline.learner.STYLE_GUIDE_PATH", tmp_path / "guide.json"):
                with patch("pipeline.learner.OUTPUT_DIR", tmp_path):
                    result = build_style_guide([SAMPLE_PAIR])

        assert "patterns_by_draft_type" in result
        assert "title_review_summary" in result["patterns_by_draft_type"]

    def test_saves_to_disk(self, tmp_path):
        guide_path = tmp_path / "guide.json"
        with patch("pipeline.learner._call_llm", return_value=VALID_PATTERN_JSON):
            with patch("pipeline.learner.STYLE_GUIDE_PATH", guide_path):
                with patch("pipeline.learner.OUTPUT_DIR", tmp_path):
                    build_style_guide([SAMPLE_PAIR])

        assert guide_path.exists()
        saved = json.loads(guide_path.read_text())
        assert "patterns_by_draft_type" in saved

    def test_handles_multiple_edit_pairs(self, tmp_path):
        memo_pair = {
            "draft_type": "case_status_memo",
            "system_draft": "basic memo",
            "operator_edited_version": "prioritized memo",
            "key_edits": ["add priorities"],
        }
        memo_json = json.dumps({
            "draft_type": "case_status_memo",
            "patterns": [{"pattern_name": "priority", "instruction": "add priorities"}],
        })

        responses = [VALID_PATTERN_JSON, memo_json]
        with patch("pipeline.learner._call_llm", side_effect=responses):
            with patch("pipeline.learner.STYLE_GUIDE_PATH", tmp_path / "g.json"):
                with patch("pipeline.learner.OUTPUT_DIR", tmp_path):
                    result = build_style_guide([SAMPLE_PAIR, memo_pair])

        types = set(result["patterns_by_draft_type"].keys())
        assert "title_review_summary" in types
        assert "case_status_memo" in types


class TestFormatStyleGuideForPrompt:

    @pytest.fixture
    def sample_guide(self):
        return {
            "patterns_by_draft_type": {
                "title_review_summary": [
                    {"pattern_name": "inst", "instruction": "include instrument numbers"},
                ],
                "case_status_memo": [
                    {"pattern_name": "pri", "instruction": "prioritize action items"},
                ],
            }
        }

    def test_empty_guide_returns_empty_string(self):
        assert format_style_guide_for_prompt({}) == ""

    def test_specific_draft_type_only_returns_its_patterns(self, sample_guide):
        result = format_style_guide_for_prompt(sample_guide, draft_type="title_review_summary")
        assert "include instrument numbers" in result
        assert "prioritize action items" not in result

    def test_no_draft_type_returns_all_patterns(self, sample_guide):
        result = format_style_guide_for_prompt(sample_guide)
        assert "include instrument numbers" in result
        assert "prioritize action items" in result


class TestCompareDrafts:

    def test_calls_llm_with_both_texts(self):
        captured = []

        def capture(prompt):
            captured.append(prompt)
            return "The improved version is better."

        with patch("pipeline.learner._call_llm", side_effect=capture):
            compare_drafts("baseline text", "improved text")

        assert "baseline text" in captured[0]
        assert "improved text" in captured[0]

    def test_returns_llm_response(self):
        expected = "The improved version is clearly more actionable."
        with patch("pipeline.learner._call_llm", return_value=expected):
            result = compare_drafts("a", "b")
        assert result == expected
