"""
Learner — Part 4 of the pipeline.

Responsibilities:
  - Load operator edit pairs from sample_edits.json
  - Use the LLM to analyze what changed between system and operator versions
  - Distill those changes into a reusable "style guide" (a named set of patterns)
  - Apply the style guide when regenerating drafts to demonstrate improvement

The improvement loop:
  1. Generate a baseline draft (no style guide)
  2. Operator reviews and corrects it
  3. LLM analyzes the diff and extracts learnable patterns
  4. Patterns are stored as a JSON style guide on disk
  5. Next generation call injects the guide into the prompt
  6. The result is closer to what the operator expects

The style guide accumulates with each new edit pair. Running the learner
again after adding more edits to sample_edits.json updates the guide.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import google.generativeai as genai

from pipeline.config import EDITS_FILE, GEMINI_API_KEY, GEMINI_MODEL, OUTPUT_DIR

logger = logging.getLogger(__name__)

# configure once at module level
genai.configure(api_key=GEMINI_API_KEY)

STYLE_GUIDE_PATH = OUTPUT_DIR / "learned_style_guide.json"


def _call_llm(prompt: str) -> str:
    """Call Gemini and return response text. Returns empty string on failure."""
    model = genai.GenerativeModel(GEMINI_MODEL)
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return ""


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that the model sometimes wraps JSON in."""
    text = re.sub(r'^```[a-z]*\n?', '', text.strip())
    text = re.sub(r'\n?```$', '', text)
    return text


def load_edit_pairs() -> list[dict]:
    """Load operator edit pairs from disk."""
    with open(EDITS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_edit_pair(edit_pair: dict) -> dict:
    """
    Ask the LLM to compare a system draft to the operator-edited version
    and extract concrete, reusable writing patterns.

    The key_edits field provides operator-written ground truth explanations.
    We give those to the LLM as well so it has both the raw text diff and
    the explicit intent behind each change.
    """
    draft_type = edit_pair["draft_type"]
    system_draft = edit_pair["system_draft"]
    operator_version = edit_pair["operator_edited_version"]
    key_edits = edit_pair.get("key_edits", [])

    prompt = (
        f'You are a quality improvement analyst for a legal document AI system.\n\n'
        f'A system generated a "{draft_type}". An operator improved it. '
        f'Extract 5-10 concrete, actionable writing patterns from this edit that '
        f'should be applied to ALL future drafts of this type.\n\n'
        f'SYSTEM DRAFT:\n---\n{system_draft}\n---\n\n'
        f'OPERATOR-EDITED VERSION:\n---\n{operator_version}\n---\n\n'
        f'OPERATOR\'S OWN EXPLANATION OF CHANGES:\n{json.dumps(key_edits, indent=2)}\n\n'
        f'Return ONLY a JSON object with this structure — no explanation:\n'
        f'{{\n'
        f'  "draft_type": "{draft_type}",\n'
        f'  "patterns": [\n'
        f'    {{"pattern_name": "short name", "instruction": "specific actionable instruction"}}\n'
        f'  ]\n'
        f'}}\n'
    )

    raw = _call_llm(prompt)
    raw = _strip_code_fences(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse pattern JSON for %s — falling back to key_edits", draft_type)
        # use the operator's own explanations as a fallback so we don't lose the data
        return {
            "draft_type": draft_type,
            "patterns": [
                {"pattern_name": f"edit_{i}", "instruction": edit}
                for i, edit in enumerate(key_edits)
            ],
        }


def build_style_guide(edit_pairs: list[dict]) -> dict:
    """
    Analyze all edit pairs and build a combined style guide.
    The guide is saved to disk so it persists across runs and accumulates
    improvements as more edit pairs are added.
    """
    patterns_by_type: dict[str, list[dict]] = {}

    for pair in edit_pairs:
        logger.info("Analyzing edit pair for %s ...", pair["draft_type"])
        analysis = analyze_edit_pair(pair)
        draft_type = analysis["draft_type"]
        patterns_by_type[draft_type] = analysis.get("patterns", [])

    style_guide = {
        "version": 1,
        "description": "Writing patterns learned from operator edit pairs",
        "patterns_by_draft_type": patterns_by_type,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STYLE_GUIDE_PATH, "w", encoding="utf-8") as f:
        json.dump(style_guide, f, indent=2)

    logger.info("Style guide saved to %s", STYLE_GUIDE_PATH)
    return style_guide


def load_style_guide() -> dict:
    """Load a previously built style guide from disk. Returns empty dict if none exists."""
    if STYLE_GUIDE_PATH.exists():
        with open(STYLE_GUIDE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def format_style_guide_for_prompt(
    style_guide: dict,
    draft_type: str | None = None,
) -> str:
    """
    Flatten the style guide into a plain-text instruction block for prompt injection.
    If draft_type is given, only that type's patterns are included.
    Otherwise all patterns are included (useful for new draft types not in the guide).
    """
    if not style_guide:
        return ""

    patterns_by_type: dict[str, list[dict]] = style_guide.get("patterns_by_draft_type", {})
    lines: list[str] = []

    if draft_type and draft_type in patterns_by_type:
        lines.append(f"Patterns for {draft_type}:")
        for p in patterns_by_type[draft_type]:
            lines.append(f"  - {p['instruction']}")
    else:
        for dt, patterns in patterns_by_type.items():
            lines.append(f"Patterns for {dt}:")
            for p in patterns:
                lines.append(f"  - {p['instruction']}")
            lines.append("")

    return "\n".join(lines)


def compare_drafts(baseline: str, improved: str) -> str:
    """
    Ask the LLM to compare a baseline and improved draft and return a
    plain-English evaluation of what got better and what gaps remain.
    """
    prompt = (
        "Compare these two versions of a legal document draft.\n"
        "Explain concretely what improved and what is still missing.\n\n"
        f"BASELINE DRAFT:\n---\n{baseline}\n---\n\n"
        f"IMPROVED DRAFT (generated after applying the learned style guide):\n---\n{improved}\n---\n\n"
        "Write 3-5 plain-English sentences covering:\n"
        "1. What specific content was added or reorganized\n"
        "2. Whether the improved draft is more actionable\n"
        "3. Whether grounding and source citations improved\n"
        "4. Any remaining gaps\n"
    )
    return _call_llm(prompt)
