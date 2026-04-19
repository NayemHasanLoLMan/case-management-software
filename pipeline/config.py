"""
Configuration — API keys, model names, paths.
Load GEMINI_API_KEY from a .env file or set it in the environment before running.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# core paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "ai_engineer_assignment_data"
DOCS_DIR = DATA_DIR / "sample_documents"
EDITS_FILE = DATA_DIR / "sample_edits.json"
CASE_CONTEXT_FILE = DATA_DIR / "case_context.json"
OUTPUT_DIR = BASE_DIR / "sample_outputs"

# LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# use full model path — required by some API key tiers
GEMINI_MODEL = "models/gemini-2.5-flash"

# retrieval
CHUNK_SIZE = 300           # approximate word target per chunk
CHUNK_OVERLAP = 50         # words shared between adjacent chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # local, no cost, no data leaves machine
TOP_K = 5                  # default number of chunks to retrieve per query
