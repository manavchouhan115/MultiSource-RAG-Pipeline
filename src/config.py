"""
config.py
---------
Central configuration for the hybrid-RAG pipeline.

Fully free stack:
  Embeddings : HuggingFace sentence-transformers (runs locally, no API key)
  LLM        : Groq inference API (free tier – get key at console.groq.com)

All modules should import constants from here rather than hard-coding
paths, model names, or chunk sizes inline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Resolve project root (two levels up from src/config.py) ──────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load .env from the project root; raises no error if the file is missing
load_dotenv(ROOT_DIR / ".env")

# ── API credentials ───────────────────────────────────────────────────────────
# Groq is used for LLM chat/completion only (NOT embeddings).
# Get a free key at https://console.groq.com
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    import warnings
    warnings.warn(
        "GROQ_API_KEY is not set.  Copy .env.example to .env and add your key.",
        stacklevel=2,
    )

# ── Storage paths ─────────────────────────────────────────────────────────────
# ChromaDB persisted vector store
CHROMA_PATH: Path = ROOT_DIR / "data" / "chroma"

# SQLite database used for structured (SQL) retrieval
DB_PATH: Path = ROOT_DIR / "data" / "db" / "store.db"

# Directory where source PDFs are placed for ingestion
PDF_DIR: Path = ROOT_DIR / "data" / "pdfs"

# ── Model names ───────────────────────────────────────────────────────────────
# HuggingFace sentence-transformer — downloaded once, runs fully locally.
# bge-small-en-v1.5 is compact (~90 MB) and ranks top on MTEB benchmarks.
EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"

# Groq-hosted Llama 3.3 70B — fast inference, generous free tier.
LLM_MODEL: str = "llama-3.3-70b-versatile"

# ── Chunking parameters ───────────────────────────────────────────────────────
CHUNK_SIZE: int = 500      # Target tokens per chunk
CHUNK_OVERLAP: int = 50    # Token overlap between consecutive chunks
