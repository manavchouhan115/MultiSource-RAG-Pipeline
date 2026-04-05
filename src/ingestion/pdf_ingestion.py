"""
src/ingestion/pdf_ingestion.py
------------------------------
Full PDF → ChromaDB ingestion pipeline.

Pipeline stages
---------------
  1. LOAD      – Walk data/pdfs/, load every PDF via PyPDFLoader.
  2. CHUNK     – Split pages into token-aware chunks using tiktoken
                 (cl100k_base) so chunk_size is measured in TOKENS,
                 not characters.  Each chunk carries source / page /
                 chunk_index metadata.
  3. EMBED     – Generate embeddings locally via HuggingFace sentence-transformers
                 (BAAI/bge-small-en-v1.5, ~90 MB, downloaded once then cached).
  4. STORE     – Persist to a ChromaDB collection called "knowledge_base"
                 at CHROMA_PATH.  The collection is wiped and recreated on
                 every run so re-ingestion is always idempotent.
  5. VERIFY    – Print per-source chunk counts, then run a smoke-test
                 similarity search and display the top-3 results.

Usage
-----
  # From project root:
  python src/ingestion/pdf_ingestion.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import sys
import os
from pathlib import Path
from collections import Counter

# ── Make src/ importable when the script is run directly ─────────────────────
# __file__ is  …/src/ingestion/pdf_ingestion.py
# parent.parent is …/src/
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Project config (loads .env automatically) ─────────────────────────────────
from config import (
    CHROMA_PATH,
    PDF_DIR,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ── Third-party ───────────────────────────────────────────────────────────────
import tiktoken
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – TIKTOKEN LENGTH FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
# RecursiveCharacterTextSplitter accepts a custom length_function so we can
# measure chunk size in TOKENS instead of characters.  This is model-agnostic
# and gives far more predictable chunk sizes than character counting.
# "cl100k_base" is a widely-used general-purpose BPE tokenizer.

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text: str) -> int:
    """Return the number of cl100k_base tokens in *text*."""
    return len(_TOKENIZER.encode(text, disallowed_special=()))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – PDF LOADING
# ══════════════════════════════════════════════════════════════════════════════
# PyPDFLoader maps each PDF page to one LangChain Document object.
# document.metadata["source"] already contains the file path; we also store
# just the filename so metadata stays readable.

def load_pdfs(pdf_dir: Path) -> list:
    """
    Load every *.pdf in *pdf_dir*.

    Returns a flat list of LangChain Document objects (one per page).
    Prints filename + page count for each file as it loads.
    """
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"[WARN] No PDFs found in {pdf_dir}.  Drop files there and re-run.")
        return []

    all_docs = []
    print(f"\n{'─'*55}")
    print(f"  STAGE 1 – Loading PDFs from: {pdf_dir}")
    print(f"{'─'*55}")

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs   = loader.load()          # one Document per page

        # Normalise the 'source' metadata to just the filename
        filename = pdf_path.name
        for doc in docs:
            doc.metadata["source"] = filename

        print(f"  [OK] {filename:<30}  {len(docs):>3} page(s)")
        all_docs.extend(docs)

    print(f"\n  Total pages loaded: {len(all_docs)}\n")
    return all_docs


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – TEXT CHUNKING
# ══════════════════════════════════════════════════════════════════════════════
# RecursiveCharacterTextSplitter tries to split on paragraph boundaries first,
# then sentences, then words — preserving semantic coherence where possible.
# We override length_function=tiktoken_len so chunk_size=500 means 500 TOKENS.
#
# After splitting we stamp three extra metadata fields onto every chunk:
#   source      – filename (already set by the loader, kept for clarity)
#   page        – 0-based page number from the original PDF
#   chunk_index – global position in the final list of all chunks

def chunk_documents(docs: list) -> list:
    """
    Split page-level Documents into token-aware chunks.

    Returns a list of Document chunks with enriched metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],   # try paragraph → line → word
    )

    print(f"{'─'*55}")
    print(f"  STAGE 2 – Chunking  (size={CHUNK_SIZE} tok, overlap={CHUNK_OVERLAP} tok)")
    print(f"{'─'*55}")

    chunks = splitter.split_documents(docs)

    # Stamp chunk_index onto each chunk
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        # page may already exist from PyPDFLoader; ensure it's always present
        chunk.metadata.setdefault("page", 0)

    print(f"  Total chunks produced: {len(chunks)}\n")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – EMBEDDING + CHROMADB STORAGE
# ══════════════════════════════════════════════════════════════════════════════
# We use the low-level chromadb client (not the LangChain wrapper) so we have
# fine-grained control over collection management (easy delete/recreate).
#
# HuggingFaceEmbeddings runs entirely on CPU/GPU locally — no rate limits.
# We still batch for memory efficiency during encode().
# Each document is stored with:
#   - a deterministic string ID  ("chunk-<index>")
#   - its embedding vector
#   - its text content
#   - its metadata dict

COLLECTION_NAME = "knowledge_base"
EMBED_BATCH_SIZE = 128          # efficient batch size for local inference


def embed_and_store(chunks: list, chroma_path: Path) -> chromadb.Collection:
    """
    Embed *chunks* locally with HuggingFace and persist them to ChromaDB.

    The collection is deleted and recreated on every call so the pipeline
    is always idempotent (safe to re-run after adding new PDFs).

    Returns the populated ChromaDB Collection object.
    """
    print(f"{'─'*55}")
    print(f"  STAGE 3 – Embedding + storing in ChromaDB")
    print(f"  Embedder   : {EMBED_MODEL}  (local, no API key)")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Persist dir: {chroma_path}")
    print(f"{'─'*55}")

    # ── Initialise persistent ChromaDB client ─────────────────────────────
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    # ── Delete existing collection so re-runs start clean ─────────────────
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"  [RESET] Deleted existing '{COLLECTION_NAME}' collection.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine distance for similarity
    )

    # ── HuggingFace local embedder ────────────────────────────────────────
    # First call downloads the model (~90 MB) into ~/.cache/huggingface/.
    # encode_kwargs normalises vectors to unit length for accurate cosine sim.
    print(f"  Loading embedder (downloading model on first run…)")
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Batch-embed and upsert ────────────────────────────────────────────
    total = len(chunks)
    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]

        texts     = [c.page_content for c in batch]
        metadatas = [c.metadata     for c in batch]
        ids       = [f"chunk-{batch_start + i}" for i in range(len(batch))]

        # embed_documents returns List[List[float]]
        vectors = embedder.embed_documents(texts)

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

        end = min(batch_start + EMBED_BATCH_SIZE, total)
        print(f"  [OK] Embedded chunks {batch_start+1:>4} - {end:>4} / {total}")

    print(f"\n  Collection now holds {collection.count()} vectors.\n")
    return collection


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
# Two checks:
#   A) Per-source chunk count – confirms every PDF contributed content.
#   B) Smoke-test query – embeds a test string and retrieves the top-3
#      nearest neighbours to prove the index is queryable end-to-end.

def verify(chunks: list, collection: chromadb.Collection) -> None:
    """Print ingestion stats and run a smoke-test similarity search."""

    # ── A) Per-source chunk counts ────────────────────────────────────────
    print(f"{'─'*55}")
    print(f"  STAGE 5 – Verification")
    print(f"{'─'*55}")
    print("\n  Chunks per source PDF:")

    source_counts = Counter(c.metadata.get("source", "unknown") for c in chunks)
    for source, count in sorted(source_counts.items()):
        print(f"    {source:<35} {count:>4} chunks")

    # ── B) Smoke-test similarity search ───────────────────────────────────
    TEST_QUERY = "what is the return policy"
    TOP_K      = 3

    print(f"\n  Smoke-test query: \"{TEST_QUERY}\"")
    print(f"  Retrieving top {TOP_K} results …\n")

    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    query_vector = embedder.embed_query(TEST_QUERY)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score (0–1) for readability
        similarity = round(1 - dist / 2, 4)

        print(f"  {'─'*51}")
        print(f"  Result #{rank}")
        print(f"    Source    : {meta.get('source', 'N/A')}")
        print(f"    Page      : {meta.get('page', 'N/A')}")
        print(f"    Similarity: {similarity}")
        print(f"    Excerpt   : {doc[:200].strip()} …")

    print(f"\n  {'─'*51}")
    print("  [DONE] Ingestion and verification complete!\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def ingest_pdfs() -> chromadb.Collection:
    """
    Run the full ingestion pipeline and return the populated collection.

    Can be called programmatically by other modules (e.g. app.py) as well
    as executed directly from the command line.
    """
    docs       = load_pdfs(PDF_DIR)
    if not docs:
        return None

    chunks     = chunk_documents(docs)
    collection = embed_and_store(chunks, CHROMA_PATH)
    verify(chunks, collection)
    return collection


if __name__ == "__main__":
    ingest_pdfs()
