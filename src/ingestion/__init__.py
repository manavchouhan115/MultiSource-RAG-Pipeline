"""
src/ingestion/__init__.py
-------------------------
PDF loading and chunking pipeline.

Responsibilities
----------------
- Walk data/pdfs/ and load every PDF with PyPDFLoader
- Split documents into fixed-size token chunks (RecursiveCharacterTextSplitter)
- Embed chunks with the configured OpenAI embedding model
- Persist the resulting vector store to CHROMA_PATH

Typical usage (run once, or re-run to refresh):
    from ingestion.loader import ingest_pdfs
    ingest_pdfs()
"""
