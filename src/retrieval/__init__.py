"""
src/retrieval/__init__.py
-------------------------
Retrieval layer – two retrieval strategies:

vector (unstructured)
    Similarity search over ChromaDB embeddings built from PDF chunks.

sql (structured)
    Text-to-SQL translation executed against the SQLite database.

Each strategy returns a plain string "context" that the synthesiser
can insert into the prompt directly.
"""
