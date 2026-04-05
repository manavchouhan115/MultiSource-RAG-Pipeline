"""
src/synthesis/__init__.py
-------------------------
Answer generation / synthesis layer.

Takes the retrieved context (either SQL result rows or vector-matched
PDF chunks) together with the original query and produces a coherent
natural-language answer using the configured LLM.

The synthesiser is route-aware: it uses a different system prompt
depending on whether the context came from SQL or vector retrieval,
so the model knows the nature of the data it is working with.
"""
