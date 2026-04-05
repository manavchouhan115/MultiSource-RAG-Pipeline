"""
src/router/__init__.py
----------------------
Query classification / routing layer.

The router reads the incoming natural-language question and decides
which retrieval path to invoke:

  "structured"   → the question is best answered from tabular/SQL data
                   (e.g. "How many orders were placed in Q1?")

  "unstructured" → the question is best answered from PDFs / free text
                   (e.g. "Summarise the methodology section of the report")

Classification can be implemented as:
  - A zero-shot LLM call with a brief system prompt
  - A lightweight keyword / intent heuristic (faster, cheaper)
  - A fine-tuned classifier (most accurate)
"""
