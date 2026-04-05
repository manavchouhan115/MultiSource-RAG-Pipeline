"""
app.py
------
Streamlit entry point for the hybrid-RAG application.

Architecture
------------
 User query
     │
     ▼
 Router     ──► "structured" ──► SQL Retriever  ──► SQLite DB
     │                                │
     └──► "unstructured" ─► Vector Retriever ──► ChromaDB
                                      │
                                      ▼
                               Synthesiser  ──► GPT-4o-mini
                                      │
                                      ▼
                               Answer displayed here

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Hybrid RAG Pipeline")
st.caption(
    "Ask anything — the router decides whether to query the "
    "vector store (PDF knowledge) or the SQL database (structured data)."
)

# ── Sidebar – status indicators ───────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Status")
    from config import CHROMA_PATH, DB_PATH, LLM_MODEL, EMBED_MODEL

    st.markdown(f"**LLM:** `{LLM_MODEL}`")
    st.markdown(f"**Embeddings:** `{EMBED_MODEL}`")
    st.markdown(
        f"**ChromaDB:** {'✅ exists' if CHROMA_PATH.exists() else '⚠️ not yet created'}"
    )
    st.markdown(
        f"**SQLite DB:** {'✅ exists' if DB_PATH.exists() else '⚠️ run seed_database.py'}"
    )

# ── Main chat interface ───────────────────────────────────────────────────────
if "messages" in st.session_state:
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
else:
    st.session_state["messages"] = []

query = st.chat_input("Ask a question…")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            # ── Lazy imports so missing dependencies show a clear error ───────
            try:
                from router.classifier   import classify_query
                from retrieval.vector    import retrieve_vector
                from retrieval.sql       import retrieve_sql
                from synthesis.generator import generate_answer

                route  = classify_query(query)
                st.caption(f"🔀 Route: **{route}**")

                if route == "structured":
                    context = retrieve_sql(query)
                else:
                    context = retrieve_vector(query)

                answer = generate_answer(query, context, route=route)

            except ImportError as exc:
                answer = (
                    f"⚠️ Pipeline modules not yet fully implemented.\n\n"
                    f"Error: `{exc}`\n\n"
                    "Build out the modules under `src/` to enable end-to-end queries."
                )

        st.write(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
