"""
app.py
------
Streamlit frontend for Hybrid Knowledge Assistant
"""

import sys
import sqlite3
import pandas as pd
from pathlib import Path
import streamlit as st

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import PDF_DIR, DB_PATH
from synthesis.synthesizer import answer_question

# 1. PAGE SETUP
st.set_page_config(
    page_title="Hybrid Knowledge Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "example_question" not in st.session_state:
    st.session_state["example_question"] = None

def clear_chat():
    st.session_state["messages"] = []

def handle_example_click(q):
    st.session_state["example_question"] = q

with st.sidebar:
    st.title("🧠 Hybrid Knowledge Assistant")
    st.caption("Ask questions answered from PDF documents or live database — the system decides where to look.")
    
    st.divider()
    
    st.subheader("📄 Loaded PDFs")
    if PDF_DIR.exists():
        pdfs = [f.name for f in PDF_DIR.glob("*.pdf")]
        if pdfs:
            for pdf in pdfs:
                st.markdown(f"- {pdf}")
        else:
            st.write("No PDFs found.")
    else:
        st.write("PDF directory not found.")
        
    st.divider()
    
    st.subheader("🗄️ Database Tables")
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            for table in ["products", "orders", "customers"]:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    st.markdown(f"- **{table}**: {count} rows")
                except sqlite3.Error:
                    st.markdown(f"- **{table}**: Error fetching rows")
            conn.close()
        except sqlite3.Error as e:
            st.error(f"Failed to connect to database: {e}")
    else:
        st.write("Database not found.")
        
    st.divider()
    
    st.button("🗑️ Clear Chat", on_click=clear_chat, use_container_width=True)
    
    st.divider()
    
    st.subheader("💡 Example questions")
    questions = [
        "What is the sick leave policy?",
        "Top 5 products by revenue?",
        "What GPIO pins does Arduino UNO have?",
        "Which customer spent the most?",
        "How does Apple describe their services growth?",
        "Compare our top sales region to the business targets"
    ]
    for q in questions:
        st.button(q, on_click=handle_example_click, args=(q,), use_container_width=True)

# 4. RESULT PANEL
def render_result(result: dict, msg_idx: int):
    route = result.get("route", "")
    
    # a. Route badge
    if route == "RAG":
        st.markdown('<span style="background-color:#E3F2FD; color:#0D47A1; padding:4px 8px; border-radius:4px; font-weight:bold;">📄 Document Search</span>', unsafe_allow_html=True)
    elif route == "SQL":
        st.markdown('<span style="background-color:#E8F5E9; color:#1B5E20; padding:4px 8px; border-radius:4px; font-weight:bold;">🗄️ Database Query</span>', unsafe_allow_html=True)
    elif route == "BOTH":
        st.markdown('<span style="background-color:#F3E5F5; color:#4A148C; padding:4px 8px; border-radius:4px; font-weight:bold;">🔀 Hybrid Search</span>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # b. Answer
    st.markdown(result.get("answer", "No answer provided."))
    
    # c. EXPANDER: View Sources
    is_rag = route in ["RAG", "BOTH"]
    if is_rag and "chunks" in result and result["chunks"]:
        with st.expander("📄 View Sources"):
            for i, chunk in enumerate(result["chunks"]):
                st.subheader(f"{chunk.get('source', 'Unknown')} - Page {chunk.get('page', 'Unknown')}")
                st.text_area(label="chunk", label_visibility="collapsed", value=chunk.get("text", ""), height=100, disabled=True, key=f"source_msg_{msg_idx}_chunk_{i}")
                
    # d. EXPANDER: View SQL Query
    is_sql = route in ["SQL", "BOTH"]
    if is_sql and "sql_data" in result and result["sql_data"]:
        sql_data = result["sql_data"]
        with st.expander("🗄️ View SQL Query"):
            st.code(sql_data.get("sql", ""), language="sql")
            
            columns = sql_data.get("columns", [])
            rows = sql_data.get("rows", [])
            
            if columns and rows:
                df = pd.DataFrame(rows, columns=columns)
                st.dataframe(df, use_container_width=True)
            elif "error" in sql_data:
                st.error(f"SQL Error: {sql_data['error']}")
            else:
                st.write("No rows returned from query.")
                
    # e. METRICS ROW
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Route", route)
    col2.metric("Chunks used", result.get("chunks_used", 0))
    sources_list = result.get("sources", [])
    sources_str = ", ".join(sources_list) if sources_list else "None"
    col3.metric("Sources", sources_str)


# 2. CHAT HISTORY
for idx, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["question"])
        else:
            if msg.get("result"):
                render_result(msg["result"], idx)
            elif msg.get("error"):
                st.error(msg["error"])

# 3. CHAT INPUT
query = st.chat_input("Ask anything about your documents or data... ")
if st.session_state["example_question"]:
    query = st.session_state["example_question"]
    st.session_state["example_question"] = None

if query:
    # Immediately render the user message
    with st.chat_message("user"):
        st.write(query)
    
    st.session_state["messages"].append({
        "role": "user",
        "question": query,
        "result": None
    })
    
    # Render Assistant Loading state
    with st.chat_message("assistant"):
        with st.spinner("Thinking... routing your question"):
            # 6. ERROR HANDLING
            try:
                result = answer_question(query)
                msg_idx = len(st.session_state["messages"])
                render_result(result, msg_idx)
                
                st.session_state["messages"].append({
                    "role": "assistant",
                    "question": query,
                    "result": result
                })
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.session_state["messages"].append({
                    "role": "assistant",
                    "question": query,
                    "result": None,
                    "error": str(e)
                })
