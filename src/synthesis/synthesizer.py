"""
src/synthesis/synthesizer.py
----------------------------
Answer synthesizer that merges RAG chunks + SQL results into a single cited answer.
"""

import sys
import io
import concurrent.futures
from pathlib import Path

# Add src to sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import CHROMA_PATH, EMBED_MODEL, LLM_MODEL, GROQ_API_KEY
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from router.query_router import route_question, Route
from retrieval.sql_retriever import run_sql_pipeline

def retrieve_chunks(question: str, n_results: int = 4) -> list[dict]:
    """
    Retrieve top semantic chunks from ChromaDB.
    Uses HuggingFace embeddings as configured in the free tech stack (EMBED_MODEL).
    Filters out chunks with a cosine distance > 0.8.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection("knowledge_base")
    
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    query_vector = embedder.embed_query(question)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = []
    if not results["documents"] or not results["documents"][0]:
        return chunks
        
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if dist > 0.8:
            continue
            
        chunks.append({
            "text": doc,
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", 0),
            "score": dist
        })
        
    return chunks

def format_rag_context(chunks: list[dict]) -> str:
    """Format the retrieved chunks into a clean string for the LLM prompt."""
    if not chunks:
        return "No document context available."
        
    formatted = []
    for chunk in chunks:
        formatted.append(f"[Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}")
        
    return "\n\n".join(formatted)

def format_sql_context(sql_result: dict) -> str:
    """Format the SQL result dict into a readable table-like string."""
    if not sql_result or "error" in sql_result:
        return f"Database query failed or unavailable. Error: {sql_result.get('error', 'None')}"
        
    sql = sql_result.get("sql", "Unknown query")
    columns = sql_result.get("columns", [])
    rows = sql_result.get("rows", [])
    row_count = sql_result.get("row_count", 0)
    
    context = []
    context.append(f"SQL Query: {sql}")
    context.append(f"Results ({row_count} rows):")
    
    if not columns or not rows:
        context.append("No results found.")
        return "\n".join(context)
        
    col_widths = [
        max(len(str(col)), max((len(str(row[i])) for row in rows), default=0))
        for i, col in enumerate(columns)
    ]
    
    header_str = " | ".join(str(col).ljust(col_widths[i]) for i, col in enumerate(columns))
    separator_str = "-+-".join("-" * w for w in col_widths)
    
    context.append(f"| {header_str} |")
    context.append(f"| {separator_str} |")
    
    for row in rows:
        row_str = " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        context.append(f"| {row_str} |")
        
    return "\n".join(context)

def generate_answer(
    question: str,
    rag_chunks: list[dict],
    sql_result: dict | None,
    route: Route
) -> dict:
    """Generates an answer using ChatGroq based on the provided route and context."""
    context_str = ""
    if route == Route.RAG:
        context_str = format_rag_context(rag_chunks)
    elif route == Route.SQL:
        context_str = format_sql_context(sql_result)
    elif route == Route.BOTH:
        context_str = (
            "DOCUMENT CONTEXT:\n"
            f"{format_rag_context(rag_chunks)}\n\n"
            "DATABASE CONTEXT:\n"
            f"{format_sql_context(sql_result)}"
        )
        
    system_prompt = (
        "Answer the question using ONLY the provided context.\n"
        "Always cite your sources:\n"
        "- For document info: mention the filename and page number\n"
        "- For SQL data: mention it comes from the live database\n"
        "If the context does not contain enough information to answer, "
        "say so clearly rather than making up an answer."
    )
    
    sources = set()
    for chunk in rag_chunks:
        sources.add(chunk['source'])
    sources = list(sources)
    
    sql_used = sql_result.get("sql") if sql_result else None
    
    # Do not count chunks that were not passed into context
    # We pass all retrieved chunks in this case.
    chunks_used = len(rag_chunks)
    
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.0
    )
    
    human_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    
    return {
        "answer": response.content.strip(),
        "sources": sources,
        "sql_used": sql_used,
        "route": route.value if hasattr(route, 'value') else str(route),
        "chunks_used": chunks_used,
        "chunks": rag_chunks,
        "sql_data": sql_result
    }

def answer_question(question: str) -> dict:
    """Main pipeline function that routes, retrieves, and synthesizes in parallel where applicable."""
    route = route_question(question)
    
    rag_chunks = []
    sql_result = None
    
    if route == Route.RAG:
        rag_chunks = retrieve_chunks(question)
    elif route == Route.SQL:
        sql_result = run_sql_pipeline(question)
    elif route == Route.BOTH:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_rag = executor.submit(retrieve_chunks, question)
            future_sql = executor.submit(run_sql_pipeline, question)
            rag_chunks = future_rag.result()
            sql_result = future_sql.result()
            
    return generate_answer(question, rag_chunks, sql_result, route)

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    
    test_questions = [
        "What is the policy for taking sick leave?",
        "Which product category generates the most revenue?",
        "How does the Apple report describe their services growth, and does our orders data reflect similar trends?",
        "What GPIO pins does the Arduino UNO have?"
    ]
    
    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"QUESTION: {q}")
        print(f"{'='*70}")
        
        result = answer_question(q)
        
        print(f"\nRoute Taken : {result['route']}")
        print(f"Sources     : {result['sources']}")
        print(f"SQL Used    : {result['sql_used']}")
        print(f"Chunks Used : {result['chunks_used']}")
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\n{'='*70}\n")
