"""
src/router/query_router.py
--------------------------
Query router for the hybrid-RAG pipeline.
Decides whether a question should go to RAG, SQL, or BOTH.
"""

import sys
import io
from enum import Enum
from pathlib import Path

# ── Make src/ importable when this script is run directly ────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Project config ────────────────────────────────────────────────────────────
from config import LLM_MODEL, GROQ_API_KEY

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ROUTE ENUM
# ══════════════════════════════════════════════════════════════════════════════
# Defines the three valid routing paths
class Route(str, Enum):
    RAG = "RAG"     # For questions about documents, policies, manuals
    SQL = "SQL"     # For questions requiring data aggregation or numbers from the DB
    BOTH = "BOTH"   # For questions needing information from both sources


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: RULE-BASED ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def rule_based_route(question: str) -> Route | None:
    """
    Fast keyword-based router. Doesn't use an LLM API call.
    Checks for explicit signal words to immediately assign a route.
    """
    q_lower = question.lower()

    # Define keyword lists
    sql_keywords = [
        "how many", "total", "sum", "average", "highest", "lowest", 
        "revenue", "sales", "count", "which customer", "top 5", 
        "top 10", "per region", "ranking"
    ]
    rag_keywords = [
        "what is", "how do i", "explain", "policy", "manual", 
        "procedure", "guideline", "according to", "document", 
        "what does", "instructions", "how to"
    ]
    both_keywords = [
        "compare", "versus", "vs", "relate", "based on the report", 
        "according to the data", "does the data match"
    ]

    has_both = any(kw in q_lower for kw in both_keywords)
    has_sql = any(kw in q_lower for kw in sql_keywords)
    has_rag = any(kw in q_lower for kw in rag_keywords)

    if has_both:
        print("  [Router] Rule-based match (BOTH) on keywords.")
        return Route.BOTH

    # Fix: if it mentions both data and documents/reports, it goes to BOTH
    if has_sql and (has_rag or "report" in q_lower):
        print("  [Router] Rule-based match (BOTH) due to mixed SQL/Document signals.")
        return Route.BOTH

    if has_sql:
        print("  [Router] Rule-based match (SQL) on keywords.")
        return Route.SQL

    if has_rag:
        print("  [Router] Rule-based match (RAG) on keywords.")
        return Route.RAG

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LLM-BASED ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def llm_based_route(question: str) -> Route:
    """
    Fallback router called only when rule-based routing is ambiguous.
    Uses the LLM to classify the query.
    """
    system_prompt = (
        "You are a routing assistant for a hybrid knowledge system that has:\n"
        "1. A VECTOR DATABASE with PDF documents: an HR policy manual, "
        "an Arduino product manual, and an Apple annual report.\n"
        "2. A SQL DATABASE with tables: products, orders, customers.\n\n"
        "Classify the user question into exactly one of these routes:\n"
        "- RAG: question is about policies, procedures, manuals, or report content\n"
        "- SQL: question requires querying structured sales/customer/product data\n"
        "- BOTH: question needs both document content AND database data\n\n"
        "Respond with ONLY one word: RAG, SQL, or BOTH."
    )

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,  # Deterministic response is crucial here
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ])

    # Parse and clean response
    answer = response.content.strip().upper()
    
    # Strip any potential punctuation attached by mistake
    answer = ''.join(c for c in answer if c.isalpha())

    print(f"  [Router] LLM-based logic selected: {answer}")

    if answer == "SQL":
        return Route.SQL
    elif answer == "BOTH":
        return Route.BOTH
    elif answer == "RAG":
        return Route.RAG
    else:
        # Default to RAG if the LLM hallucinated an invalid response
        print("  [Router] Warning! Unrecognized LLM response, defaulting to RAG.")
        return Route.RAG


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MAIN ROUTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def route_question(question: str) -> Route:
    """
    The main routing pipeline. Tries the fast rule-based lookup,
    and falls back to LLM processing if the question is ambiguous.
    """
    print(f"\n{'-'*65}")
    print(f"Routing Question: {question}")
    
    route = rule_based_route(question)
    
    if route is None:
        print("  [Router] Rule-based engine found no signal. Falling back to LLM...")
        route = llm_based_route(question)
        
    print(f"Final Decision:   {route.value}")
    print(f"{'-'*65}")
    return route


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ROUTER METADATA
# ══════════════════════════════════════════════════════════════════════════════
def get_route_explanation(question: str, route: Route) -> str:
    """
    Generates a human-readable string explaining WHY the system chose this route.
    Uses the LLM to write a single-sentence justification.
    """
    system_prompt = (
        "You are an assistant explaining system behavior to an end-user.\n"
        "A hybrid-RAG system routed the user's question to the '{route}' backend.\n"
        "Write a single sentence explaining why this route was chosen.\n"
        "For context:\n"
        "- RAG backend holds text documents, manuals, and policies.\n"
        "- SQL backend holds tabular data regarding orders, products, and customers.\n"
        "- BOTH backend holds both formats and implies a combination of information.\n"
        "Return ONLY the explanation sentence. Example: Routed to SQL because the "
        "question asks for aggregated sales data across regions, which requires "
        "querying the orders table."
    ).replace("{route}", route.value)

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {question}"),
    ])

    return response.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TESTING
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Ensure stdout handles Windows UTF-8 cleanly
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    
    TEST_QUESTIONS = [
        "What is the leave policy for sick days?",
        "What are total sales by product category?",
        "How do the reported iPhone revenues in the annual report compare to our top product sales?",
        "How do I reset the Arduino to factory settings?",
        "Which region has the highest average order value?",
        "Does our sales data support the growth targets mentioned in the Apple report?"
    ]

    print("\n" + "=" * 65)
    print("  QUERY ROUTER -- Test Run")
    print("=" * 65)

    for q in TEST_QUESTIONS:
        selected_route = route_question(q)
        explanation = get_route_explanation(q, selected_route)
        print(f"Explanation:      {explanation}")

    print("\n" + "=" * 65)
    print("  [DONE] All test queries complete.")
    print("=" * 65)
