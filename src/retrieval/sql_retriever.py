"""
src/retrieval/sql_retriever.py
------------------------------
Text-to-SQL retrieval engine for the hybrid-RAG pipeline.

Pipeline
--------
  User question
      │
      ▼
  generate_sql()   ← LLM (ChatGroq) + live schema context
      │
      ▼
  validate_sql()   ← starts-with-SELECT guard + SQLite EXPLAIN QUERY PLAN
      │  (retry once on failure)
      ▼
  execute_sql()    ← SQLAlchemy query, max 50 rows
      │
      ▼
  result dict  {"sql", "columns", "rows", "row_count"}

Usage
-----
  # From project root:
  python src/retrieval/sql_retriever.py
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import sys
import warnings
from pathlib import Path

# ── Make src/ importable when this script is run directly ────────────────────
# __file__ is  …/src/retrieval/sql_retriever.py
# parent.parent is …/src/
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Project config (loads .env automatically) ─────────────────────────────────
from config import DB_PATH, LLM_MODEL, GROQ_API_KEY  # noqa: E402

# ── Third-party ───────────────────────────────────────────────────────────────
from sqlalchemy import create_engine, inspect, text
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ── SQLAlchemy engine (module-level singleton) ─────────────────────────────────
# Created once when the module is imported so repeated calls share the
# connection pool rather than opening a new file handle every time.
_engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – SCHEMA CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
# WHY: The LLM has no knowledge of our specific database schema.  We must
# inject it into the prompt as a formatted string so the model knows what
# tables and columns exist before it writes any SQL.
#
# HOW: SQLAlchemy's Inspector API lets us introspect any database without
# needing to hard-code table definitions.  This means the schema context
# stays accurate even if someone adds new tables or columns later.

def get_schema_context() -> str:
    """
    Inspect the live SQLite database and return a formatted string that
    describes every table, its columns, and their data types.

    Example output
    --------------
    Table: products
      - id: INTEGER
      - name: VARCHAR(120)
      - category: VARCHAR(60)
      - price: FLOAT
      - stock: INTEGER

    Table: orders
      ...

    Returns
    -------
    str
        Ready-to-inject schema description for LLM prompts.
    """
    inspector = inspect(_engine)
    table_names = inspector.get_table_names()

    lines: list[str] = []
    for table in table_names:
        lines.append(f"Table: {table}")
        for col in inspector.get_columns(table):
            lines.append(f"  - {col['name']}: {col['type']}")
        lines.append("")          # blank line between tables

    return "\n".join(lines).strip()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – SQL GENERATION
# ══════════════════════════════════════════════════════════════════════════════
# WHY: We want the LLM to translate a plain-English question into a valid
# SQLite SELECT query.  Giving it the live schema plus a strict system prompt
# ("return ONLY the SQL, no markdown, no explanation") maximises the chance
# of getting clean, directly executable output.
#
# HOW: ChatGroq with LLM_MODEL from config.py.  The function also accepts an
# optional `extra_hint` string so the pipeline can inject a retry message
# ("the previous query was invalid") without duplicating prompt logic.

def generate_sql(user_question: str, extra_hint: str = "") -> str:
    """
    Ask the LLM to translate *user_question* into a SQLite SELECT query.

    Parameters
    ----------
    user_question : str
        The natural-language question to convert.
    extra_hint : str, optional
        Additional instruction appended to the user message (used for retries).

    Returns
    -------
    str
        The raw SQL string returned by the LLM, stripped of surrounding
        whitespace.  May still contain backticks/markdown on rare LLM
        misbehaviour — validate_sql() will catch those.
    """
    schema = get_schema_context()

    system_prompt = (
        "You are a SQL expert. Given the following SQLite database schema, "
        "write a single valid SQLite SELECT query to answer the user's question.\n"
        "Return ONLY the SQL query — no explanation, no markdown, no backticks, "
        "no code fences. The query must start with SELECT."
    )

    user_content = (
        f"Database schema:\n{schema}\n\n"
        f"Question: {user_question}"
    )
    if extra_hint:
        user_content += f"\n\nNote: {extra_hint}"

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,          # deterministic — we want one correct answer
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])

    raw_sql = response.content.strip()

    # ── Defensive clean-up: strip markdown code fences if the LLM added them ──
    # Some models wrap output in ```sql ... ``` despite the system instruction.
    if raw_sql.startswith("```"):
        lines = raw_sql.splitlines()
        # drop first line (```sql or ```) and last line (```)
        raw_sql = "\n".join(lines[1:-1]).strip()

    return raw_sql


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – SQL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
# WHY: We never want to execute LLM-generated SQL blindly.  Two checks:
#   1. Syntactic guard  – rejects anything that doesn't start with SELECT,
#      preventing accidental DROP / INSERT / UPDATE from a confused model.
#   2. Semantic dry-run – SQLite's EXPLAIN QUERY PLAN parses the full SQL and
#      builds a query plan without touching any data.  If it raises, the SQL
#      was syntactically invalid (unknown table, bad column name, etc.).
#
# HOW: SQLAlchemy's text() wraps the raw string in a safe Clause object.
# We execute EXPLAIN QUERY PLAN <sql> inside a transaction that we do NOT
# commit, so it's always a true read-only dry-run.

def validate_sql(sql: str) -> bool:
    """
    Check that *sql* is a safe, syntactically valid SQLite SELECT query.

    Checks performed
    ----------------
    1. The query must start with SELECT (case-insensitive).
    2. ``EXPLAIN QUERY PLAN <sql>`` must execute without error.

    Parameters
    ----------
    sql : str
        The SQL string to validate (as returned by generate_sql).

    Returns
    -------
    bool
        True if both checks pass, False otherwise.
    """
    # ── Guard 1: must be a SELECT statement ───────────────────────────────────
    if not sql.upper().lstrip().startswith("SELECT"):
        warnings.warn(
            f"[SQL VALIDATION] Rejected — query does not start with SELECT.\n"
            f"  Query: {sql[:120]}",
            stacklevel=2,
        )
        return False

    # ── Guard 2: dry-run via EXPLAIN QUERY PLAN ───────────────────────────────
    try:
        with _engine.connect() as conn:
            conn.execute(text(f"EXPLAIN QUERY PLAN {sql}"))
        return True
    except Exception as exc:
        warnings.warn(
            f"[SQL VALIDATION] EXPLAIN QUERY PLAN failed: {exc}\n"
            f"  Query: {sql[:200]}",
            stacklevel=2,
        )
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – SQL EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
# WHY: We need a structured return value (not just raw rows) so the rest of
# the pipeline — and eventually the synthesis layer — can present the results
# cleanly without having to understand SQLAlchemy result objects.
#
# HOW: SQLAlchemy's connection.execute() returns a CursorResult whose
# .keys()  gives column names and .fetchmany() limits the row count.
# We wrap everything in a try/except so any runtime error becomes a
# recoverable "error" key in the dict rather than an unhandled exception.

def execute_sql(sql: str) -> dict:
    """
    Execute *sql* against the SQLite database and return structured results.

    Parameters
    ----------
    sql : str
        A validated SELECT query (as returned by validate_sql passing True).

    Returns
    -------
    dict
        On success::

            {
                "sql":       str,          # the query that was run
                "columns":   list[str],    # column names in result order
                "rows":      list[tuple],  # up to 50 data rows
                "row_count": int,          # number of rows returned
            }

        On failure::

            {
                "sql":   str,
                "error": str,   # exception message
            }
    """
    try:
        with _engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows    = result.fetchmany(50)          # hard cap at 50 rows

        return {
            "sql":       sql,
            "columns":   columns,
            "rows":      [tuple(r) for r in rows],
            "row_count": len(rows),
        }
    except Exception as exc:
        return {
            "sql":   sql,
            "error": str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – MAIN PIPELINE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
# WHY: Chains the three stages (generate → validate → execute) into a single
# callable entry-point.  The retry logic sits here so callers never have to
# think about it — they just pass a question and get a result dict back.
#
# Retry strategy: on first validation failure we ask the LLM again with an
# explicit note that the previous query was invalid.  One retry is usually
# enough; if it fails again we still execute so the caller gets meaningful
# error information rather than a silent failure.

def run_sql_pipeline(user_question: str) -> dict:
    """
    Full Text-to-SQL pipeline: generate → validate → (retry?) → execute.

    Parameters
    ----------
    user_question : str
        Plain-English question to answer from the database.

    Returns
    -------
    dict
        Result dict from execute_sql (see its docstring for schema).
        Always contains at minimum the "sql" key.
    """
    print(f"\n{'-'*60}")
    print(f"  Question: {user_question}")
    print(f"{'-'*60}")


    # ── Attempt 1 ─────────────────────────────────────────────────────────────
    sql = generate_sql(user_question)
    print(f"  Generated SQL:\n  {sql}\n")

    valid = validate_sql(sql)
    print(f"  Validation: {'PASSED' if valid else 'FAILED'}")

    # ── Retry on failure ──────────────────────────────────────────────────────
    if not valid:
        print("  Retrying with simplified prompt...")
        sql = generate_sql(
            user_question,
            extra_hint="The previous query was invalid. Generate a simpler query.",
        )
        print(f"  Retry SQL:\n  {sql}\n")

        valid = validate_sql(sql)
        print(f"  Retry Validation: {'PASSED' if valid else 'FAILED (proceeding anyway)'}")

    # ── Execute ───────────────────────────────────────────────────────────────
    result = execute_sql(sql)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – TESTING
# ══════════════════════════════════════════════════════════════════════════════

def _print_result(result: dict) -> None:
    """Pretty-print a result dict from run_sql_pipeline."""
    if "error" in result:
        print(f"  [ERROR] Execution error: {result['error']}")
        return

    cols = result["columns"]
    rows = result["rows"]

    if not rows:
        print("  (no rows returned)")
        return

    # -- Column header --
    col_widths = [
        max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
        for i, c in enumerate(cols)
    ]
    header = " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols))
    separator = "-+-".join("-" * w for w in col_widths)

    print(header)
    print(separator)
    for row in rows:
        print(" | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row)))
    print(f"\n{result['row_count']} row(s) returned.\n")


if __name__ == "__main__":
    # Ensure stdout can handle any UTF-8 characters on Windows terminals
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    TEST_QUESTIONS = [
        "What are the top 5 products by total revenue?",
        "How many orders were placed per region?",
        "Which customer has the highest total spent?",
    ]

    print("\n" + "=" * 60)
    print("  SQL RETRIEVER -- Test Run")
    print("=" * 60)

    for question in TEST_QUESTIONS:
        result = run_sql_pipeline(question)
        _print_result(result)

    print("=" * 60)
    print("  [DONE] All test queries complete.")
    print("=" * 60)
