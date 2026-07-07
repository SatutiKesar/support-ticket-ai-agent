"""
Natural-language question answering over the tickets table.

Pipeline:
  1. Ask the LLM to translate the user's question into a single read-only
     SQL SELECT statement (given the table schema).
  2. Validate the SQL is safe (SELECT-only, single statement, no dangerous
     keywords) before ever executing it.
  3. Execute it against a read-only SQLite connection.
  4. Ask the LLM to phrase a short natural-language answer from the
     question + the (capped) result rows.

Splitting "generate SQL" and "phrase the answer" into two explicit steps
(rather than asking one model call to both guess an answer and be right)
makes the system auditable: every answer can be traced back to the exact
SQL that produced it, which is what you want for a support-ops tool.
"""
import json
import re
import sqlite3

from app import config, data, llm

MAX_ROWS_RETURNED_TO_LLM = 30
MAX_SQL_REPAIR_ATTEMPTS = 2  # total tries = 1 initial + this many repair retries

_DISALLOWED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|PRAGMA|CREATE|REPLACE|VACUUM)\b",
    re.IGNORECASE,
)


class NLQueryError(Exception):
    pass


def _build_sql_prompt(question: str) -> list[dict]:
    schema = data.schema_description()
    system = (
        "You are a precise data analyst. You translate natural-language questions "
        "into a single read-only SQLite SELECT query against the schema below.\n\n"
        f"{schema}\n"
        "Rules:\n"
        "- Output ONLY a JSON object: {\"sql\": \"<query>\", \"explanation\": \"<one sentence>\"}\n"
        "- No markdown fences, no extra text outside the JSON object.\n"
        "- The query MUST start with SELECT and must be a single statement (no semicolon-chained statements).\n"
        "- Never use INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA/CREATE.\n"
        "- Use SQLite date functions (datetime(), julianday(), strftime()) for time-based questions.\n"
        "- If the question is ambiguous, make a reasonable assumption and note it in 'explanation'."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip accidental markdown fences if the model adds them anyway.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise NLQueryError(f"Could not find a JSON object in model output: {text[:200]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise NLQueryError(f"Model returned invalid JSON: {exc}") from exc


def _validate_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";").strip()
    if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
        raise NLQueryError("Generated query is not a SELECT statement; refusing to run it.")
    if ";" in cleaned:
        raise NLQueryError("Multiple SQL statements detected; refusing to run it.")
    if _DISALLOWED_KEYWORDS.search(cleaned):
        raise NLQueryError("Generated query contains a disallowed keyword; refusing to run it.")
    return cleaned


def _build_repair_prompt(question: str, failed_sql: str, error: str) -> list[dict]:
    schema = data.schema_description()
    system = (
        "You are a precise data analyst. Your previous SQL query failed. Fix it.\n\n"
        f"{schema}\n"
        "Rules:\n"
        "- Output ONLY a JSON object: {\"sql\": \"<query>\", \"explanation\": \"<one sentence>\"}\n"
        "- No markdown fences, no extra text outside the JSON object.\n"
        "- The query MUST start with SELECT and must be a single statement.\n"
        "- Never use INSERT/UPDATE/DELETE/DROP/ALTER/ATTACH/PRAGMA/CREATE.\n"
        "- Only use column names that appear in the schema above."
    )
    user = (
        f"Original question: {question}\n"
        f"Query that failed: {failed_sql}\n"
        f"Error: {error}\n"
        "Return a corrected query."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _run_sql(sql: str) -> list[dict]:
    conn = data.get_readonly_connection()
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    except sqlite3.Error as exc:
        raise NLQueryError(f"SQL execution failed: {exc}") from exc
    finally:
        conn.close()


def _build_answer_prompt(question: str, sql: str, rows: list[dict]) -> list[dict]:
    truncated = rows[:MAX_ROWS_RETURNED_TO_LLM]
    note = ""
    if len(rows) > MAX_ROWS_RETURNED_TO_LLM:
        note = f" (showing first {MAX_ROWS_RETURNED_TO_LLM} of {len(rows)} rows)"
    system = (
        "You answer a user's question about support tickets using ONLY the query "
        "result data provided. Be concise (1-3 sentences). If the result set is "
        "empty, say so plainly. Do not invent numbers not present in the data."
    )
    user = (
        f"Question: {question}\n"
        f"SQL used: {sql}\n"
        f"Result rows{note}: {json.dumps(truncated, default=str)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _generate_and_run_sql(question: str) -> tuple[str, str, list[dict], int]:
    """
    Generates SQL for the question, validates and executes it. If it's invalid
    or fails to execute, feeds the exact error back to the LLM and retries
    (up to MAX_SQL_REPAIR_ATTEMPTS times) instead of failing on the first
    slip-up -- LLMs occasionally emit a column typo or a non-existent
    function name, and a one-line "here's the error, fix it" round trip
    resolves most of those without ever surfacing an error to the user.

    Returns (sql, explanation, rows, attempts_used).
    """
    raw = llm.chat(_build_sql_prompt(question))
    parsed = _extract_json(raw)
    sql = parsed.get("sql", "")
    explanation = parsed.get("explanation", "")
    if not sql:
        raise NLQueryError("Model did not return a 'sql' field.")

    last_error = None
    for attempt in range(MAX_SQL_REPAIR_ATTEMPTS + 1):
        try:
            safe_sql = _validate_sql(sql)
            rows = _run_sql(safe_sql)
            return safe_sql, explanation, rows, attempt
        except NLQueryError as exc:
            last_error = exc
            if attempt >= MAX_SQL_REPAIR_ATTEMPTS:
                break
            repair_raw = llm.chat(_build_repair_prompt(question, sql, str(exc)))
            parsed = _extract_json(repair_raw)
            sql = parsed.get("sql", "")
            explanation = parsed.get("explanation", explanation)
            if not sql:
                break

    raise NLQueryError(
        f"Could not produce a working query after {MAX_SQL_REPAIR_ATTEMPTS + 1} attempt(s). "
        f"Last error: {last_error}"
    )


def answer_question(question: str) -> dict:
    """
    Full pipeline. Returns a dict with keys: question, sql, explanation, rows,
    answer, repair_attempts. Raises NLQueryError on any unrecoverable failure.
    """
    if not question or not question.strip():
        raise NLQueryError("Question must not be empty.")

    safe_sql, explanation, rows, attempts_used = _generate_and_run_sql(question)
    answer_raw = llm.chat(_build_answer_prompt(question, safe_sql, rows))

    return {
        "question": question,
        "sql": safe_sql,
        "explanation": explanation,
        "row_count": len(rows),
        "rows": rows[:MAX_ROWS_RETURNED_TO_LLM],
        "answer": answer_raw.strip(),
        "repair_attempts": attempts_used,
    }
