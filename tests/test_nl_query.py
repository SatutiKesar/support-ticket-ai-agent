import json

import pytest

from app import data, nl_query


@pytest.fixture(autouse=True, scope="module")
def _ensure_db():
    data.initialize()


def test_validate_sql_accepts_plain_select():
    sql = nl_query._validate_sql("SELECT COUNT(*) FROM tickets WHERE status = 'Open';")
    assert sql.upper().startswith("SELECT")


@pytest.mark.parametrize(
    "bad_sql",
    [
        "DROP TABLE tickets;",
        "DELETE FROM tickets;",
        "SELECT * FROM tickets; DROP TABLE tickets;",
        "UPDATE tickets SET status='Closed';",
        "not even sql",
    ],
)
def test_validate_sql_rejects_unsafe_queries(bad_sql):
    with pytest.raises(nl_query.NLQueryError):
        nl_query._validate_sql(bad_sql)


def test_extract_json_handles_markdown_fences():
    text = '```json\n{"sql": "SELECT 1", "explanation": "test"}\n```'
    parsed = nl_query._extract_json(text)
    assert parsed["sql"] == "SELECT 1"


def test_extract_json_raises_on_garbage():
    with pytest.raises(nl_query.NLQueryError):
        nl_query._extract_json("no json here at all")


def test_generate_and_run_sql_repairs_after_bad_column(monkeypatch):
    """First LLM call returns SQL with a non-existent column; the repair
    round trip should be triggered and succeed on the second attempt."""
    responses = [
        json.dumps({"sql": "SELECT nonexistent_column FROM tickets", "explanation": "bad"}),
        json.dumps({"sql": "SELECT COUNT(*) AS n FROM tickets", "explanation": "fixed"}),
    ]

    def fake_chat(messages, temperature=0.0, timeout=30):
        return responses.pop(0)

    monkeypatch.setattr(nl_query.llm, "chat", fake_chat)

    sql, explanation, rows, attempts = nl_query._generate_and_run_sql("how many tickets?")
    assert "COUNT" in sql.upper()
    assert attempts == 1
    assert rows[0]["n"] == 500


def test_generate_and_run_sql_raises_after_exhausting_repairs(monkeypatch):
    def fake_chat(messages, temperature=0.0, timeout=30):
        return json.dumps({"sql": "SELECT still_bad_column FROM tickets", "explanation": "bad"})

    monkeypatch.setattr(nl_query.llm, "chat", fake_chat)

    with pytest.raises(nl_query.NLQueryError):
        nl_query._generate_and_run_sql("how many tickets?")
