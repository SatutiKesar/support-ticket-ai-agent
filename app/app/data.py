"""
Data ingestion layer.

Loads support_tickets.csv with pandas, does light cleaning/type coercion,
and materializes it into a local SQLite file so it can be queried with SQL
(used by the NL -> SQL pipeline) as well as with pandas (used by the
anomaly detector).
"""
import sqlite3
from pathlib import Path

import pandas as pd

from app import config

REQUIRED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]


class DataLoadError(Exception):
    pass


def load_dataframe(csv_path: str = config.CSV_PATH) -> pd.DataFrame:
    """Load and lightly clean the tickets CSV. Raises DataLoadError on problems."""
    path = Path(csv_path)
    if not path.exists():
        raise DataLoadError(f"CSV file not found at {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Failed to parse CSV: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(f"CSV is missing expected columns: {missing}")

    # Type coercion, tolerant of bad rows (coerced to NaT/NaN rather than raising)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    for col in ["response_time_hrs", "resolution_time_hrs", "customer_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["ticket_id", "category", "priority", "status", "agent_id", "issue_summary"]:
        df[col] = df[col].astype(str).str.strip()

    dropped = df["created_at"].isna().sum()
    if dropped:
        # Keep going, but this is worth surfacing to the caller/logs.
        print(f"[data] warning: {dropped} row(s) had unparseable created_at values")

    return df


def build_sqlite_db(df: pd.DataFrame, db_path: str = config.SQLITE_PATH) -> None:
    """(Re)builds the local SQLite file from the given dataframe."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        # created_at stored as ISO text so SQLite's date functions work on it
        out = df.copy()
        out["created_at"] = out["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out.to_sql(config.TABLE_NAME, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def get_readonly_connection(db_path: str = config.SQLITE_PATH) -> sqlite3.Connection:
    """
    Fresh, read-only connection per call. Opening a new connection per request
    (rather than sharing one global connection/cursor across threads) avoids
    the classic Streamlit/FastAPI 'database is locked' problem you get from
    reusing a single SQLite connection across multiple requests/refreshes.
    """
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


def initialize() -> pd.DataFrame:
    """Convenience entrypoint: load CSV, (re)build SQLite, return the dataframe."""
    df = load_dataframe()
    build_sqlite_db(df)
    return df


def schema_description() -> str:
    """Human/LLM-readable schema description used in NL->SQL prompts."""
    return f"""Table: {config.TABLE_NAME}
Columns:
  ticket_id            TEXT    - unique ticket identifier, e.g. 'TKT-001'
  created_at           TEXT    - timestamp the ticket was created, format 'YYYY-MM-DD HH:MM:SS'
  category             TEXT    - e.g. 'Billing', 'Technical', 'General', 'Account'
  priority             TEXT    - one of 'Low', 'Medium', 'High', 'Critical'
  status               TEXT    - e.g. 'Open', 'In Progress', 'Resolved', 'Closed'
  response_time_hrs    REAL    - hours until first response
  resolution_time_hrs  REAL    - hours until resolution
  agent_id             TEXT    - support agent identifier, e.g. 'AGT-03'
  customer_rating      REAL    - customer satisfaction rating, 1-5 (may be null)
  issue_summary        TEXT    - free-text one-line description of the issue
"""
