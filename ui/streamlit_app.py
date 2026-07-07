"""
Minimal Streamlit UI for the Support Ticket AI System.

This is a thin client over the FastAPI backend (app/main.py) -- it holds no
business logic of its own, so the API remains the single source of truth
and can be used independently (e.g. via curl or the /docs Swagger page).

Run with:  streamlit run ui/streamlit_app.py
(the FastAPI backend must already be running, see README)
"""
import json
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Ticket AI System", layout="wide")
st.title("🎫 Support Ticket AI System")
st.caption(f"Backend: {API_BASE_URL}")

EXAMPLE_QUESTIONS = [
    "How many tickets are currently open?",
    "Which agent resolved the most tickets?",
    "What is the average customer rating for Technical category tickets?",
    "Show me all Critical tickets not resolved within 12 hours.",
]


def api_get(path: str, params: dict | None = None):
    resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: dict):
    resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(resp.json().get("detail", resp.text))
    return resp.json()


def render_result_visual(rows: list[dict]) -> None:
    """
    Auto-pick a sensible visualization based on the shape of the query
    result, instead of always dumping a raw table:
      - a single row with a single numeric value  -> big metric card
      - a date/time-like column + a numeric column -> line chart
      - one text/category column + one numeric column -> bar chart
      - anything else -> plain table
    """
    if not rows:
        st.info("No rows returned.")
        return

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    if len(df) == 1 and len(numeric_cols) == 1:
        col_name = numeric_cols[0]
        st.metric(col_name.replace("_", " ").title(), df[col_name].iloc[0])
        return

    date_like = [
        c for c in non_numeric_cols
        if "date" in c.lower() or "created" in c.lower()
        or ("time" in c.lower() and "hrs" not in c.lower())
    ]
    if date_like and numeric_cols:
        try:
            chart_df = df.copy()
            chart_df[date_like[0]] = pd.to_datetime(chart_df[date_like[0]])
            chart_df = chart_df.set_index(date_like[0])[numeric_cols]
            st.line_chart(chart_df)
            st.dataframe(df)
            return
        except Exception:  # noqa: BLE001 - fall through to table/bar below
            pass

    if len(non_numeric_cols) == 1 and len(numeric_cols) == 1 and len(df) <= 30:
        chart_df = df.set_index(non_numeric_cols[0])[numeric_cols[0]]
        st.bar_chart(chart_df)
        st.dataframe(df)
        return

    st.dataframe(df)


# --- Sidebar: health / status ---
with st.sidebar:
    st.header("System status")
    try:
        health = api_get("/health")
        st.success(f"API reachable — {health['rows_loaded']} tickets loaded")
        st.write(f"LLM provider: `{health['llm_provider']}`")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot reach API at {API_BASE_URL}\n\n{exc}")
        st.stop()

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""

tab_query, tab_anomalies, tab_overview = st.tabs(
    ["💬 Ask a question", "🚨 Anomalies", "📊 Dataset overview"]
)

# --- Tab 1: NL query ---
with tab_query:
    st.subheader("Ask a natural-language question about the tickets")

    st.caption("Try one of these, or type your own below:")
    chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, ex_q in zip(chip_cols, EXAMPLE_QUESTIONS):
        if col.button(ex_q, use_container_width=True):
            st.session_state.pending_question = ex_q

    question = st.text_input(
        "Your question",
        value=st.session_state.pending_question,
        placeholder="e.g. Which category has the worst average resolution time?",
    )

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking..."):
            try:
                result = api_post("/query", {"question": question})
                st.markdown(f"### {result['answer']}")

                repairs = result.get("repair_attempts", 0)
                if repairs:
                    st.caption(
                        f"ℹ️ The first generated query failed validation/execution; "
                        f"it self-corrected after {repairs} repair attempt(s)."
                    )

                render_result_visual(result["rows"])

                with st.expander("How this was computed"):
                    st.write(result.get("explanation", ""))
                    st.code(result["sql"], language="sql")
                    st.write(f"{result['row_count']} row(s) returned")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Query failed: {exc}")
        st.session_state.pending_question = ""

# --- Tab 2: anomalies ---
with tab_anomalies:
    st.subheader("Anomaly scan")
    if st.button("Run anomaly scan"):
        with st.spinner("Scanning..."):
            try:
                report = api_get("/anomalies")
                st.session_state["anomaly_report"] = report
            except Exception as exc:  # noqa: BLE001
                st.error(f"Anomaly scan failed: {exc}")

    report = st.session_state.get("anomaly_report")
    if report:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total tickets", report["summary"]["total_tickets"])
        col2.metric(
            "Resolution-time outliers",
            report["summary"]["resolution_time_outlier_count"],
        )
        col3.metric(
            "Stale high-priority open",
            report["summary"]["stale_high_priority_open_count"],
        )
        st.caption(
            f"Reference 'now' = {report['reference_time']} "
            f"(mode: {report['reference_time_mode']}) · "
            f"stale threshold = {report['stale_hours_threshold']}h · "
            f"IQR multiplier = {report['iqr_multiplier']}"
        )

        st.download_button(
            "⬇️ Download full report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="anomaly_report.json",
            mime="application/json",
        )

        st.markdown("**Resolution-time outliers** (per-category IQR)")
        if report["resolution_time_outliers"]:
            st.dataframe(pd.DataFrame(report["resolution_time_outliers"]))
        else:
            st.info("None found.")

        st.markdown("**Stale High/Critical tickets still open**")
        if report["stale_high_priority_open"]:
            st.dataframe(pd.DataFrame(report["stale_high_priority_open"]))
        else:
            st.info("None found.")
    else:
        st.info("Click 'Run anomaly scan' to check for issues.")

# --- Tab 3: dataset overview ---
with tab_overview:
    st.subheader("Dataset overview")
    try:
        listing = api_get("/tickets", params={"limit": 500})
        df = pd.DataFrame(listing["tickets"])
        st.write(f"{listing['count']} tickets loaded")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**By category**")
            st.bar_chart(df["category"].value_counts())
        with c2:
            st.markdown("**By priority**")
            st.bar_chart(df["priority"].value_counts())
        st.markdown("**Sample rows**")
        st.dataframe(df.head(20))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load dataset overview: {exc}")
