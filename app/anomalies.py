"""
Anomaly detection.

Deliberately deterministic/statistical rather than LLM-based: anomaly
flagging is exactly the kind of thing that should be reproducible, testable,
and explainable to a support-ops manager -- an LLM guessing at outliers adds
cost and nondeterminism with no real benefit over IQR/threshold rules here.
(The LLM is used elsewhere, for the NL query pipeline, where it's actually
doing something only an LLM can do: understanding open-ended English.)

Two checks are implemented:
  1. Resolution-time outliers, via the IQR method, computed per category
     (a "long" resolution time means something different for a Critical
     Technical bug than for a General documentation request).
  2. Stale high-priority tickets: High/Critical priority tickets that are
     not Resolved/Closed and have been open longer than STALE_HOURS_THRESHOLD.
"""
from datetime import datetime, timedelta

import pandas as pd

from app import config


def _reference_now(df: pd.DataFrame) -> datetime:
    if config.ANOMALY_REFERENCE_TIME == "now":
        return datetime.now()
    # "data": use the most recent timestamp actually present in the dataset.
    # This dataset is historical (2024 dates), so comparing against real
    # wall-clock time would flag almost every ticket as "stale" -- using the
    # dataset's own most recent timestamp as "today" is the meaningful choice.
    max_ts = df["created_at"].max()
    if pd.isna(max_ts):
        return datetime.now()
    return max_ts.to_pydatetime()


def _resolution_time_outliers(df: pd.DataFrame) -> list[dict]:
    outliers = []
    resolved = df.dropna(subset=["resolution_time_hrs"])
    for category, group in resolved.groupby("category"):
        if len(group) < 4:
            # Not enough samples in this category for IQR to be meaningful.
            continue
        q1 = group["resolution_time_hrs"].quantile(0.25)
        q3 = group["resolution_time_hrs"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + config.IQR_MULTIPLIER * iqr
        flagged = group[group["resolution_time_hrs"] > upper_bound]
        for _, row in flagged.iterrows():
            outliers.append(
                {
                    "ticket_id": row["ticket_id"],
                    "category": category,
                    "priority": row["priority"],
                    "status": row["status"],
                    "resolution_time_hrs": row["resolution_time_hrs"],
                    "category_upper_bound_hrs": round(float(upper_bound), 2),
                    "agent_id": row["agent_id"],
                }
            )
    outliers.sort(key=lambda r: r["resolution_time_hrs"], reverse=True)
    return outliers


def _stale_high_priority_open(df: pd.DataFrame, now: datetime) -> list[dict]:
    is_high_priority = df["priority"].isin(config.HIGH_PRIORITY_LEVELS)
    is_open = ~df["status"].isin(config.OPEN_STATUSES_EXCLUDED)
    candidates = df[is_high_priority & is_open].dropna(subset=["created_at"])

    stale = []
    threshold = timedelta(hours=config.STALE_HOURS_THRESHOLD)
    for _, row in candidates.iterrows():
        age = now - row["created_at"].to_pydatetime()
        if age >= threshold:
            stale.append(
                {
                    "ticket_id": row["ticket_id"],
                    "category": row["category"],
                    "priority": row["priority"],
                    "status": row["status"],
                    "created_at": row["created_at"].isoformat(),
                    "age_hours": round(age.total_seconds() / 3600, 1),
                    "agent_id": row["agent_id"],
                }
            )
    stale.sort(key=lambda r: r["age_hours"], reverse=True)
    return stale


def detect_anomalies(df: pd.DataFrame) -> dict:
    now = _reference_now(df)
    resolution_outliers = _resolution_time_outliers(df)
    stale_tickets = _stale_high_priority_open(df, now)

    return {
        "reference_time": now.isoformat(),
        "reference_time_mode": config.ANOMALY_REFERENCE_TIME,
        "stale_hours_threshold": config.STALE_HOURS_THRESHOLD,
        "iqr_multiplier": config.IQR_MULTIPLIER,
        "summary": {
            "total_tickets": int(len(df)),
            "resolution_time_outlier_count": len(resolution_outliers),
            "stale_high_priority_open_count": len(stale_tickets),
        },
        "resolution_time_outliers": resolution_outliers,
        "stale_high_priority_open": stale_tickets,
    }
