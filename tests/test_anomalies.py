from datetime import datetime, timedelta

import pandas as pd

from app import anomalies, config


def _make_df(rows):
    return pd.DataFrame(rows)


def test_stale_high_priority_open_flags_old_unresolved_ticket():
    now = datetime(2024, 6, 1, 12, 0, 0)
    df = _make_df(
        [
            {
                "ticket_id": "TKT-1",
                "category": "Technical",
                "priority": "Critical",
                "status": "Open",
                "created_at": now - timedelta(hours=48),
                "resolution_time_hrs": None,
                "agent_id": "AGT-01",
            },
            {
                "ticket_id": "TKT-2",
                "category": "Technical",
                "priority": "Low",
                "status": "Open",
                "created_at": now - timedelta(hours=48),
                "resolution_time_hrs": None,
                "agent_id": "AGT-02",
            },
            {
                "ticket_id": "TKT-3",
                "category": "Technical",
                "priority": "Critical",
                "status": "Resolved",
                "created_at": now - timedelta(hours=48),
                "resolution_time_hrs": 5.0,
                "agent_id": "AGT-03",
            },
        ]
    )
    stale = anomalies._stale_high_priority_open(df, now)
    stale_ids = {r["ticket_id"] for r in stale}
    assert stale_ids == {"TKT-1"}


def test_resolution_time_outliers_detects_extreme_value():
    rows = []
    for i in range(10):
        rows.append(
            {
                "ticket_id": f"TKT-{i}",
                "category": "Billing",
                "priority": "Low",
                "status": "Resolved",
                "resolution_time_hrs": 4.0 + (i % 3),
                "agent_id": "AGT-01",
            }
        )
    rows.append(
        {
            "ticket_id": "TKT-OUTLIER",
            "category": "Billing",
            "priority": "Low",
            "status": "Resolved",
            "resolution_time_hrs": 500.0,
            "agent_id": "AGT-01",
        }
    )
    df = _make_df(rows)
    outliers = anomalies._resolution_time_outliers(df)
    ids = {r["ticket_id"] for r in outliers}
    assert "TKT-OUTLIER" in ids
