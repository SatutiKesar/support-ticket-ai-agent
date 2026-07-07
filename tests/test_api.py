from fastapi.testclient import TestClient

from app.main import app


def test_tickets_endpoint_serializes_unresolved_rows_without_error():
    """
    Regression test: unresolved tickets have NaN resolution_time_hrs /
    customer_rating in the dataframe. NaN is not valid JSON, so returning
    raw dataframe records without sanitizing them used to 500 on the very
    first unresolved row (which /tickets?limit=500 always includes).
    """
    with TestClient(app) as client:
        resp = client.get("/tickets", params={"limit": 500})
        assert resp.status_code == 200

        payload = resp.json()
        assert payload["count"] == 500

        open_tickets = [t for t in payload["tickets"] if t["status"] == "Open"]
        assert open_tickets, "expected at least one Open ticket in the dataset"
        # Should be JSON null (Python None), never NaN/float('nan').
        assert open_tickets[0]["resolution_time_hrs"] is None
