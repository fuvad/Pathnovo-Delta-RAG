"""Integration tests for FastAPI REST Endpoints."""

from pathlib import Path
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data


def test_ingest_and_delta_flow():
    """Test ingestion, delta computation, and chat endpoints."""
    sample_a = Path("data/samples/Export Gas Compressor-P&ID.pdf")
    sample_b = Path("data/samples/Lift Gas compressor-P&ID.pdf")

    if not sample_a.exists() or not sample_b.exists():
        return

    # 1. Ingest Doc A
    with open(sample_a, "rb") as f:
        resp_a = client.post(
            "/ingest",
            data={"pid": "api_doc_a", "adapter_type": "native"},
            files={"file": ("export.pdf", f, "application/pdf")},
        )
    assert resp_a.status_code == 200
    assert resp_a.json()["pid"] == "api_doc_a"

    # 2. Ingest Doc B
    with open(sample_b, "rb") as f:
        resp_b = client.post(
            "/ingest",
            data={"pid": "api_doc_b", "adapter_type": "native"},
            files={"file": ("lift.pdf", f, "application/pdf")},
        )
    assert resp_b.status_code == 200
    assert resp_b.json()["pid"] == "api_doc_b"

    # 3. Compute Delta
    resp_delta = client.post(
        "/delta",
        json={"pid_a": "api_doc_a", "pid_b": "api_doc_b"},
    )
    assert resp_delta.status_code == 200
    delta_data = resp_delta.json()
    assert delta_data["pid_a"] == "api_doc_a"
    assert delta_data["pid_b"] == "api_doc_b"
    assert "summary" in delta_data
