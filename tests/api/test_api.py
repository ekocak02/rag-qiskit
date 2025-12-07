import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Import the app to test
from src.api.main import app

client = TestClient(app)

# Mock pipeline result
MOCK_RAG_RESULT = {
    "answer": "This is a test answer.",
    "source_documents": [
        {
            "content": "Doc 1 content",
            "metadata": {"source": "doc1.pdf"},
            "rerank_score": 0.95
        },
        {
            "content": "Doc 2 content",
            "metadata": {"source": "doc2.py"},
            "rerank_score": 0.88
        }
    ]
}

@pytest.fixture
def mock_pipeline():
    mock_instance = MagicMock()
    mock_instance.run.return_value = MOCK_RAG_RESULT

    with patch("src.api.main.pipeline", mock_instance):
        yield mock_instance

def test_health_check(mock_pipeline):

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "pipeline_loaded": True}

def test_query_endpoint(mock_pipeline):

    payload = {"query": "What is Qiskit?"}
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a test answer."
    assert len(data["sources"]) == 2
    assert data["sources"][0]["metadata"]["source"] == "doc1.pdf"

def test_query_no_pipeline():

    with patch("src.api.main.pipeline", None):
        response = client.post("/query", json={"query": "fail"})
        assert response.status_code == 503
        assert "Pipeline not initialized" in response.json()["detail"]
