import io
import pandas as pd
import pytest

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_analyze_dataset_endpoint(client):
    # Create a small dummy CSV
    df = pd.DataFrame({
        "gender": ["male", "female", "male", "female"],
        "target": [1, 0, 1, 0],
        "age": [25, 30, 35, 40]
    })
    csv_content = df.to_csv(index=False).encode("utf-8")
    
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/api/analyze", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "detected_target" in data
    assert "fairness_metrics" in data

def test_analyze_text_endpoint(client):
    payload = {"text": "We are looking for a young and energetic male candidate."}
    response = client.post("/api/analyze-text", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "bias_detected" in data
    assert data["bias_detected"] in ["Yes", "Possible"]
