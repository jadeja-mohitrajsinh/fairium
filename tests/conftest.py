import pytest
from fastapi.testclient import TestClient
from app.main import app
import pandas as pd

@pytest.fixture
def client():
    # Using a context manager to ensure startup/shutdown events run
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "gender": ["male", "female", "male", "female", "male", "female"],
        "target": [1, 0, 1, 0, 1, 1],
        "age": [25, 30, 35, 40, 45, 50]
    })
