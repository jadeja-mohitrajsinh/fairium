import pandas as pd
import pytest
from app.services.mitigation.strategies import compute_reweighting_weights

def test_compute_reweighting_weights_logic():
    df = pd.DataFrame({
        "gender": ["M", "M", "F", "F"],
        "target": [1, 0, 1, 0]
    })
    weights = compute_reweighting_weights(df, "target", "gender")
    assert len(weights) == 4
    assert all(w > 0 for w in weights)

def test_compute_reweighting_weights_imbalance():
    df = pd.DataFrame({
        "gender": ["M", "M", "M", "F"],
        "target": [1, 1, 1, 0]
    })
    weights = compute_reweighting_weights(df, "target", "gender")
    assert len(set(weights)) > 1
