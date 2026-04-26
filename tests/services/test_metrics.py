import pandas as pd
import pytest
from app.services.bias.metrics import encode_positive_mask, compute_group_selection_rates, compute_fairness_metrics

def test_encode_positive_mask_bool():
    series = pd.Series([True, False, True])
    mask, label = encode_positive_mask(series)
    assert mask.tolist() == [True, False, True]
    assert label == "True"

def test_encode_positive_mask_numeric():
    series = pd.Series([1, 0, 1])
    mask, label = encode_positive_mask(series)
    assert mask.tolist() == [True, False, True]
    assert label == "1"

def test_compute_group_selection_rates():
    df = pd.DataFrame({"gender": ["M", "F", "M", "F"]})
    mask = pd.Series([True, False, True, True])
    rates = compute_group_selection_rates(df, "gender", mask)
    assert rates["M"] == 1.0
    assert rates["F"] == 0.5

def test_compute_fairness_metrics():
    rates = {"M": 0.8, "F": 0.4}
    metrics = compute_fairness_metrics(rates)
    assert metrics["dp_diff"] == 0.4
    assert metrics["di_ratio"] == pytest.approx(0.5, rel=1e-5)
