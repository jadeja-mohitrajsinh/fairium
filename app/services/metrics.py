from typing import Dict, Tuple

import numpy as np
import pandas as pd


EPSILON = 1e-6


def encode_positive_mask(series: pd.Series) -> Tuple[pd.Series, str]:
    non_null = series.dropna()
    if non_null.empty:
        raise ValueError("Target column has no usable values.")

    # Handle boolean
    if pd.api.types.is_bool_dtype(non_null):
        return series.fillna(False).astype(bool), "True"

    # Handle numeric
    if pd.api.types.is_numeric_dtype(non_null):
        unique_values = sorted(pd.unique(non_null))
        if len(unique_values) == 2:
            positive_value = unique_values[-1]
            return series == positive_value, str(positive_value)
        # For multi-class or continuous, use median as threshold
        median_value = non_null.median()
        return pd.to_numeric(series, errors="coerce").fillna(median_value) >= median_value, f">= {median_value:.2f}"

    # Handle categorical/string
    normalized = series.astype(str).str.strip()
    lowered = normalized.str.lower()
    positive_tokens = {"1", "true", "yes", "y", "approved", "accept", "accepted", "selected", "hired", "positive", "good"}

    unique_values = list(pd.unique(lowered.dropna()))
    if len(unique_values) == 2:
        positive_candidates = [value for value in unique_values if value in positive_tokens]
        positive_value = positive_candidates[0] if positive_candidates else sorted(unique_values)[-1]
        return lowered == positive_value, positive_value
    
    # For multi-class categorical, use the most frequent class as positive
    if len(unique_values) > 2:
        value_counts = lowered.value_counts()
        most_frequent = value_counts.index[0]
        return lowered == most_frequent, most_frequent

    # Single value - treat all as positive
    if len(unique_values) == 1:
        return pd.Series([True] * len(series), index=series.index), unique_values[0]

    raise ValueError("Target column has no usable values.")


def compute_group_selection_rates(dataframe: pd.DataFrame, sensitive_column: str, positive_mask: pd.Series) -> Dict[str, float]:
    working = dataframe[[sensitive_column]].copy()
    working["_positive"] = positive_mask.astype(int)
    # Exclude NaN groups from analysis
    grouped = working.groupby(sensitive_column, dropna=True)["_positive"].agg(["sum", "count"])

    rates: Dict[str, float] = {}
    for group_name, row in grouped.iterrows():
        label = str(group_name)
        rate = float(row["sum"]) / float(row["count"])
        rates[label] = float(round(rate, 6))
    return rates


def compute_fairness_metrics(group_rates: Dict[str, float]) -> Dict[str, float]:
    rates = np.array(list(group_rates.values()), dtype=float)
    max_rate = float(rates.max())
    min_rate = float(rates.min())
    dp_diff = max_rate - min_rate
    di_ratio = (min_rate + EPSILON) / (max_rate + EPSILON)
    return {
        "dp_diff": round(dp_diff, 6),
        "di_ratio": round(di_ratio, 6),
    }
