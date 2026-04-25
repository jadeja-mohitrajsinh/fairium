from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from app.services.metrics import encode_positive_mask


def _prepare_feature_matrix(dataframe: pd.DataFrame, target_column: str) -> pd.DataFrame:
    features = dataframe.drop(columns=[target_column]).copy()
    encoded = pd.DataFrame(index=features.index)

    for column in features.columns:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            median_value = numeric_series.median()
            if pd.isna(median_value):
                median_value = 0.0
            encoded[column] = numeric_series.fillna(median_value)
        else:
            encoded[column] = LabelEncoder().fit_transform(series.astype(str).fillna("missing"))

    # Final safety pass: guarantee model-ready matrix with no NaN/inf values.
    encoded = encoded.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return encoded


def detect_bias_drivers(dataframe: pd.DataFrame, target_column: str, top_k: int = 3) -> List[Dict[str, float]]:
    positive_mask, _ = encode_positive_mask(dataframe[target_column])
    target = positive_mask.astype(int)
    features = _prepare_feature_matrix(dataframe, target_column)

    if features.empty or target.nunique(dropna=True) < 2:
        return []

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        model.fit(features, target)
    except ValueError:
        return []
    importances = model.feature_importances_

    ranked = sorted(
        (
            {"feature": feature, "impact": round(float(importance), 6)}
            for feature, importance in zip(features.columns, importances)
        ),
        key=lambda item: item["impact"],
        reverse=True,
    )
    return ranked[:top_k]


def detect_proxy_features(
    dataframe: pd.DataFrame,
    target_column: str,
    sensitive_columns: List[str],
    threshold: float = 0.5,
    top_k: int = 5,
) -> List[Dict[str, float]]:
    features = _prepare_feature_matrix(dataframe, target_column)
    proxy_candidates: List[Dict[str, float]] = []

    if features.empty:
        return proxy_candidates

    for sensitive_column in sensitive_columns:
        if sensitive_column not in dataframe.columns:
            continue

        sensitive_series = dataframe[sensitive_column].astype(str).fillna("missing")
        if sensitive_series.nunique() < 2:
            continue

        sensitive_encoded = LabelEncoder().fit_transform(sensitive_series)
        feature_columns = [column for column in features.columns if column not in sensitive_columns]
        if not feature_columns:
            continue

        for feature in feature_columns:
            feature_values = features[feature].to_numpy(dtype=float)
            score = float(abs(np.corrcoef(feature_values, sensitive_encoded)[0, 1]))
            if np.isnan(score):
                score = 0.0
            if score >= threshold:
                proxy_candidates.append(
                    {
                        "sensitive_column": sensitive_column,
                        "feature": feature,
                        "correlation": float(round(score, 6)),
                    }
                )

    proxy_candidates.sort(key=lambda item: item["correlation"], reverse=True)
    deduped: List[Dict[str, float]] = []
    seen = set()
    for candidate in proxy_candidates:
        key = (candidate["sensitive_column"], candidate["feature"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= top_k:
            break
    return deduped
