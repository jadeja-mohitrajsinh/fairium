from dataclasses import dataclass
from typing import List

import pandas as pd


TARGET_HINTS = [
    "target",
    "label",
    "outcome",
    "approved",
    "approval",
    "hired",
    "hire",
    "selected",
    "accept",
    "accepted",
    "admit",
    "decision",
    "default",
    "risk",
    "attrition",
    "churn",
    "fraud",
    "pass",
    "loan_status",
    "status",
]

SENSITIVE_HINTS = [
    "gender",
    "sex",
    "race",
    "ethnicity",
    "age_group",
    "age group",
    "ageband",
    "age_band",
    "marital",
    "religion",
    "caste",
    "disability",
    "nationality",
    "education",
    "income",
    "location",
    "region",
    "city",
    "zip",
    "postal",
    "department",
    "job",
    "position",
    "role",
    "salary",
    "wage",
    "credit",
    "score",
    "rating",
    "class",
    "category",
    "type",
    "status",
]


@dataclass
class InferenceResult:
    target_column: str
    sensitive_columns: List[str]


def _normalized(value: str) -> str:
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def _is_usable_target(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_count = non_null.nunique()
    if unique_count < 1:
        return False

    # Accept any non-empty column as a potential target
    # The encoding function will handle different types appropriately
    return True


def _target_score(column: str, series: pd.Series) -> tuple[int, int]:
    normalized = _normalized(column)
    hint_score = max((len(hint) for hint in TARGET_HINTS if hint in normalized), default=0)
    cardinality = series.dropna().nunique()
    return (hint_score, -cardinality)


def infer_target_column(dataframe: pd.DataFrame) -> str:
    candidates = [
        column for column in dataframe.columns
        if _is_usable_target(dataframe[column])
    ]
    if not candidates:
        raise ValueError("Could not infer a target column. The dataset needs a binary or small-category outcome column.")

    hinted = [column for column in candidates if any(hint in _normalized(column) for hint in TARGET_HINTS)]
    ranked = hinted or candidates
    ranked.sort(key=lambda column: _target_score(column, dataframe[column]), reverse=True)
    return ranked[0]


def infer_sensitive_columns(dataframe: pd.DataFrame, target_column: str) -> List[str]:
    explicit = [
        column
        for column in dataframe.columns
        if column != target_column and any(hint in _normalized(column) for hint in SENSITIVE_HINTS)
    ]
    if explicit:
        return explicit[:3]

    fallback = []
    for column in dataframe.columns:
        if column == target_column:
            continue
        series = dataframe[column].dropna()
        if series.empty:
            continue
        unique_count = series.nunique()
        # More lenient fallback: accept columns with 2-20 unique values
        # Also accept numeric columns that could represent categories
        if 2 <= unique_count <= 20:
            fallback.append(column)

    if not fallback:
        raise ValueError("Could not infer sensitive columns. Add columns like gender, race, sex, age_group, education, income, or department.")

    return fallback[:3]


def infer_analysis_columns(dataframe: pd.DataFrame) -> InferenceResult:
    target_column = infer_target_column(dataframe)
    sensitive_columns = infer_sensitive_columns(dataframe, target_column)
    return InferenceResult(
        target_column=target_column,
        sensitive_columns=sensitive_columns,
    )
