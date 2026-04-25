import json
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class ValidatedInput:
    dataframe: pd.DataFrame
    target_column: str
    sensitive_columns: List[str]


def _parse_sensitive_columns(raw_value: str) -> List[str]:
    value = (raw_value or "").strip()
    if not value:
        raise ValueError("Sensitive columns are required.")

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            columns = [str(item).strip() for item in parsed if str(item).strip()]
            if columns:
                return columns
    except json.JSONDecodeError:
        pass

    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns:
        raise ValueError("Sensitive columns are required.")
    return columns


def _validate_target(dataframe: pd.DataFrame, target_column: str) -> None:
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in the dataset.")

    unique_values = dataframe[target_column].dropna().nunique()
    if unique_values < 2:
        raise ValueError("Target column must contain at least two classes.")

    if unique_values > 20 and not pd.api.types.is_numeric_dtype(dataframe[target_column]):
        raise ValueError("Target column must be binary or categorical.")


def validate_analysis_input(
    dataframe: pd.DataFrame,
    target_column: str,
    sensitive_columns: str,
) -> ValidatedInput:
    target_column = (target_column or "").strip()
    if not target_column:
        raise ValueError("Target column is required.")

    parsed_sensitive_columns = _parse_sensitive_columns(sensitive_columns)
    missing_sensitive = [column for column in parsed_sensitive_columns if column not in dataframe.columns]
    if missing_sensitive:
        raise ValueError(f"Sensitive columns not found: {', '.join(missing_sensitive)}")

    _validate_target(dataframe, target_column)

    required_columns = [target_column, *parsed_sensitive_columns]
    cleaned = dataframe.dropna(subset=required_columns).copy()
    if cleaned.empty:
        raise ValueError("No rows remain after dropping missing values in required columns.")

    insufficient_groups = [
        column for column in parsed_sensitive_columns
        if cleaned[column].nunique(dropna=True) < 2
    ]
    if insufficient_groups:
        raise ValueError(
            f"Sensitive column(s) need at least two groups after cleaning: {', '.join(insufficient_groups)}"
        )

    return ValidatedInput(
        dataframe=cleaned,
        target_column=target_column,
        sensitive_columns=parsed_sensitive_columns,
    )
