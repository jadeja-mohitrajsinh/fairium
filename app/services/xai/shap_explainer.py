"""SHAP-based explainability for bias analysis.

This module provides SHAP (SHapley Additive exPlanations) explanations
for model predictions, with special focus on per-group feature importance
to detect discriminatory patterns.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from app.core.logging import logger


def prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Prepare features for SHAP analysis by encoding categorical variables.
    
    Args:
        df: Input dataframe
        feature_columns: Columns to use as features
        exclude_columns: Columns to exclude (e.g., target, predictions)
    
    Returns:
        Tuple of (encoded features dataframe, encoders dict)
    """
    exclude_columns = exclude_columns or []
    encoders = {}
    
    # Filter to only requested feature columns that exist
    available_cols = [c for c in feature_columns if c in df.columns and c not in exclude_columns]
    features_df = df[available_cols].copy()
    
    # Encode categorical variables
    for col in features_df.columns:
        if features_df[col].dtype == 'object' or features_df[col].dtype.name == 'category':
            le = LabelEncoder()
            # Handle missing values
            features_df[col] = features_df[col].fillna('MISSING')
            features_df[col] = le.fit_transform(features_df[col].astype(str))
            encoders[col] = le
        else:
            # Fill numeric missing values
            features_df[col] = features_df[col].fillna(features_df[col].median())
    
    return features_df, encoders


def train_surrogate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
) -> Any:
    """
    Train a surrogate model for SHAP explanation.
    
    When the actual model is not available (e.g., black-box API),
    we train a surrogate on the predictions to explain them.
    
    Args:
        X: Feature matrix
        y: Target/prediction values
        model_type: Type of surrogate model to train
    
    Returns:
        Trained model
    """
    y_binary = _to_binary_series(y)
    
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    model.fit(X, y_binary)
    logger.info(f"Trained surrogate {model_type} with accuracy: {model.score(X, y_binary):.3f}")
    return model


def compute_shap_values(
    X: pd.DataFrame,
    model: Any,
    sample_size: int = 100,
) -> Tuple[np.ndarray, shap.Explainer]:
    """
    Compute SHAP values for feature importance.
    
    Args:
        X: Feature matrix
        model: Trained model to explain
        sample_size: Number of samples to use for background data
    
    Returns:
        Tuple of (shap_values array, explainer object)
    """
    # Use a sample of data as background
    background_size = min(sample_size, len(X))
    background_data = X.sample(n=background_size, random_state=42) if len(X) > background_size else X
    
    # Create TreeExplainer for tree-based models
    if hasattr(model, 'estimators_'):
        explainer = shap.TreeExplainer(model)
    else:
        # Fallback to KernelExplainer for other models
        explainer = shap.KernelExplainer(model.predict_proba, background_data)
    
    shap_values = explainer.shap_values(X)
    
    # Handle multi-class output (return values for positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    return shap_values, explainer


def analyze_feature_importance_by_group(
    df: pd.DataFrame,
    model: Any,
    sensitive_column: str,
    feature_columns: List[str],
    exclude_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze feature importance separately for each demographic group.
    
    This helps identify if certain features drive predictions differently
    for different groups, which can indicate discriminatory patterns.
    
    Args:
        df: Input dataframe with features and sensitive attributes
        model: Trained model to explain
        sensitive_column: Column with demographic groups
        feature_columns: List of feature column names
        exclude_columns: Columns to exclude from analysis
    
    Returns:
        Dict with per-group and comparative feature importance
    """
    X, encoders = prepare_features(df, feature_columns, exclude_columns)
    
    results = {
        "sensitive_attribute": sensitive_column,
        "groups": {},
        "comparative_analysis": {},
        "potential_bias_indicators": [],
    }
    
    # Overall feature importance
    shap_values, explainer = compute_shap_values(X, model)
    overall_importance = np.abs(shap_values).mean(axis=0)
    feature_names = X.columns.tolist()
    
    results["overall_top_features"] = [
        {"feature": name, "importance": float(imp)}
        for name, imp in sorted(
            zip(feature_names, overall_importance),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    ]
    
    # Per-group analysis
    for group_name, group_df in df.groupby(sensitive_column):
        if len(group_df) < 10:  # Skip small groups
            continue
            
        group_indices = group_df.index
        group_X = X.loc[group_indices]
        
        group_shap = shap_values[group_indices.values - X.index.min()]
        group_importance = np.abs(group_shap).mean(axis=0)
        
        results["groups"][str(group_name)] = {
            "sample_size": len(group_df),
            "top_features": [
                {"feature": name, "importance": float(imp)}
                for name, imp in sorted(
                    zip(feature_names, group_importance),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            ],
            "mean_shap_values": {
                name: float(val) for name, val in zip(feature_names, group_shap.mean(axis=0))
            },
        }
    
    # Detect divergent feature importance (potential bias indicator)
    if len(results["groups"]) >= 2:
        results["comparative_analysis"] = _compare_group_importance(
            results["groups"], feature_names
        )
        results["potential_bias_indicators"] = _detect_bias_indicators(
            results["groups"], feature_names
        )
    
    return results


def explain_predictions(
    df: pd.DataFrame,
    model: Any,
    prediction_col: str,
    feature_columns: List[str],
    sample_indices: Optional[List[int]] = None,
    num_samples: int = 10,
) -> Dict[str, Any]:
    """
    Generate SHAP explanations for individual predictions.
    
    Args:
        df: Dataframe with features and predictions
        model: Trained model
        prediction_col: Column with model predictions
        feature_columns: List of feature columns
        sample_indices: Specific row indices to explain (optional)
        num_samples: Number of random samples if indices not provided
    
    Returns:
        Dict with explanations for each sample
    """
    X, encoders = prepare_features(df, feature_columns, exclude_columns=[prediction_col])
    
    # Select samples to explain
    if sample_indices is None:
        sample_indices = np.random.choice(
            len(X), 
            size=min(num_samples, len(X)), 
            replace=False
        ).tolist()
    
    # Compute SHAP values
    shap_values, explainer = compute_shap_values(X, model)
    
    explanations = []
    for idx in sample_indices:
        if idx >= len(X):
            continue
            
        row_shap = shap_values[idx]
        row_features = X.iloc[idx]
        
        # Get top contributing features
        contributions = [
            {"feature": name, "contribution": float(val)}
            for name, val in zip(X.columns, row_shap)
        ]
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        
        explanations.append({
            "index": int(idx),
            "prediction": float(df.iloc[idx][prediction_col]),
            "top_positive_features": [c for c in contributions if c["contribution"] > 0][:5],
            "top_negative_features": [c for c in contributions if c["contribution"] < 0][:5],
            "baseline_value": float(explainer.expected_value if hasattr(explainer, 'expected_value') else 0),
            "explanation_summary": _generate_explanation_summary(contributions[:3]),
        })
    
    return {
        "total_samples": len(X),
        "explained_indices": sample_indices,
        "explanations": explanations,
        "feature_names": X.columns.tolist(),
    }


def _to_binary_series(series: pd.Series) -> pd.Series:
    """Convert series to binary values."""
    if series.dtype == 'bool':
        return series.astype(int)
    
    if pd.api.types.is_numeric_dtype(series):
        unique = sorted(series.dropna().unique())
        if set(unique).issubset({0, 1, 0.0, 1.0}):
            return series.fillna(0).astype(int)
        # Threshold at median
        median = series.median()
        return (series >= median).astype(int)
    
    # String conversion
    lowered = series.astype(str).str.strip().str.lower()
    positive = {"1", "true", "yes", "y", "approved", "accept", "accepted", "selected", "hired", "positive"}
    return lowered.isin(positive).astype(int)


def _compare_group_importance(
    groups: Dict[str, Any],
    feature_names: List[str],
) -> Dict[str, Any]:
    """Compare feature importance across groups to find divergences."""
    group_names = list(groups.keys())
    comparisons = {}
    
    for feature in feature_names:
        importances = {
            group: data["mean_shap_values"].get(feature, 0)
            for group, data in groups.items()
        }
        
        if len(importances) >= 2:
            values = list(importances.values())
            max_diff = max(values) - min(values)
            
            comparisons[feature] = {
                "per_group": importances,
                "max_difference": float(max_diff),
                "ratio": float(max(values) / (min(values) + 1e-9)),
            }
    
    # Find features with highest divergence
    divergent = sorted(
        comparisons.items(),
        key=lambda x: x[1]["max_difference"],
        reverse=True
    )[:5]
    
    return {
        "most_divergent_features": [
            {"feature": name, **data} for name, data in divergent
        ],
        "all_comparisons": comparisons,
    }


def _detect_bias_indicators(
    groups: Dict[str, Any],
    feature_names: List[str],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Detect features that may indicate bias based on importance divergence."""
    indicators = []
    
    for feature in feature_names:
        importances = [
            data["mean_shap_values"].get(feature, 0)
            for data in groups.values()
        ]
        
        if len(importances) >= 2:
            max_imp = max(importances)
            min_imp = min(importances)
            
            # Flag if one group has high importance and another very low
            if max_imp > threshold and min_imp < threshold / 4:
                indicators.append({
                    "feature": feature,
                    "indicator_type": "divergent_importance",
                    "max_group_importance": float(max_imp),
                    "min_group_importance": float(min_imp),
                    "ratio": float(max_imp / (min_imp + 1e-9)),
                    "explanation": (
                        f"'{feature}' significantly influences predictions for one group "
                        f"(importance: {max_imp:.3f}) but not another (importance: {min_imp:.3f}). "
                        f"This may indicate the model uses this feature differently across groups."
                    ),
                    "severity": "HIGH" if max_imp > 1.0 else "MODERATE",
                })
    
    return indicators


def _generate_explanation_summary(top_contributions: List[Dict]) -> str:
    """Generate human-readable explanation summary."""
    parts = []
    for contrib in top_contributions:
        direction = "increased" if contrib["contribution"] > 0 else "decreased"
        parts.append(f"'{contrib['feature']}' {direction} the score")
    
    if parts:
        return "This prediction was primarily influenced by: " + "; ".join(parts) + "."
    return "No significant feature contributions identified."
