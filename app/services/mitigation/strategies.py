from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from app.core.logging import logger

def compute_reweighting_weights(
    dataframe: pd.DataFrame,
    target_column: str,
    sensitive_column: str,
) -> pd.Series:
    """
    Compute sample weights to balance representation across sensitive groups and target outcomes
    using the Kamiran/Calders reweighting method.
    """
    df = dataframe.copy()
    total = len(df)
    
    # Compute group sizes
    counts = df.groupby([sensitive_column, target_column]).size()
    p_s = df[sensitive_column].value_counts() / total
    p_y = df[target_column].value_counts() / total
    
    weights = []
    for _, row in df.iterrows():
        s_val = row[sensitive_column]
        y_val = row[target_column]
        
        # Expected probability assuming independence
        p_expected = p_s[s_val] * p_y[y_val]
        # Observed probability in data
        p_observed = counts.get((s_val, y_val), 0) / total
        
        # Weight = Expected / Observed
        weight = p_expected / p_observed if p_observed > 0 else 1.0
        weights.append(weight)
        
    return pd.Series(weights, index=df.index)

def apply_active_mitigation(
    dataframe: pd.DataFrame,
    target_column: str,
    sensitive_column: str,
    method: str
) -> pd.DataFrame:
    """Apply active bias mitigation to the dataset."""
    logger.info(f"Applying {method} mitigation for {sensitive_column}")
    df_mitigated = dataframe.copy()
    
    if method == "reweighing":
        weights = compute_reweighting_weights(df_mitigated, target_column, sensitive_column)
        df_mitigated['fairness_weight'] = weights
    elif method == "dir":
        # Disparate Impact Remover simulation
        df_mitigated['mitigation_applied'] = "Disparate Impact Remover (Simulated)"
    
    return df_mitigated

def suggest_feature_removal(
    proxy_features: List[Dict],
    correlation_threshold: float = 0.7,
) -> List[Dict]:
    """Suggest removing high-correlation proxy features."""
    removals = []
    for proxy in proxy_features:
        if proxy["correlation"] >= correlation_threshold:
            removals.append({
                "feature": proxy["feature"],
                "sensitive_column": proxy["sensitive_column"],
                "correlation": proxy["correlation"],
                "rationale": (
                    f"Feature '{proxy['feature']}' has high correlation ({proxy['correlation']:.2f}) "
                    f"with '{proxy['sensitive_column']}' and should be removed or transformed "
                    f"to prevent indirect bias."
                ),
                "priority": "HIGH" if proxy["correlation"] >= 0.8 else "MEDIUM",
            })
    return removals

def generate_preprocessing_recommendations(
    dataframe: pd.DataFrame,
    sensitive_columns: List[str],
    fairness_metrics: Dict[str, Dict],
) -> List[Dict]:
    """Generate data preprocessing recommendations to reduce bias."""
    steps = []
    
    # Check for missing values in sensitive columns
    for col in sensitive_columns:
        missing_pct = dataframe[col].isna().sum() / len(dataframe) * 100
        if missing_pct > 0:
            steps.append({
                "type": "missing_values",
                "column": col,
                "issue": f"{missing_pct:.1f}% missing values",
                "recommendation": (
                    f"Impute missing values in '{col}' using median/mode or "
                    f"create a separate 'unknown' category to avoid bias from exclusion."
                ),
                "priority": "MEDIUM" if missing_pct < 5 else "HIGH",
            })
            
    return steps
