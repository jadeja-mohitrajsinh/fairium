"""Explainable AI (XAI) services for bias analysis."""

from app.services.xai.shap_explainer import (
    explain_predictions,
    analyze_feature_importance_by_group,
    compute_shap_values,
)
from app.services.xai.counterfactual import (
    generate_counterfactuals,
    find_minimum_changes,
)

__all__ = [
    "explain_predictions",
    "analyze_feature_importance_by_group",
    "compute_shap_values",
    "generate_counterfactuals",
    "find_minimum_changes",
]
